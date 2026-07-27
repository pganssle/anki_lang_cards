"""Find duplicate notes across the whole deck and fix them in Anki.

    python scripts/find_duplicates.py

This is the one tool that writes to Anki: for each duplicate group it updates
the surviving note with the merged content, deletes the superseded notes, and
unsuspends the survivor if studying had moved to a duplicate. Everything else
about finding and answering duplicate questions is the shared flow — the same
screen, decision cache and decision-file protocol as the import scripts, so a
group answered anywhere is answered here too, including "keep separate".

Safety rails carried over from the original: a note whose Image or Note field
would be lost by deletion stops that group (DataLossError); a Freq value on a
superseded note moves to the survivor; each group is confirmed interactively
unless --accept is passed; and --dry-run plans without touching Anki.

Splits are recorded like any other answer but not applied here — creating
notes is an import's job — so split groups are reported for the import
workflow instead.
"""

import argparse
import asyncio
import csv
import io
import pathlib
import sys
from collections.abc import Collection, Mapping, Sequence
from typing import NewType, TypedDict

from anki_lang_cards import _types as types
from anki_lang_cards import (
    anki_import,
    apply_merges,
    conflict_resolver,
    dedup_flow,
    triage,
)

try:
    import textual  # noqa: F401
except ImportError:  # pragma: no cover - depends on the merge_ui extra
    UI_AVAILABLE = False
else:
    UI_AVAILABLE = True

Rank = NewType("Rank", int)

DEFAULT_DECISION_CACHE = pathlib.Path(".cache/merges/decisions.json")
DEFAULT_MERGE_CACHE = pathlib.Path(".cache/merges/duplicates.json")
DEFAULT_PENDING = pathlib.Path("output/deck_duplicates.json")
DEFAULT_DECK_NAME = "Spanish::Vocab by Frequency"

EXIT_DECISIONS_NEEDED = 3


class DataLossError(ValueError): ...


class ActionSet(TypedDict, total=False):
    unsuspends: Collection[anki_import.CardId]
    replacements: Mapping[
        anki_import.NoteId, types.Word | tuple[types.Word, Mapping[str, str]]
    ]
    deletions: Collection[anki_import.NoteId]

    # These are for display / convenience
    notes_to_ranks: Mapping[anki_import.NoteId, Rank]
    cards_to_notes: Mapping[anki_import.CardId, anki_import.NoteId]


def words_to_csv(words: Sequence[types.Word]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(types.Word.make_csv_header())

    for word in words:
        writer.writerow(word.to_csv_line(require_rank=True))

    output.seek(0)
    return output.read()


async def fetch_all_notes(
    deck_name: str, note_cache: pathlib.Path | None
) -> Sequence[types.Word]:
    """Fetch every note in the deck as a Word, reusing unchanged ones."""
    async with anki_import.create_client() as client:
        note_ids = await client.get_note_ids(deck_name, '"Rank:re:.*"')
        print(f"  Retrieving details for {len(note_ids)} notes...")
        cache = None if note_cache is None else anki_import.NoteCache(note_cache)
        notes = await anki_import.fetch_note_words(
            client, note_ids, cache=cache, description="Fetching deck notes"
        )
    return list(notes.values())


async def prepare_duplicate_merge(
    deck_name: str, merge: apply_merges.Merge
) -> ActionSet:
    """Plan the Anki-side work for one merge: update, delete, unsuspend."""
    ranks = sorted(
        {w.rank for w in (merge.result, *merge.sources) if w.rank is not None}
    )
    merged = merge.result
    async with anki_import.create_client() as client:
        note_ids = await client.get_note_ids(
            deck_name,
            query="(" + " or ".join(f"Rank:{rank}" for rank in ranks) + ")",
        )
        rank_cards: dict[int, anki_import.CardId] = {}
        card_notes: dict[anki_import.CardId, anki_import.NoteId] = {}
        note_data: dict[anki_import.NoteId, anki_import.NoteData] = {}
        async for note in client.get_anki_notes(note_ids):
            note_data[note["noteId"]] = note
            rank = int(note["fields"]["Rank"]["value"])
            card_id = int(note["cards"][0])
            rank_cards[rank] = card_id
            card_notes[card_id] = note["noteId"]

        cards_suspended = await client.are_suspended(list(rank_cards.values()))

    # Figure out which note will be the target of the merge. If the card with
    # the target rank is suspended, we will assume that we added the duplicate
    # and started studying it, so we should merge into the one we've been
    # studying: the lowest-ranked note that isn't suspended (this all assumes
    # one card per note).
    if not cards_suspended[rank_cards[merged.rank]]:
        target = merged.rank
    else:
        for rank in sorted(rank_cards):
            if not cards_suspended[rank_cards[rank]]:
                target = rank
                break
        else:
            target = merged.rank

    target_card = rank_cards[target]

    # Prepare a plan of action — what to delete, what to replace, etc
    deletions = {
        card_notes[card_id] for card_id in rank_cards.values() if card_id != target_card
    }

    data_loss = {}
    freq_data: str | None = None
    for deletion in deletions:
        deletion_freq = note_data[deletion]["fields"]["Freq"]["value"]
        if deletion_freq and freq_data:
            raise ValueError("Frequency specified more than once!")
        elif deletion_freq:
            freq_data = deletion_freq
        for lossy_field in ("Image", "Note"):
            if note_data[deletion]["fields"][lossy_field]["value"] != "":
                if deletion not in data_loss:
                    data_loss[deletion] = []
                data_loss[deletion].append(lossy_field)

    if data_loss:
        for note_id, losses in data_loss.items():
            rank = note_data[note_id]["fields"]["Rank"]
            raise DataLossError(
                f"Data loss in card with rank {rank} ({note_id})\n"
                + f"  lossy fields: {', '.join(losses)}"
            )
    if cards_suspended[target_card]:
        unsuspends = [target_card]
    else:
        unsuspends = []

    replacements = {
        card_notes[target_card]: (
            merged if not freq_data else (merged, {"Freq": freq_data})
        )
    }

    return {
        "unsuspends": unsuspends,
        "replacements": replacements,
        "deletions": deletions,
        "notes_to_ranks": {
            card_notes[card_id]: rank for rank, card_id in rank_cards.items()
        },
        "cards_to_notes": card_notes,
    }


async def execute_plan(action_sets: Sequence[ActionSet]) -> None:
    # TODO: Possibly batch these
    for action_set in action_sets:
        async with anki_import.create_client() as client:
            # Start with unsuspending cards, the least dangerous operation
            if action_set["unsuspends"]:
                await client.unsuspend_cards(action_set["unsuspends"])

            # Next replace the card we want to merge into
            await client.update_note_info_from_words(action_set["replacements"])

            # # Now delete all other duplicates
            await client.delete_notes(action_set["deletions"])


async def apply_merge_to_anki(
    deck_name: str,
    merge: apply_merges.Merge,
    *,
    accept: bool,
    dry_run: bool,
    verbose: bool,
) -> bool:
    """Plan and (maybe) carry out one merge. Returns False to stop the run."""
    try:
        action_set = await prepare_duplicate_merge(deck_name, merge)
    except DataLossError as error:
        print(f"Skipping {merge.result.word} due to potential data loss:\n  {error}")
        return True

    num_notes = len(action_set["replacements"]) + len(action_set["deletions"])
    if num_notes <= 1:
        print(f"Skipping {merge.result.word} (no duplicate notes in Anki)")
        return True
    print(f"{merge.result.word}: Merging {num_notes} duplicate notes")

    deleted_ranks = [
        action_set["notes_to_ranks"][note] for note in action_set["deletions"]
    ]
    replaced_ranks = [
        action_set["notes_to_ranks"][note] for note in action_set["replacements"]
    ]

    if len(replaced_ranks) > 1:
        raise ValueError(f"Replacing too many cards: {replaced_ranks}")
    elif len(replaced_ranks) < 1 and len(deleted_ranks) > 0:
        raise ValueError(f"Deleting cards but no replacements: {action_set}")

    if action_set["unsuspends"]:
        unsuspend_ranks = [
            action_set["notes_to_ranks"][action_set["cards_to_notes"][unsuspend]]
            for unsuspend in action_set["unsuspends"]
        ]
        if len(unsuspend_ranks) != 1:
            raise ValueError(f"Unsuspending too many cards: {unsuspend_ranks}")
        print(f"  Unsuspending {unsuspend_ranks[0]}")
    print(f"  Deleting {', '.join(map(str, deleted_ranks))}")
    print(f"  Merging into {replaced_ranks[0]}")
    if verbose:
        word = merge.result
        print(f"  Merged card: {word.word} ({word.part_of_speech}) ({word.rank})")
        for sentence in word.sentences:
            print(f"    {sentence}")
    print("")

    if dry_run:
        return True
    if not accept:
        confirm = input("Continue with merge? (y)es / (s)kip / (q)uit ")
        if confirm == "s":
            return True
        if confirm == "q":
            return False
        if confirm != "y":
            return True
    await execute_plan([action_set])
    return True


async def amain() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cache-file",
        type=pathlib.Path,
        default=None,
        help="Word dump to read instead of fetching the whole deck; written "
        "after a fresh fetch when given",
    )
    parser.add_argument(
        "--accept",
        action="store_true",
        help="Apply every planned merge without confirming each",
    )
    parser.add_argument("--deck-name", default=DEFAULT_DECK_NAME)
    parser.add_argument(
        "--note-cache",
        type=pathlib.Path,
        default=anki_import.DEFAULT_NOTE_CACHE,
        help="Fetched notes, reused while their mod times are unchanged",
    )
    parser.add_argument(
        "--decision-cache", type=pathlib.Path, default=DEFAULT_DECISION_CACHE
    )
    parser.add_argument(
        "--merge-cache",
        type=pathlib.Path,
        default=DEFAULT_MERGE_CACHE,
        help="Legacy merge-wording cache, consulted for merges decided "
        "before decisions carried their wording",
    )
    parser.add_argument(
        "--pending-file",
        type=pathlib.Path,
        default=DEFAULT_PENDING,
        help="Where unanswered groups go for the no-UI decision loop",
    )
    parser.add_argument(
        "--decision-file",
        type=pathlib.Path,
        default=None,
        help="An answered pending file to fold in before anything else",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Never open a screen: write unanswered groups to the pending "
        "file and exit 3",
    )
    parser.add_argument(
        "--sentences",
        type=int,
        default=0,
        help="Sentences per note per page in the UI (0 fits the pane)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan every merge without changing anything in Anki",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    interactive = not args.no_ui
    if interactive and not UI_AVAILABLE:
        raise SystemExit(
            "The UI needs textual; install the merge_ui extra, or pass "
            "--no-ui to answer through the decision file."
        )

    if args.cache_file is not None and args.cache_file.exists():
        print("Reading from cache file")
        all_words: Sequence[types.Word] = types.Word.words_from_csv(args.cache_file)
    else:
        all_words = await fetch_all_notes(args.deck_name, args.note_cache)
        if args.cache_file is not None:
            args.cache_file.write_text(words_to_csv(all_words))

    decisions = triage.DecisionCache(args.decision_cache)
    print(f"Read {len(decisions)} decisions from {args.decision_cache}")
    if args.decision_file is not None:
        recorded = dedup_flow.record_decision_file(args.decision_file, decisions)
        print(f"Folded in {recorded} new answers from {args.decision_file}")

    result = await dedup_flow.resolve_duplicates(
        (),
        all_words,
        decisions=decisions,
        merge_cache=conflict_resolver.MergeCache(args.merge_cache),
        interactive=interactive,
        per_page=args.sentences or None,
    )

    for line in apply_merges.summarize(result.outcome):
        print(f"  {line}")

    if result.outcome.splits:
        print(
            f"\n{len(result.outcome.splits)} groups were answered as splits. "
            "This script does not create notes; run the splits through the "
            "import workflow (scripts/apply_merges.py) instead."
        )

    applied = 0
    for merge in result.outcome.merges:
        if not merge.superseded:
            continue
        proceed = await apply_merge_to_anki(
            args.deck_name,
            merge,
            accept=args.accept,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        applied += 1
        if not proceed:
            break

    if not applied:
        print("No duplicate groups with notes to merge in Anki.")

    if result.interrupted:
        print(f"\n{len(result.pending)} groups still need an answer.")
        if not interactive:
            dedup_flow.write_pending_file(args.pending_file, result.pending)
            print(
                f'Edit {args.pending_file} (set "disposition") and re-run '
                f"with --decision-file {args.pending_file}"
            )
            return EXIT_DECISIONS_NEEDED
        print("Answers made so far were saved; re-run to continue.")
        return 1
    return 0


def main() -> None:
    sys.exit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
