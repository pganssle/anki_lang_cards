"""Find and merge duplicates between the deck and an import, in one sitting.

    python scripts/apply_merges.py --inputs output/diff_from_existing.csv

The deck comes from Anki Connect by default (Anki has to be running with the
add-on installed); pass --exported-deck to work from a text export instead.

The flow asks about every flagged group once — merge, keep separate, edit, or
split, with the proposed merged note on screen — and carries the answers out,
looping back when a merge creates a new collision. Every answer is saved the
moment it is given, so quitting midway loses nothing: re-run and it picks up
at the first unanswered question.

On completion it writes the merged import CSV and, separately, the existing
notes whose content merged into a different rank — importing cannot delete
those, so they are listed for deletion by hand (scripts/find_duplicates.py is
the tool that deletes). The input CSV is never modified.

Without a screen: --auto accepts every proposed merge mechanically, and
--no-ui answers nothing — it writes the unanswered groups to a decision file
and exits with code 3. Edit the file (set "disposition"; a merge with no
"replacements" accepts the proposal shown) and re-run; repeat until exit 0.
That path is the whole flow for agentic harnesses and anyone the TUI does not
serve.
"""

import argparse
import asyncio
import csv
import json
import pathlib
import sys
from collections.abc import Sequence

import attrs

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

DEFAULT_DECISION_CACHE = pathlib.Path(".cache/merges/decisions.json")
DEFAULT_MERGE_CACHE = pathlib.Path(".cache/merges/duplicates.json")
DEFAULT_OUTPUT = pathlib.Path("output/merged_import.csv")
DEFAULT_DELETIONS = pathlib.Path("output/notes_to_delete.csv")
DEFAULT_PENDING = pathlib.Path("output/merge_candidates.json")
DEFAULT_DECK_NAME = "Spanish::Vocab by Frequency"

EXIT_DECISIONS_NEEDED = 3
"""The run went as far as recorded answers allow; the pending file says what
is left. Distinct from 1 so a harness can tell "answer and re-run" from "look
at a traceback"."""


def write_import_csv(
    path: pathlib.Path, words: Sequence[types.Word], deck_name: str | None
) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        handle.write("\n".join(types.Word.make_anki_directives(deck_name)) + "\n")
        csv.writer(handle).writerows(
            word.to_csv_line(require_rank=True) for word in words
        )


def write_deletions_csv(path: pathlib.Path, words: Sequence[types.Word]) -> None:
    """The superseded notes, as a plain reference CSV rather than an import."""
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("rank", "word", "part_of_speech", "meaning"))
        writer.writerows(
            (word.rank, word.word, str(word.part_of_speech), word.meaning)
            for word in words
        )


async def load_deck(
    args: argparse.Namespace, added: Sequence[types.Word]
) -> Sequence[types.Word]:
    if args.exported_deck is not None:
        deck = anki_import.load_anki_words(args.exported_deck)
        print(f"Loaded {len(deck)} words from {args.exported_deck}")
        return deck
    print("Loading related notes from Anki Connect")
    async with anki_import.create_client() as client:
        notes = await client.retrieve_related_words(
            args.deck_name, added, cache=anki_import.NoteCache(args.note_cache)
        )
    deck = tuple(notes.values())
    print(f"Loaded {len(deck)} related notes from deck {args.deck_name!r}")
    return deck


async def next_free_rank(args: argparse.Namespace, words: Sequence[types.Word]) -> int:
    """Where fresh ranks start: past everything the deck or this run knows.

    On the live path the loaded notes are only the ones related to the
    inputs, whose ranks say nothing about the deck's highest, so the deck is
    asked directly.
    """
    highest = max((word.rank for word in words if word.rank is not None), default=0)
    if args.exported_deck is None:
        async with anki_import.create_client() as client:
            highest = max(highest, await client.find_highest_rank(args.deck_name))
    return highest + 1


def assign_missing_ranks(
    words: Sequence[types.Word], start: int
) -> Sequence[types.Word]:
    out: list[types.Word] = []
    rank = start
    for word in words:
        if word.rank is None:
            word = attrs.evolve(word, rank=rank)
            rank += 1
        out.append(word)
    return out


def report(
    result: dedup_flow.DedupResult,
    added: Sequence[types.Word],
    deck: Sequence[types.Word],
) -> None:
    outcome = result.outcome
    print()
    for line in apply_merges.summarize(outcome):
        print(f"  {line}")

    if result.rank_only_decisions:
        print(
            f"\nWARNING: {len(result.rank_only_decisions)} recorded decisions "
            "match this run's notes by rank only — the notes' text has "
            "changed since they were answered. Look before trusting these:"
        )
        for group in result.rank_only_decisions:
            for word in sorted(group):
                print(
                    f"  {word.rank} {word.word} ({word.part_of_speech}) "
                    f"{word.meaning}"
                )

    automatic = [merge for merge in outcome.merges if not merge.decided]
    if automatic:
        print("\nCombined without a decision, being identical after rewriting:")
        for merge in automatic:
            ranks = sorted(w.rank for w in merge.sources if w.rank is not None)
            print(f"  {merge.result.rank:>5} {merge.result.word} <- ranks {ranks}")

    renamed = [merge for merge in outcome.merges if merge.dropped_headwords]
    if renamed:
        print("\nOne spelling kept, the other dropped:")
        for merge in renamed:
            print(
                f"  kept {merge.result.word!r}, dropped {list(merge.dropped_headwords)}"
            )

    discarded = apply_merges.discarded_imports(outcome, added, deck)
    if discarded:
        print(
            f"\nWARNING: {len(discarded)} notes came out exactly as the deck already"
            " has them, so the imported version was dropped:"
        )
        for word in discarded:
            print(f"  {word.rank} {word.word} ({word.part_of_speech}) {word.meaning}")
        print(
            "  Check these before importing; the new sentences are not in the output."
        )


def report_pending(result: dedup_flow.DedupResult) -> None:
    print(f"\n{len(result.pending)} groups still need an answer:")
    for candidate in result.pending:
        print(f"  [{candidate.reason.value}]")
        for line in candidate.summarize():
            print(f"      {line}")


def write_report_json(
    path: pathlib.Path,
    result: dedup_flow.DedupResult,
    *,
    status: str,
    files: dict[str, str],
    superseded: int,
) -> None:
    payload = {
        "status": status,
        "passes": result.passes,
        "merged": len(result.outcome.merges),
        "split": len(result.outcome.splits),
        "revised": len(result.outcome.revised),
        "pending": len(result.pending),
        "superseded": superseded,
        "rank_only_decisions": len(result.rank_only_decisions),
        "files": files,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Report written to {path}")


async def amain() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--inputs", type=pathlib.Path, nargs="*", default=())
    parser.add_argument(
        "--exported-deck",
        type=pathlib.Path,
        default=None,
        help="Work from a deck text export instead of live Anki Connect",
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
    parser.add_argument("--merge-cache", type=pathlib.Path, default=DEFAULT_MERGE_CACHE)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--notes-to-delete", type=pathlib.Path, default=DEFAULT_DELETIONS
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
        "--auto",
        action="store_true",
        help="Accept every proposed merge mechanically; nobody is asked",
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
        "--report-json",
        type=pathlib.Path,
        default=None,
        help="Write a machine-readable run summary to this path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without writing any output files",
    )
    args = parser.parse_args()

    added: list[types.Word] = []
    for path in args.inputs:
        from_csv = types.Word.words_from_csv(path)
        print(f"Loaded {len(from_csv)} words from {path}")
        added.extend(from_csv)
    if not added:
        raise SystemExit("Nothing to apply; pass --inputs.")

    interactive = not args.no_ui and not args.auto
    if interactive and not UI_AVAILABLE:
        raise SystemExit(
            "The UI needs textual; install the merge_ui extra, or pass "
            "--no-ui to answer through the decision file, or --auto to "
            "accept every proposal mechanically."
        )

    deck = await load_deck(args, added)
    decisions = triage.DecisionCache(args.decision_cache)
    print(f"Read {len(decisions)} decisions from {args.decision_cache}")
    if args.decision_file is not None:
        recorded = dedup_flow.record_decision_file(args.decision_file, decisions)
        print(f"Folded in {recorded} new answers from {args.decision_file}")

    result = await dedup_flow.resolve_duplicates(
        added,
        deck,
        decisions=decisions,
        merge_cache=conflict_resolver.MergeCache(args.merge_cache),
        interactive=interactive,
        auto=args.auto,
        per_page=args.sentences or None,
    )
    report(result, added, deck)

    if result.interrupted:
        report_pending(result)
        if args.no_ui or args.auto:
            dedup_flow.write_pending_file(args.pending_file, result.pending)
            print(
                f"\nAnswers needed: edit {args.pending_file} (set "
                '"disposition"; a merge with no "replacements" accepts the '
                "proposal), then re-run with "
                f"--decision-file {args.pending_file}"
            )
            if args.report_json is not None:
                write_report_json(
                    args.report_json,
                    result,
                    status="decisions-needed",
                    files={"pending": str(args.pending_file)},
                    superseded=0,
                )
            return EXIT_DECISIONS_NEEDED
        print("\nNothing final was written; re-run to continue where you left off.")
        if args.report_json is not None:
            write_report_json(
                args.report_json,
                result,
                status="interrupted",
                files={},
                superseded=0,
            )
        return 1

    to_write = apply_merges.words_to_import(result.outcome, added)
    if any(word.rank is None for word in to_write):
        start = await next_free_rank(args, (*deck, *result.outcome.words))
        to_write = assign_missing_ranks(to_write, start)

    orphans = apply_merges.superseded_notes(result.outcome, deck)
    if orphans:
        print("\nNotes Anki will still hold that nothing will update — delete these:")
        for word in orphans:
            print(f"  {word.rank} {word.word} ({word.part_of_speech}) {word.meaning}")

    if args.dry_run:
        print(f"\nWould write {len(to_write)} notes to {args.output}")
        if orphans:
            print(f"Would write {len(orphans)} notes to {args.notes_to_delete}")
        return 0

    write_import_csv(args.output, to_write, args.deck_name)
    print(f"\nWrote {len(to_write)} notes to {args.output}")
    files = {"import": str(args.output)}
    if orphans:
        write_deletions_csv(args.notes_to_delete, orphans)
        print(f"Wrote {len(orphans)} notes to delete to {args.notes_to_delete}")
        files["notes_to_delete"] = str(args.notes_to_delete)
    print(f"{args.inputs[0] if args.inputs else 'The input'} is unchanged.")
    if args.report_json is not None:
        write_report_json(
            args.report_json,
            result,
            status="complete",
            files=files,
            superseded=len(orphans),
        )
    return 0


def main() -> None:
    sys.exit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
