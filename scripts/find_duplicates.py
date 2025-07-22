import argparse
import asyncio
import csv
import io
import itertools
import pathlib
from collections.abc import Collection, Mapping, Sequence
from typing import NewType, TypedDict

from anki_lang_cards import _types as types
from anki_lang_cards import anki_import, conflict_resolver, word_merger

Rank = NewType("Rank", int)


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
    deck_name: str,
) -> Sequence[types.Word]:
    """
    Fetches all notes from the specified Anki deck, organizes them by rank,
    and finds the highest rank.

    Returns:
        A tuple containing:
        - A dict mapping rank numbers to a list of note IDs.
        - The highest rank number found.
        - A dict mapping every note ID to its rank.
    """
    out = []
    async with anki_import.create_client() as client:
        note_match = ".*"
        note_ids = await client.get_note_ids(deck_name, f'"Rank:re:{note_match}"')

        print(f"  Retrieving details for {len(note_ids)} notes...")
        async with anki_import.create_client() as client:
            async for note in client.get_anki_notes(note_ids):
                word = client.word_from_anki_connect_response(note)
                out.append(word)
    return out


async def prepare_duplicate_merge(
    deck_name: str, merged: types.Word, word_lookup: conflict_resolver.WordLookup
) -> ActionSet:
    async with anki_import.create_client() as client:
        try:
            note_ids = await client.get_note_ids(
                deck_name,
                query="("
                + " or ".join(f"Rank:{word.rank}" for word in word_lookup.lookup(merged))
                + ")",
            )
        except:
            breakpoint()
            raise
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

    # Figure out which note will be the target of the merge. If the card with the target
    # rank is suspended, we will assume that we added the duplicate and started studying
    # it, so we should merge into the one we've been studying. We'll merge into the
    # lowest rank note that isn't suspended (this all assumes one card per note)
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


def merge_action_sets(action_sets: Sequence[ActionSet]) -> ActionSet:
    output = {
        "unsuspends": [],
        "replacements": {},
        "deletions": [],
        "notes_to_ranks": {},
        "cards_to_notes": {},
    }

    for action_set in action_sets:
        output["unsuspends"] = output["unsuspends"].extend(action_set["unsuspends"])
        output["replacements"] = {
            **output["replacements"],
            **action_set["replacements"],
        }
        output["deletions"] = output["deletions"].extend(action_set["deletions"])

    return output


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


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find and fix duplicate and missing ranks in an Anki deck.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--cache-file",
        type=pathlib.Path,
        default=None,
        help="First place to look for gaps or duplicates",
    )

    parser.add_argument(
        "--accept",
        action="store_true",
        help="Accept all merges automatically",
    )

    parser.add_argument(
        "--merge-cache",
        type=pathlib.Path,
        default=".cache/merges/duplicates.json",
        help="File caches choices made when resolving merge conflicts",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    deck_name = "Spanish::Vocab by Frequency"

    if args.cache_file is not None and args.cache_file.exists():
        print("Reading from cache file")
        all_words: Sequence[types.Word] = types.Word.words_from_csv(args.cache_file)
    else:
        all_words = await fetch_all_notes(deck_name)
        if args.cache_file is not None:
            args.cache_file.write_text(words_to_csv(all_words))

    all_word_lookup = conflict_resolver.WordLookup(all_words)
    conflicts_only = conflict_resolver.WordLookup(
        itertools.chain.from_iterable(
            group for group in all_word_lookup if len(group) > 1
        )
    )

    merge_cache = conflict_resolver.MergeCache(args.merge_cache)
    merge_results = await word_merger.merge_words(
        tuple(conflicts_only),
        cache=merge_cache,
        cache_behavior=word_merger.CacheHitBehavior.SKIP,
    )

    merge_plans: dict[types.Word, ActionSet] = {}
    for merged in merge_results:
        if merged is None:
            continue

        # TODO: Use an executor to do these in parallel
        try:
            action_set = await prepare_duplicate_merge(
                deck_name, merged, conflicts_only
            )
        except DataLossError:
            print(f"Skipping {merged.word} due to potential data loss")
            continue
        merge_plans[merged] = action_set

        # Explain the plan
        word = merged
        num_notes = len(action_set["replacements"]) + len(action_set["deletions"])
        if num_notes > 1:
            print(f"{word.word}: Merging {num_notes} duplicate notes")
        else:
            print(f"Skipping {word.word} (no duplicates)")
            continue

        deleted_ranks = [
            action_set["notes_to_ranks"][note] for note in action_set["deletions"]
        ]

        replaced_ranks = [
            action_set["notes_to_ranks"][note]
            for note in action_set["replacements"].keys()
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
        if args.verbose:
            print(f"  Merged card: {word.word} ({word.part_of_speech}) ({word.rank})")
            for sentence in word.sentences:
                print(f"    {sentence}")
        print("")

        if not args.accept:
            confirm = input("Continue with merge? (y)es / (s)kip / (q)uit ")
            if confirm == "s":
                continue

            if confirm == "q":
                break

            if confirm == "y":
                await execute_plan([action_set])
        else:
            await execute_plan([action_set])


class DataLossError(ValueError): ...


if __name__ == "__main__":
    asyncio.run(main())
