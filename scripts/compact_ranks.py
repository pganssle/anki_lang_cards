import argparse
import asyncio
import itertools
import json
import pathlib
from collections.abc import Mapping, Sequence

from anki_lang_cards import anki_import


async def fetch_all_notes(
    deck_name: str,
    start_at: int,
) -> tuple[Mapping[int, Sequence[int]], int, Mapping[int, int]]:
    """
    Fetches all notes from the specified Anki deck, organizes them by rank,
    and finds the highest rank.

    Returns:
        A tuple containing:
        - A dict mapping rank numbers to a list of note IDs.
        - The highest rank number found.
        - A dict mapping every note ID to its rank.
    """
    ranks_to_notes: dict[int, list[int]] = {}
    noteid_to_rank: dict[int, int] = {}
    highest_rank: int = 0
    tidx: int = start_at // 1000

    print("Stage 1: Fetching all note ranks from Anki...")
    while True:
        async with anki_import.create_client() as client:
            # Fetches notes in batches of 1000 ranks to handle large decks
            if tidx == 0:
                note_match = "^([0-9][0-9][0-9]|[0-9][0-9]|[0-9])$"
            else:
                note_match = f"{tidx}[0-9][0-9][0-9]"
            note_ids = await client.get_note_ids(deck_name, f'"Rank:re:{note_match}"')

        # Stop searching a few empty blocks after the last found card
        if not note_ids:
            if tidx > (highest_rank // 1000) + 2:
                break
            tidx += 1
            continue

        print(f"  Retrieving details for {len(note_ids)} notes...")
        async with anki_import.create_client() as client:
            async for note in client.get_anki_notes(note_ids):
                try:
                    rank = int(note["fields"]["Rank"]["value"])
                    note_id = note["noteId"]
                    if rank not in ranks_to_notes:
                        ranks_to_notes[rank] = []
                    ranks_to_notes[rank].append(note_id)
                    noteid_to_rank[note_id] = rank
                    highest_rank = max(highest_rank, rank)
                except (ValueError, KeyError):
                    print(
                        f"⚠️ Warning: Could not parse rank for note ID {note['noteId']}. Skipping."
                    )
        tidx += 1

    print(f"✅ Fetch complete. Found notes up to rank {highest_rank}.")
    return ranks_to_notes, highest_rank, noteid_to_rank


def create_move_plan(
    ranks_to_notes: Mapping[int, Sequence[int]],
    highest_rank: int,
    noteid_to_rank: Mapping[int, int],
    start_at: int,
) -> dict[int, int]:
    """
    Analyzes ranks to find duplicates and gaps, then creates a plan to fix them.

    The plan is created in two phases:
    1. Duplicates are moved into gaps.
    2. High-ranked cards are moved into any remaining gaps (compaction).

    Returns:
        A dictionary mapping the note ID to be moved to its new target rank.
    """
    print("\nStage 2: Analyzing ranks and creating a move plan...")

    # 1. Identify duplicates and gaps
    duplicates_to_move: list[int] = []
    for note_ids in ranks_to_notes.values():
        if len(note_ids) > 1:
            duplicates_to_move.extend(note_ids[1:])

    present_ranks = set(ranks_to_notes.keys()) | set(range(start_at))
    all_possible_ranks = set(range(1, highest_rank + 1))
    missing_ranks = sorted(all_possible_ranks - present_ranks)

    print(f"🔍 Found {len(duplicates_to_move)} duplicate rank(s).")
    print(f"🔍 Found {len(missing_ranks)} gap(s) in the sequence.")

    move_plan: dict[int, int] = {}  # note_id -> new_rank

    # 2. Phase 1: Move duplicates into gaps
    gaps_iter = iter(missing_ranks)
    for dupe_note_id, target_rank in zip(
        duplicates_to_move,
        itertools.chain(gaps_iter, itertools.count(highest_rank + 1)),
    ):
        move_plan[dupe_note_id] = target_rank

    # 3. Phase 2: Compact ranks by moving cards from the end into remaining gaps
    remaining_gaps = list(gaps_iter)
    if not remaining_gaps:
        return move_plan

    final_card_count = len(present_ranks)
    notes_to_compact: list[int] = []
    # Find notes with ranks higher than the ideal total, sorted descending
    for rank in sorted(ranks_to_notes.keys(), reverse=True):
        if rank <= final_card_count:
            break
        for note_id in ranks_to_notes[rank]:
            if note_id not in move_plan:
                notes_to_compact.append(note_id)

    # Match high-ranked notes to remaining gaps
    for note_to_move, target_rank in zip(notes_to_compact, remaining_gaps):
        original_rank = noteid_to_rank[note_to_move]
        if target_rank < original_rank:  # Only move if it's beneficial
            move_plan[note_to_move] = target_rank
        else:
            break  # Stop when moves are no longer beneficial

    return move_plan


def print_plan(
    move_plan: Mapping[int, int],
    noteid_to_rank: Mapping[int, int],
) -> None:
    """Prints a human-readable summary of the move plan."""
    print(f"\n📋 Summary of planned changes ({len(move_plan)} notes to move):")
    if not move_plan:
        print("  No moves planned.")
        return

    # Sort by the new rank for a clear view of the result
    sorted_moves = sorted(move_plan.items(), key=lambda item: item[1])

    for note_id, new_rank in sorted_moves:
        old_rank = noteid_to_rank[note_id]
        print(f"  - Move Note ID {note_id}: Rank {old_rank} -> {new_rank}")


async def execute_plan(move_plan: Mapping[int, int]) -> None:
    """Executes the move plan by updating the notes in Anki."""
    print(f"\nStage 3: Executing plan to update {len(move_plan)} notes...")
    updated_notes = []
    for note_id, new_rank in move_plan.items():
        new_note_info = {
            "id": note_id,
            "fields": {"Rank": str(new_rank)},
        }
        updated_notes.append(new_note_info)

    # Update notes in batches for efficiency
    for i, batch in enumerate(itertools.batched(updated_notes, 50)):
        print(f"  Sending batch {i+1}...")
        async with anki_import.create_client() as client:
            await client.update_note_info(list(batch))

    print("✅ Execution complete.")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find and fix duplicate and missing ranks in an Anki deck.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--start-at",
        type=int,
        default=5000,
        help="First place to look for gaps or duplicates",
    )

    parser.add_argument(
        "--cache-file",
        type=pathlib.Path,
        default=None,
        help="First place to look for gaps or duplicates",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what changes would be made without executing them.",
    )
    args = parser.parse_args()

    deck_name = "Spanish::Vocab by Frequency"

    if args.cache_file is not None and args.cache_file.exists():
        print("Reading from cache file")
        ranks_to_notes_str_keys, highest_rank, noteid_to_rank_str_keys, start_at = (
            json.loads(args.cache_file.read_text())
        )

        # JSON requires that all keys be strings.
        ranks_to_notes = {
            int(rank): notes for rank, notes in ranks_to_notes_str_keys.items()
        }
        noteid_to_rank = {
            int(note_id): rank for note_id, rank in noteid_to_rank_str_keys.items()
        }
        cache_valid = start_at == args.start_at
    else:
        cache_valid = False

    if not cache_valid:
        print("Cache miss, loading notes from Anki Connect")
        ranks_to_notes, highest_rank, noteid_to_rank = await fetch_all_notes(
            deck_name, start_at=args.start_at
        )

    if args.cache_file is not None and not cache_valid:
        breakpoint()
        print("Writing to cache file")
        args.cache_file.write_text(
            json.dumps([ranks_to_notes, highest_rank, noteid_to_rank, args.start_at])
        )

    move_plan = create_move_plan(
        ranks_to_notes, highest_rank, noteid_to_rank, start_at=args.start_at
    )

    if not move_plan:
        print("\n🎉 No issues found. Ranks are clean!")
        return

    print_plan(move_plan, noteid_to_rank)

    if args.dry_run:
        print("\n--dry-run enabled. No changes were made to your Anki deck.")
        return

    try:
        confirm = input("\nProceed with these changes? (y/N): ")
        if confirm.lower() == "y":
            await execute_plan(move_plan)
        else:
            print("Operation cancelled by user.")
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled by user.")


if __name__ == "__main__":
    asyncio.run(main())
