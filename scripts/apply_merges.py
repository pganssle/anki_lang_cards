"""Carry out the decisions made in triage and write an importable CSV.

    python scripts/apply_merges.py \
        --exported-deck input/existing_deck_3.txt \
        --inputs output/diff_from_existing.csv

Rewrites the notes that were edited, combines the groups answered "merge", and
writes the result to output/merged_import.csv. The input CSV is left alone: this
produces a new file to look at before anything reaches Anki.

Deciding that two notes are one card does not decide what the card reads, so
groups whose definitions disagree go to the merge UI. Groups whose notes are
already identical do not — there is only one thing they can say. Previously
resolved merges come from the merge cache and are not asked about again.
Passing --auto combines everything mechanically instead, which is only sensible
if nobody is going to read the result.

Merging repeats until nothing more changes, because one decision can create work
for the next. Two notes rewritten to say the same thing are combined without
being asked about again; anything else that newly collides is reported so it can
go back through triage.

Where a merge keeps one note's rank and discards another that Anki already has,
the leftover note is listed. Importing cannot delete it — that has to be done in
Anki, and scripts/find_duplicates.py is the tool for it.
"""

import argparse
import asyncio
import csv
import pathlib
from collections.abc import Sequence

from anki_lang_cards import _types as types
from anki_lang_cards import (
    anki_import,
    apply_merges,
    conflict_resolver,
    definition_similarity,
    triage,
)

try:
    from anki_lang_cards import word_merger
except ImportError:  # pragma: no cover - depends on the merge_ui extra
    MERGE_UI_AVAILABLE = False
else:
    MERGE_UI_AVAILABLE = True

DEFAULT_DECISION_CACHE = pathlib.Path(".cache/merges/decisions.json")
DEFAULT_MERGE_CACHE = pathlib.Path(".cache/merges/duplicates.json")
DEFAULT_OUTPUT = pathlib.Path("output/merged_import.csv")


def make_resolver(merge_cache: conflict_resolver.MergeCache) -> apply_merges.Resolver:
    """Settle each group's wording in the merge UI, reusing anything cached."""

    def resolve(
        groups: Sequence[Sequence[types.Word]],
    ) -> Sequence[types.Word | None]:
        print(f"\n  {len(groups)} groups need their wording settled...")
        resolved = asyncio.run(
            word_merger.merge_words(
                [frozenset(group) for group in groups],
                cache=merge_cache,
                cache_behavior=word_merger.CacheHitBehavior.SKIP,
            )
        )
        merge_cache.save_cache()
        return resolved

    return resolve


def write_csv(
    path: pathlib.Path, words: Sequence[types.Word], deck_name: str | None
) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        handle.write("\n".join(types.Word.make_anki_directives(deck_name)) + "\n")
        csv.writer(handle).writerows(
            word.to_csv_line(require_rank=True) for word in words
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--inputs", type=pathlib.Path, nargs="*", default=())
    parser.add_argument("--exported-deck", type=pathlib.Path, default=None)
    parser.add_argument(
        "--decision-cache", type=pathlib.Path, default=DEFAULT_DECISION_CACHE
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--deck-name", default=None)
    parser.add_argument("--merge-cache", type=pathlib.Path, default=DEFAULT_MERGE_CACHE)
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Combine definitions mechanically instead of opening the merge UI",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without writing the CSV",
    )
    args = parser.parse_args()

    deck: list[types.Word] = []
    if args.exported_deck is not None:
        deck.extend(anki_import.load_anki_words(args.exported_deck))
        print(f"Loaded {len(deck)} words from {args.exported_deck}")
    added: list[types.Word] = []
    for path in args.inputs:
        from_csv = types.Word.words_from_csv(path)
        print(f"Loaded {len(from_csv)} words from {path}")
        added.extend(from_csv)
    if not added:
        raise SystemExit("Nothing to apply; pass --inputs.")

    words = (*deck, *added)
    decisions = triage.DecisionCache(args.decision_cache)
    print(f"Read {len(decisions)} decisions from {args.decision_cache}")

    if not args.auto and not MERGE_UI_AVAILABLE:
        raise SystemExit(
            "The merge UI needs textual; install the merge_ui extra, or pass "
            "--auto to combine definitions without being asked."
        )

    resolver = None
    if not args.auto:
        resolver = make_resolver(conflict_resolver.MergeCache(args.merge_cache))

    comparer = definition_similarity.DefinitionComparer(w.meaning for w in words)
    outcome = apply_merges.apply_decisions(
        words, decisions, comparer, resolver=resolver
    )

    print()
    for line in apply_merges.summarize(outcome):
        print(f"  {line}")

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

    orphans = apply_merges.superseded_notes(outcome, deck)
    if orphans:
        print("\nNotes Anki will still hold that nothing will update — delete these:")
        for word in orphans:
            print(f"  {word.rank} {word.word} ({word.part_of_speech}) {word.meaning}")

    if outcome.declined:
        print("\nLeft unmerged in the merge UI:")
        for candidate in outcome.declined:
            for line in candidate.summarize():
                print(f"  {line}")

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

    mine = apply_merges.new_collisions(outcome, added)
    if mine:
        print("\nNew collisions this import created, for another triage pass:")
        for candidate in mine:
            print(f"  [{candidate.reason.value}]")
            for line in candidate.summarize():
                print(f"      {line}")

    to_write = apply_merges.words_to_import(outcome, added)
    if args.dry_run:
        print(f"\nWould write {len(to_write)} notes to {args.output}")
        return
    write_csv(args.output, to_write, args.deck_name)
    print(f"\nWrote {len(to_write)} notes to {args.output}")
    print(f"{args.inputs[0] if args.inputs else 'The input'} is unchanged.")


if __name__ == "__main__":
    main()
