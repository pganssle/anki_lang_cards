"""Decide which possible duplicates to merge, before any merging starts.

Run in two passes. The first collects everything that might be a duplicate and
writes it to a file:

    python scripts/triage_duplicates.py \\
        --exported-deck input/existing_deck_3.txt \\
        --inputs output/diff_from_existing.csv

Edit that file — the only field that matters is "disposition", which can be
"merge", "review" or "keep-separate" — and then record the decisions:

    python scripts/triage_duplicates.py --apply

By default a run is about the words being added: groups made up entirely of
notes already in the deck are left out, since two notes that have disagreed
quietly for a year are a real finding but not one this import caused. Pass
--include-existing to go over the whole deck instead.

Anything marked "keep-separate" goes into the rejection cache and stops being
offered on later runs, so a deliberate sense split only has to be dismissed
once. Nothing is written to Anki or to the deck by either pass; this only
decides what the merge UI will later be asked about.

Passing --resolve answers the same questions in a UI instead of an editor, and
records the decisions in one go. It needs the merge_ui extra. By default it
asks only about the groups that have no suggestion; --include-proposed asks
about the proposed merges as well.

Sometimes the answer is neither merging nor separating but rewriting: "segundo"
the adjective and "segundo" the noun collide only because both are written the
same way, and "segundo/a" against "el segundo" is a better outcome than either.
Pressing "e" in the UI edits the notes in a group. Rewritten notes are written
to a CSV to be imported into Anki, since nothing here changes the deck itself.

    python scripts/triage_duplicates.py --resolve \\
        --exported-deck input/existing_deck_3.txt \\
        --inputs output/diff_from_existing.csv
"""

import argparse
import asyncio
import csv
import pathlib
from collections.abc import Sequence

from anki_lang_cards import _types as types
from anki_lang_cards import anki_import, definition_similarity, triage

try:
    from anki_lang_cards import triage_ui
except ImportError:  # pragma: no cover - depends on the merge_ui extra
    TRIAGE_UI_AVAILABLE = False
else:
    TRIAGE_UI_AVAILABLE = True

DEFAULT_TRIAGE_FILE = pathlib.Path("output/merge_candidates.json")
DEFAULT_REJECTION_CACHE = pathlib.Path(".cache/merges/rejections.json")
DEFAULT_REVISIONS_FILE = pathlib.Path("output/triage_revisions.csv")


def load_words(
    inputs: Sequence[pathlib.Path], exported_deck: pathlib.Path | None
) -> tuple[Sequence[types.Word], Sequence[types.Word]]:
    """Load the deck and the words being added, keeping the two apart."""
    deck: list[types.Word] = []
    if exported_deck is not None:
        deck.extend(anki_import.load_anki_words(exported_deck))
        print(f"Loaded {len(deck)} words from {exported_deck}")

    added: list[types.Word] = []
    for path in inputs:
        from_csv = types.Word.words_from_csv(path)
        print(f"Loaded {len(from_csv)} words from {path}")
        added.extend(from_csv)
    return tuple(deck), tuple(added)


def report(candidates: Sequence[triage.Candidate]) -> None:
    partitioned = triage.partition(candidates)
    review = partitioned[triage.Disposition.REVIEW]
    merge = partitioned[triage.Disposition.MERGE]

    print(f"\n{len(candidates)} candidate groups")
    print(f"  {len(merge):>4} proposed for merging")
    print(f"  {len(review):>4} needing a decision")

    by_reason: dict[triage.Reason, int] = {}
    for candidate in candidates:
        by_reason[candidate.reason] = by_reason.get(candidate.reason, 0) + 1
    for reason, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>4} {reason.value}")

    if review:
        print("\nNeeding a decision, strongest agreement first:\n")
        for candidate in review:
            print(f"  {candidate.comparison.score:.2f} [{candidate.reason.value}]")
            for line in candidate.summarize():
                print(f"         {line}")


def resolve_in_ui(
    candidates: Sequence[triage.Candidate], *, include_proposed: bool
) -> Sequence[triage.Candidate]:
    """Ask about candidates in the UI, leaving the ones not shown alone."""
    shown = [
        index
        for index, candidate in enumerate(candidates)
        if include_proposed or candidate.disposition is triage.Disposition.REVIEW
    ]
    if not shown:
        print("Nothing needs a decision.")
        return candidates

    resolved = asyncio.run(
        triage_ui.resolve_candidates([candidates[index] for index in shown])
    )
    out = list(candidates)
    for index, candidate in zip(shown, resolved, strict=True):
        out[index] = candidate
    return tuple(out)


def report_pending(candidates: Sequence[triage.Candidate]) -> None:
    partitioned = triage.partition(candidates)
    review = partitioned[triage.Disposition.REVIEW]
    print(f"  {len(partitioned[triage.Disposition.MERGE]):>4} proposed for merging")
    print(f"  {len(review):>4} needing a decision")
    if review:
        print("\nNeeding a decision, strongest agreement first:\n")
        for candidate in review:
            print(f"  {candidate.comparison.score:.2f} [{candidate.reason.value}]")
            for line in candidate.summarize():
                print(f"         {line}")


def write_revisions(
    path: pathlib.Path, revised: Sequence[types.Word], deck_name: str | None
) -> None:
    """Write rewritten notes out in the format Anki imports.

    Directives, not a header row: the two are alternatives, and a header row
    following directives is imported as a note.
    """
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        handle.write("\n".join(types.Word.make_anki_directives(deck_name)) + "\n")
        writer = csv.writer(handle)
        writer.writerows(word.to_csv_line(require_rank=True) for word in revised)


def report_recorded(
    candidates: Sequence[triage.Candidate], newly_rejected: int, remembered: int
) -> None:
    partitioned = triage.partition(candidates)
    separated = partitioned[triage.Disposition.KEEP_SEPARATE]

    print(f"\nRecorded {newly_rejected} newly separated groups")
    print(f"  {remembered} separated groups remembered in total")
    print(f"  {len(triage.to_merge_sets(candidates))} groups would go to the merge UI")
    print(f"  {len(partitioned[triage.Disposition.REVIEW])} still undecided")

    if separated:
        print("\nMarked as deliberately separate:\n")
        for candidate in separated:
            for line in candidate.summarize():
                print(f"  {line}")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--inputs",
        type=pathlib.Path,
        nargs="*",
        default=(),
        help="CSVs of words, such as the output or diff CSV",
    )
    parser.add_argument(
        "--exported-deck",
        type=pathlib.Path,
        default=None,
        help="Path to a text export of the deck",
    )
    parser.add_argument(
        "--triage-file",
        type=pathlib.Path,
        default=DEFAULT_TRIAGE_FILE,
        help=f"Where decisions are written and read (default: {DEFAULT_TRIAGE_FILE})",
    )
    parser.add_argument(
        "--rejection-cache",
        type=pathlib.Path,
        default=DEFAULT_REJECTION_CACHE,
        help=f"Record of deliberate splits (default: {DEFAULT_REJECTION_CACHE})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Read back the existing triage file instead of regenerating it",
    )
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="Answer the questions in a UI rather than an editor",
    )
    parser.add_argument(
        "--revisions-file",
        type=pathlib.Path,
        default=DEFAULT_REVISIONS_FILE,
        help=f"Where rewritten notes are written (default: {DEFAULT_REVISIONS_FILE})",
    )
    parser.add_argument(
        "--deck-name",
        default=None,
        help="Deck named in the revisions CSV directives",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Also review groups made up entirely of notes already in the deck",
    )
    parser.add_argument(
        "--include-proposed",
        action="store_true",
        help="With --resolve, also ask about the groups already proposed for merging",
    )
    args = parser.parse_args()

    if args.resolve and not TRIAGE_UI_AVAILABLE:
        raise SystemExit(
            "--resolve needs textual; install the merge_ui extra to use it."
        )

    rejections = triage.RejectionCache(args.rejection_cache)

    if args.apply:
        if not args.triage_file.exists():
            raise SystemExit(f"{args.triage_file} does not exist; run without --apply.")
        candidates = triage.read_triage_file(args.triage_file)
        print(f"Read {len(candidates)} candidate groups from {args.triage_file}")
    else:
        deck, added = load_words(args.inputs, args.exported_deck)
        words = (*deck, *added)
        if not words:
            raise SystemExit("No words loaded; pass --inputs and/or --exported-deck.")
        if len(rejections):
            print(f"Ignoring {len(rejections)} previously separated groups")

        # With nothing being added there is no import to be about, so a run can
        # only sensibly mean the whole deck.
        focus = None if args.include_existing or not added else added
        if focus is not None:
            print(f"Limiting to groups touching the {len(added)} added words")

        comparer = definition_similarity.DefinitionComparer(
            word.meaning for word in words
        )
        candidates = triage.find_candidates(
            words, comparer, rejections=rejections, focus=focus
        )
        print(f"\n{len(candidates)} candidate groups")

    if args.resolve:
        candidates = resolve_in_ui(candidates, include_proposed=args.include_proposed)

    triage.write_triage_file(args.triage_file, candidates)

    if args.apply or args.resolve:
        before = len(rejections)
        triage.record_rejections(candidates, rejections)
        rejections.save()
        report_recorded(candidates, len(rejections) - before, len(rejections))

        revised = triage.collect_revisions(candidates)
        if revised:
            write_revisions(args.revisions_file, revised, args.deck_name)
            print(f"\nRewrote {len(revised)} notes -> {args.revisions_file}")
            for word in revised:
                print(
                    f"  {word.rank} {word.word} ({word.part_of_speech}) {word.meaning}"
                )
            print("\nImport that CSV into Anki to apply the rewrites.")
    else:
        report_pending(candidates)
        print(f"\nWrote {args.triage_file}. Edit the 'disposition' fields and rerun")
        print("with --apply, or rerun with --resolve to answer them in a UI.")


if __name__ == "__main__":
    main()
