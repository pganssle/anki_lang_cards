"""Report single-word notes whose examples never show the word outside a phrase.

    python scripts/find_phrase_bound.py --inputs output/diff_from_existing.csv

A word is reported when its example sentences all show it inside the same run of
adjacent words — "reojo" always inside "de reojo". Words whose surroundings vary
are not reported, so "genio" stays put even though it also appears in "genio y
figura hasta la sepultura".

Where the phrase is already its own note, that is called out: the single-word
note is then a fragment of a card you already have. Where it is not, the phrase
is usually what the note should have been.

This reads nothing but the sentences already on the notes, so it keeps working
regardless of how definitions are worded. It stays quiet rather than guessing:
discontinuous phrases ("se las van a apañar") and inflected headwords are missed.
"""

import argparse
import pathlib
from collections.abc import Sequence

from anki_lang_cards import _types as types
from anki_lang_cards import anki_import, phrase_context


def load_words(
    inputs: Sequence[pathlib.Path], exported_deck: pathlib.Path | None
) -> Sequence[types.Word]:
    words: list[types.Word] = []
    if exported_deck is not None:
        deck = anki_import.load_anki_words(exported_deck)
        print(f"Loaded {len(deck)} words from {exported_deck}")
        words.extend(deck)
    for path in inputs:
        from_csv = types.Word.words_from_csv(path)
        print(f"Loaded {len(from_csv)} words from {path}")
        words.extend(from_csv)
    return tuple(words)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--inputs", type=pathlib.Path, nargs="*", default=())
    parser.add_argument("--exported-deck", type=pathlib.Path, default=None)
    parser.add_argument(
        "--min-sentences",
        type=int,
        default=phrase_context.DEFAULT_MIN_SENTENCES,
        help="Fewest examples that can say anything about how a word is used",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=phrase_context.DEFAULT_MIN_COVERAGE,
        help="Share of the examples a phrase must appear in",
    )
    args = parser.parse_args()

    words = load_words(args.inputs, args.exported_deck)
    if not words:
        raise SystemExit("No words loaded; pass --inputs and/or --exported-deck.")

    found = phrase_context.find_phrase_bound_words(
        words,
        min_sentences=args.min_sentences,
        min_coverage=args.min_coverage,
    )
    print(f"\n{len(found)} single-word notes never seen outside a phrase\n")
    for entry in found:
        note = "  <- phrase is already a note" if entry.is_redundant else ""
        print(
            f"  {entry.context.coverage:.2f} n={entry.context.sentence_count:<3}"
            f" {entry.word.word:<18} {entry.context.phrase!r}{note}"
        )


if __name__ == "__main__":
    main()
