"""Hand a word CSV to reviewers in chunks, and apply their corrections.

This is the mechanical half of the full-run harness skill
(.claude/skills/full-run): ``export`` splits a word CSV into JSON chunks a
reviewer can read, with likely duplicates already pointed out, and
``apply`` folds the reviewers' corrections back in, writing a corrected copy
of the CSV and a changelog while leaving the original alone. The CSV fields
carry JSON, so editing them by hand is an easy way to corrupt a deck; going
through the project's own reader and writer avoids that.

Corrections are a JSON list. Each entry names one row, by ``row`` (the
``row`` number in the exported chunks: its 0-based position in the whole
file) or by ``rank``, repeats the row's current
``word`` as a guard, and gives a ``reason``. It may then ``set`` any of
``word``, ``meaning``, ``part_of_speech`` and ``article`` (null clears the
article), ``drop`` the row, and edit its ``sentences``:

    {"index": 2, "remove": true}
    {"index": 0, "spanish": "...", "english": "..."}
    {"add": true, "spanish": "...", "english": "...", "first": false}

Indices refer to the sentences as exported. An edit may give ``spanish``,
``english`` or both, and keeps the sentence's source unless it also says
``"source": "reviewer"`` (for a sentence the reviewer has rewritten, or one
wrongly attributed). Added sentences are attributed to ``--reviewer-model``,
and ``"first": true`` makes one the card's main sentence. Every problem in the
corrections is reported at once, and nothing is written unless there are
none.

A ``row`` key is only meaningful for the file the chunks were exported from,
so ``apply`` refuses row-keyed corrections unless ``--index`` is given to
check that; corrections meant for a second file (the diff's, applied to
``output.csv`` as well) are keyed by ``rank``.

In the exported rows, ``csv_matches`` and ``deck_matches`` are probable
duplicates by loose string matching (see ``fold_keys``); a row's
``deck_matches`` leave out the deck note with the row's own rank, which in a
diff is the note the row updates. With ``--annotations``, each row also
carries the reading passages its book sentences came from.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import pathlib
import re
import sys
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

import attrs

from anki_lang_cards import _types as types
from anki_lang_cards import anki_import, headword

DEFAULT_CHUNK_SIZE: Final = 25
SETTABLE_FIELDS: Final = frozenset({"word", "meaning", "part_of_speech", "article"})
_GENDER_SUFFIX: Final = re.compile(r"/(a|as|es|os)\b")
_PARENTHETICAL: Final = re.compile(r"\s*\([^)]*\)")


class CorrectionError(ValueError):
    """Corrections that cannot be applied, with every problem named."""

    def __init__(self, problems: Sequence[str]):
        self.problems = tuple(problems)
        super().__init__("\n".join(f"  - {problem}" for problem in problems))


def file_digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fold_keys(word: str) -> frozenset[str]:
    """Keys under which two headwords are probably the same word.

    This starts from ``headword.lookup_key``, the form duplicate detection
    compares notes by, so that anything the generation step would group
    (``mirón/a`` and ``mirón/ona``, ``percatarse (de)`` and ``percatarse``)
    is pointed out here too. On top of that, accents, case, a gender suffix
    (``seco/a``), a parenthetical qualifier, a pronominal ``se`` and a plural
    ending on a single word are all ignored, since the misses that matter
    have been things like ``apañarselas``/``apañárselas``,
    ``despedir``/``despedirse`` and ``gatos``/``gato``. It is deliberately
    loose: the matches are leads for a reviewer, not merges.
    """
    text = unicodedata.normalize("NFD", headword.lookup_key(word))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = _PARENTHETICAL.sub("", _GENDER_SUFFIX.sub("", text))
    key = " ".join(text.split())
    keys = {key}
    first, _, rest = key.partition(" ")
    if first.endswith(("arse", "erse", "irse")):
        keys.add(" ".join(filter(None, (first.removesuffix("se"), rest))))
    if not rest and len(key) > 3:
        # A plural left as the headword (``gatos``) should still find ``gato``.
        if key.endswith("es"):
            keys.add(key[:-2])
        if key.endswith("s"):
            keys.add(key[:-1])
    return frozenset(keys)


def _describe_source(source: types.Citation | None) -> str | None:
    match source:
        case None:
            return None
        case types.BookCitation(title=title, pages=pages):
            if not pages:
                return f"book: {title}"
            first, last = min(pages), max(pages)
            span = str(first) if first == last else f"{first}-{last}"
            return f"book: {title}, p. {span}"
        case types.ModelCitation(model=model):
            return f"model: {model}"
        case _:
            return source.source_type


def _deck_entry(word: types.Word) -> Mapping[str, Any]:
    return {
        "rank": word.rank,
        "word": word.word,
        "part_of_speech": str(word.part_of_speech),
        "meaning": word.meaning,
    }


ContextKey = tuple[str, tuple[int, ...]]


def _context_key(title: str, pages: Iterable[int] | None) -> ContextKey:
    return (title, tuple(sorted(pages or ())))


def context_index(
    books: Iterable[types.CombinedBookAnnotations],
) -> Mapping[ContextKey, Sequence[str]]:
    """Reading passages by the book and pages a citation names."""
    index: dict[ContextKey, list[str]] = {}
    for book in books:
        for annotation in book.annotations:
            key = _context_key(book.book.title, annotation.page_numbers)
            passages = index.setdefault(key, [])
            if annotation.context not in passages:
                passages.append(annotation.context)
    return index


def _contexts_for(
    word: types.Word, contexts: Mapping[ContextKey, Sequence[str]]
) -> Sequence[str]:
    found: list[str] = []
    for sentence in word.sentences:
        if isinstance(sentence.source, types.BookCitation):
            key = _context_key(sentence.source.title, sentence.source.pages)
            found.extend(c for c in contexts.get(key, ()) if c not in found)
    return found


def export_rows(
    words: Sequence[types.Word],
    deck: Sequence[types.Word],
    contexts: Mapping[ContextKey, Sequence[str]] | None = None,
) -> Sequence[Mapping[str, Any]]:
    """The rows as reviewers see them, with likely duplicates pointed out."""
    by_key: dict[str, list[int]] = {}
    for position, word in enumerate(words):
        for key in fold_keys(word.word):
            by_key.setdefault(key, []).append(position)
    deck_by_key: dict[str, list[types.Word]] = {}
    for deck_word in deck:
        for key in fold_keys(deck_word.word):
            deck_by_key.setdefault(key, []).append(deck_word)

    rows = []
    for position, word in enumerate(words):
        keys = fold_keys(word.word)
        csv_matches = sorted(
            {other for key in keys for other in by_key[key]} - {position}
        )
        deck_matches = {
            id(match): match for key in keys for match in deck_by_key.get(key, ())
        }
        row: dict[str, Any] = {
            "row": position,
            "rank": word.rank,
            "word": word.word,
            "part_of_speech": str(word.part_of_speech),
            "meaning": word.meaning,
            "article": word.article,
            "model_used": word.model_used,
            "sentences": [
                {
                    "index": index,
                    "spanish": sentence.spanish,
                    "english": sentence.english,
                    "source": _describe_source(sentence.source),
                }
                for index, sentence in enumerate(word.sentences)
            ],
            "csv_matches": csv_matches,
            "deck_matches": [
                _deck_entry(match)
                for match in sorted(deck_matches.values())
                if not (word.rank is not None and match.rank == word.rank)
            ],
        }
        if contexts is not None:
            row["contexts"] = _contexts_for(word, contexts)
        rows.append(row)
    return rows


def write_chunks(
    csv_path: pathlib.Path,
    rows: Sequence[Mapping[str, Any]],
    out_dir: pathlib.Path,
    chunk_size: int,
) -> Sequence[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for number, start in enumerate(range(0, len(rows), chunk_size), start=1):
        path = out_dir / f"chunk-{number:03d}.json"
        chunk = rows[start : start + chunk_size]
        path.write_text(json.dumps(chunk, indent=2, ensure_ascii=False))
        paths.append(path)
    index = {
        "source": str(csv_path),
        "sha256": file_digest(csv_path),
        "rows": len(rows),
        "chunks": [path.name for path in paths],
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    return paths


@attrs.frozen
class Change:
    """One applied correction, for the changelog."""

    word: types.Word
    result: types.Word | None
    reason: str
    notes: Sequence[str]


def _resolve_row(
    entry: Mapping[str, Any],
    words: Sequence[types.Word],
    by_rank: Mapping[int, int],
    label: str,
) -> tuple[int | None, list[str]]:
    has_row, has_rank = "row" in entry, "rank" in entry
    if has_row == has_rank:
        return None, [f"{label}: give exactly one of 'row' and 'rank'"]
    if has_row:
        row = entry["row"]
        if not isinstance(row, int) or not 0 <= row < len(words):
            return None, [f"{label}: there is no row {row!r}"]
    else:
        row = by_rank.get(entry["rank"])
        if row is None:
            return None, [f"{label}: there is no row with rank {entry['rank']!r}"]
    if entry.get("word") != words[row].word:
        return None, [
            f"{label}: row {row} is {words[row].word!r}, not {entry.get('word')!r}; "
            "the corrections may be for a different version of the file"
        ]
    return row, []


def _edit_sentences(
    word: types.Word,
    operations: Iterable[Mapping[str, Any]],
    reviewer: types.ModelCitation,
    label: str,
) -> tuple[Sequence[types.Sentence], list[str], list[str]]:
    kept: dict[int, types.Sentence | None] = dict(enumerate(word.sentences))
    first: list[types.Sentence] = []
    last: list[types.Sentence] = []
    notes: list[str] = []
    problems: list[str] = []
    for operation in operations:
        if operation.get("add"):
            if not (operation.get("spanish") and operation.get("english")):
                problems.append(f"{label}: an added sentence needs spanish and english")
                continue
            sentence = types.Sentence(
                spanish=operation["spanish"],
                english=operation["english"],
                source=reviewer,
            )
            (first if operation.get("first") else last).append(sentence)
            notes.append(f"added: {sentence.spanish}")
            continue
        index = operation.get("index")
        if not isinstance(index, int) or kept.get(index, False) is False:
            problems.append(f"{label}: there is no sentence {index!r}")
            continue
        current = kept[index]
        if current is None:
            problems.append(f"{label}: sentence {index} was already removed")
            continue
        if operation.get("remove"):
            kept[index] = None
            notes.append(f"removed: {current.spanish}")
        elif {"spanish", "english", "source"} & operation.keys():
            source = operation.get("source")
            if source not in (None, "reviewer"):
                problems.append(
                    f"{label}: sentence {index}: 'source' can only be \"reviewer\""
                )
                continue
            edited = attrs.evolve(
                current,
                spanish=operation.get("spanish", current.spanish),
                english=operation.get("english", current.english),
                source=reviewer if source == "reviewer" else current.source,
            )
            kept[index] = edited
            if edited.spanish != current.spanish:
                notes.append(f"edited: {current.spanish} -> {edited.spanish}")
            if edited.english != current.english:
                notes.append(
                    f"translation of '{edited.spanish}': "
                    f"{current.english} -> {edited.english}"
                )
            if edited.source is not current.source:
                notes.append(f"attributed to the reviewer: {edited.spanish}")
        else:
            problems.append(f"{label}: sentence {index} has no change to make")
    sentences = (*first, *(s for s in kept.values() if s is not None), *last)
    return sentences, notes, problems


def apply_corrections(
    words: Sequence[types.Word],
    corrections: Sequence[Mapping[str, Any]],
    reviewer_model: str,
) -> tuple[Sequence[types.Word], Sequence[Change]]:
    """Apply corrections in order, or raise naming every problem."""
    by_rank = {word.rank: row for row, word in enumerate(words) if word.rank}
    current: list[types.Word | None] = list(words)
    reviewer = types.ModelCitation(model=reviewer_model)
    changes: list[Change] = []
    problems: list[str] = []
    for position, entry in enumerate(corrections):
        label = f"correction {position} ({entry.get('word', '?')})"
        row, found = _resolve_row(entry, words, by_rank, label)
        problems.extend(found)
        if row is None:
            continue
        if not entry.get("reason"):
            problems.append(f"{label}: give a reason")
        word = current[row]
        if word is None:
            problems.append(f"{label}: row {row} was already dropped")
            continue
        if entry.get("drop"):
            current[row] = None
            changes.append(Change(words[row], None, entry.get("reason", ""), ()))
            continue
        updates = dict(entry.get("set", {}))
        if unknown := updates.keys() - SETTABLE_FIELDS:
            problems.append(f"{label}: cannot set {sorted(unknown)}")
            continue
        notes = [
            f"{field}: {getattr(word, field)!s} -> {value!s}"
            for field, value in updates.items()
        ]
        sentences, sentence_notes, sentence_problems = _edit_sentences(
            word, entry.get("sentences", ()), reviewer, label
        )
        problems.extend(sentence_problems)
        try:
            revised = attrs.evolve(word, sentences=sentences, **updates)
        except (ValueError, TypeError) as error:
            problems.append(f"{label}: {error}")
            continue
        current[row] = revised
        changes.append(
            Change(words[row], revised, entry.get("reason", ""), notes + sentence_notes)
        )
    if problems:
        raise CorrectionError(problems)
    return [word for word in current if word is not None], changes


def leading_lines(csv_path: pathlib.Path) -> Sequence[str]:
    """The directive lines and header row a CSV starts with, verbatim."""
    lines = []
    with csv_path.open() as f:
        for line in f:
            if line.startswith("#"):
                lines.append(line)
                continue
            first_row = next(csv.reader([line]), [])
            header = tuple(types.Word.make_csv_header())
            if tuple(cell.strip().lower() for cell in first_row) == header:
                lines.append(line)
            break
    return lines


def write_csv(
    path: pathlib.Path, header_lines: Sequence[str], words: Iterable[types.Word]
) -> None:
    with path.open("w", newline="") as f:
        f.writelines(header_lines)
        csv.writer(f).writerows(word.to_csv_line() for word in words)


def format_changelog(source: pathlib.Path, changes: Sequence[Change]) -> str:
    lines = [f"# Corrections to `{source}`", ""]
    for change in changes:
        where = f"rank {change.word.rank}" if change.word.rank else "unranked"
        if change.result is None:
            lines.append(
                f"- **dropped** `{change.word.word}` ({where}): {change.reason}"
            )
            continue
        title = change.word.word
        if change.result.word != change.word.word:
            title = f"{change.word.word}` -> `{change.result.word}"
        lines.append(f"- `{title}` ({where}): {change.reason}")
        lines.extend(f"  - {note}" for note in change.notes)
    return "\n".join(lines) + "\n"


async def _load_live_deck(deck_name: str) -> Sequence[types.Word]:
    """Every note in the deck, mostly read back from the note cache.

    The deck has to be fetched whole: Anki's search is for exact spellings,
    so asking it for the words under review would never turn up the variant
    spellings that ``fold_keys`` exists to point out. The note cache makes
    this cheap after the first run, since only notes that changed since are
    fetched again.
    """
    note_cache = anki_import.NoteCache(anki_import.DEFAULT_NOTE_CACHE)
    async with anki_import.create_client() as client:
        note_ids = await client.get_note_ids(deck_name, '"Rank:re:.*"')
        notes = await anki_import.fetch_note_words(
            client, note_ids, cache=note_cache, description="Fetching deck notes"
        )
    return list(notes.values())


def _load_corrections(paths: Sequence[pathlib.Path]) -> Sequence[Mapping[str, Any]]:
    corrections: list[Mapping[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise CorrectionError([f"{path}: expected a JSON list of corrections"])
        corrections.extend(data)
    return corrections


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="Split a word CSV into review chunks")
    export.add_argument("csv", type=pathlib.Path)
    export.add_argument("--out-dir", type=pathlib.Path, required=True)
    export.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Rows per chunk, i.e. per reviewer (default: %(default)s)",
    )
    export.add_argument(
        "--annotations",
        type=pathlib.Path,
        help="combined_annotations.json, to give each row the reading passages "
        "its book sentences came from",
    )
    deck = export.add_mutually_exclusive_group()
    deck.add_argument(
        "--exported-deck",
        type=pathlib.Path,
        help="A text export of the deck, to point out duplicates of existing notes",
    )
    deck.add_argument(
        "--anki",
        action="store_true",
        help="Look duplicates up in the live deck through Anki Connect",
    )
    export.add_argument("--deck-name", default="Spanish::Vocab by Frequency")

    apply = commands.add_parser("apply", help="Apply corrections to a word CSV")
    apply.add_argument("csv", type=pathlib.Path)
    apply.add_argument("corrections", type=pathlib.Path, nargs="+")
    apply.add_argument("--out", type=pathlib.Path, required=True)
    apply.add_argument("--changelog", type=pathlib.Path)
    apply.add_argument(
        "--reviewer-model",
        required=True,
        help="Recorded as the source of any sentence the reviewer wrote",
    )
    apply.add_argument(
        "--index",
        type=pathlib.Path,
        help="The index.json from `export`: refuses to apply anything if the CSV "
        "has changed since, which matters for corrections keyed by row number",
    )

    args = parser.parse_args(argv)
    words = types.Word.words_from_csv(args.csv)

    if args.command == "export":
        if args.exported_deck is not None:
            deck_words = anki_import.load_anki_words(args.exported_deck)
        elif args.anki:
            deck_words = asyncio.run(_load_live_deck(args.deck_name))
        else:
            print(
                "No deck given (--anki or --exported-deck): deck_matches will "
                "be empty."
            )
            deck_words = ()
        contexts = None
        if args.annotations is not None:
            contexts = context_index(
                types.CombinedBookAnnotations.sequence_from_json(args.annotations)
            )
        rows = export_rows(words, deck_words, contexts)
        paths = write_chunks(args.csv, rows, args.out_dir, args.chunk_size)
        print(f"Wrote {len(rows)} rows in {len(paths)} chunks to {args.out_dir}")
        return 0

    if args.index is not None:
        index = json.loads(args.index.read_text())
        if index["sha256"] != file_digest(args.csv):
            print(
                f"error: {args.csv} has changed since it was exported; "
                "export it again so the row numbers match",
                file=sys.stderr,
            )
            return 2
    try:
        corrections = _load_corrections(args.corrections)
        if args.index is None and any("row" in entry for entry in corrections):
            raise CorrectionError(
                [
                    "corrections keyed by 'row' need --index, which checks that "
                    "the rows are still the ones that were exported; key them "
                    "by 'rank' to apply them to another file"
                ]
            )
        revised, changes = apply_corrections(words, corrections, args.reviewer_model)
    except CorrectionError as error:
        print(f"error: the corrections have problems:\n{error}", file=sys.stderr)
        return 2
    write_csv(args.out, leading_lines(args.csv), revised)
    if args.changelog is not None:
        args.changelog.write_text(format_changelog(args.csv, changes))
    dropped = sum(change.result is None for change in changes)
    print(
        f"Applied {len(changes)} corrections ({dropped} rows dropped); "
        f"wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
