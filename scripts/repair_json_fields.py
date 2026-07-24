"""Repair mangled fields of existing notes over AnkiConnect.

Two kinds of damage are repaired:

- The JSON-bearing fields (``Extra Sentences``, ``Sentence Source``) carry
  historical escape damage: the old card template read them through a JS
  template literal that consumed the ``\\"`` escapes, and the manual
  workarounds (doubling or stripping the backslashes) left fields that are
  not valid JSON. These are re-parsed with the pipeline's repair chain and
  rewritten in the canonical form the generator now emits: valid JSON with
  ``<``, ``>`` and ``&`` written as ``\\uXXXX`` escapes so the field is
  inert as HTML.

- The plain-text fields (``Spanish``, ``English``, ``Definition``, ...)
  accumulate HTML entities (``&#x27;``, ``&quot;``, ``&amp;``) from GUI
  imports done with "allow HTML in fields" disabled. These are unescaped
  back to plain text.

By default the script only reports what it would change; pass ``--apply``
to write the changes back. Anki must be running with the AnkiConnect
add-on enabled.
"""

import argparse
import asyncio
import itertools
import json
import pathlib
import sys
from collections.abc import Callable, Sequence
from typing import Final, TypedDict

import tqdm

from anki_lang_cards import anki_import

# Note and Image are deliberately absent: both may contain intentional
# HTML (user-authored notes, <img> tags).
JSON_FIELD_CANONICALIZERS: Final[dict[str, Callable[[str], str]]] = {
    "Sentence Source": anki_import.canonicalize_sentence_source,
    "Extra Sentences": anki_import.canonicalize_extra_sentences,
}
TEXT_FIELD_CANONICALIZERS: Final[dict[str, Callable[[str], str]]] = dict.fromkeys(
    (
        "Word",
        "Part-of-Speech",
        "Definition",
        "Spanish",
        "English",
        "Article",
        "model_used",
    ),
    anki_import.canonicalize_text_field,
)
FIELD_CANONICALIZERS: Final[dict[str, Callable[[str], str]]] = {
    **JSON_FIELD_CANONICALIZERS,
    **TEXT_FIELD_CANONICALIZERS,
}

# Search regexes matching every field the canonicalizers would change,
# except for leading/trailing-whitespace-only differences and the
# stripped-``\"``-escape damage (bare quotes inside JSON strings), which
# has no textual signature a regex can distinguish from valid JSON.
# False positives are fine — matched notes are still run through the
# canonicalizers, which are the ground truth — so these err on the side
# of matching too much.
#
# Escaping: these strings are embedded in a double-quoted Anki search
# term, where ``"`` must be written ``\"``; backslashes pass through the
# search parser to the (case-insensitive) regex engine verbatim, so a
# literal backslash is matched by ``\\`` exactly as in a plain regex.
#
# Canonical text fields contain no ``&``-entities other than ``&lt;``,
# no literal ``<`` and no U+00A0 (matching bare ``&`` and ``&lt;`` is
# harmless over-approximation).
TEXT_FIELD_DAMAGE_REGEX: Final[str] = r"[&<]|\u{00a0}"

# A ``\uXXXX`` escape other than the canonical &/</>:
# legacy ``ensure_ascii`` serialization that canonicalization rewrites.
# Spelled as an enumeration because the regex engine (Rust ``regex``)
# has no negative lookahead.
_NON_CANONICAL_U_ESCAPE: Final[str] = (
    r"\\u([1-9a-f][0-9a-f]{3}|0[1-9a-f][0-9a-f]{2}"
    r"|00[014-9a-f][0-9a-f]|002[0-57-9a-f]|003[0-9abdf])"
)

# Canonical JSON fields contain no literal ``&``/``<``/``>`` (escaped as
# ``\uXXXX``), no U+00A0, no literal control whitespace (single-line
# output; newlines in content are the two-character ``\n`` escape), no
# doubled backslashes or quotes, always a space after ``,``/``:``
# separators, and ``source`` as an object rather than a string-wrapped
# JSON document.
JSON_FIELD_DAMAGE_REGEX: Final[str] = "|".join(
    (
        r"[&<>]",
        r"[\n\r\t]",
        r"\u{00a0}",
        r"\\\\",
        r"\"\"",
        r",\"",
        r":\"",
        r"\"source\": ?\"\{",
        _NON_CANONICAL_U_ESCAPE,
    )
)


def damage_query() -> str:
    """Anki search query matching every note a full scan could change.

    See the regex comments above for the classes of damage deliberately
    left out (whitespace-only and stripped-escape damage).
    """
    terms = [
        f'"{field_name}:re:{TEXT_FIELD_DAMAGE_REGEX}"'
        for field_name in TEXT_FIELD_CANONICALIZERS
    ] + [
        f'"{field_name}:re:{JSON_FIELD_DAMAGE_REGEX}"'
        for field_name in JSON_FIELD_CANONICALIZERS
    ]
    return " or ".join(terms)


DEFAULT_CACHE_PATH: Final[pathlib.Path] = (
    pathlib.Path(__file__).parent.parent / "local" / "repair_json_fields.cache.jsonl"
)


class CacheEntry(TypedDict):
    """The outcome of examining one note, as stored in the cache."""

    note_id: int
    word: str
    fields: dict[str, str]
    failures: list[list[str]]  # [field_name, error] pairs


class RepairCache:
    """Append-only JSONL record of already-examined notes.

    Each line is one ``CacheEntry``; appending after each note means an
    interrupted run can resume without re-fetching what it already
    examined. A partially-written final line (from a hard interruption)
    is discarded on load. Note ids are collection-global, so entries
    from different decks cannot collide.
    """

    def __init__(self, path: pathlib.Path):
        self._path = path

    def load(self) -> dict[int, CacheEntry]:
        entries: dict[int, CacheEntry] = {}
        if not self._path.exists():
            return entries
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    entry: CacheEntry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entries[entry["note_id"]] = entry
        return entries

    def append(self, entry: CacheEntry) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def clear(self) -> None:
        self._path.unlink(missing_ok=True)


def _examine_note(note: anki_import.NoteData) -> CacheEntry:
    word = note["fields"].get("Word", {}).get("value", "<unknown>")
    new_fields: dict[str, str] = {}
    failures: list[list[str]] = []
    for field_name, canonicalize in FIELD_CANONICALIZERS.items():
        field = note["fields"].get(field_name)
        if field is None or not field["value"].strip():
            continue
        try:
            canonical = canonicalize(field["value"])
        except json.JSONDecodeError as e:
            failures.append([field_name, str(e)])
            continue
        if canonical != field["value"]:
            new_fields[field_name] = canonical

    return CacheEntry(
        note_id=note["noteId"], word=word, fields=new_fields, failures=failures
    )


async def repair_deck(
    deck_name: str,
    *,
    apply: bool,
    full_scan: bool = False,
    cache: RepairCache | None = None,
    batch_size: int = 50,
) -> int:
    entries = cache.load() if cache is not None else {}

    async with anki_import.create_client() as client:
        query = None if full_scan else damage_query()
        note_ids = await client.get_note_ids(deck_name, query)
        to_fetch = [note_id for note_id in note_ids if note_id not in entries]
        cached = len(note_ids) - len(to_fetch)
        print(
            f"Checking {len(note_ids)} notes in {deck_name!r}"
            f" ({cached} already examined)."
        )

        with tqdm.tqdm(
            total=len(note_ids), initial=cached, unit="note", desc="Examining"
        ) as progress:
            async for note in client.get_anki_notes(to_fetch, batch_size=batch_size):
                entry = _examine_note(note)
                entries[entry["note_id"]] = entry
                if cache is not None:
                    cache.append(entry)
                progress.set_description(f"Examining {entry['word']}")
                progress.update(1)

        updates: list[dict] = []
        failures: list[tuple[str, str, str]] = []
        for note_id in note_ids:
            entry = entries[note_id]
            for field_name, error in entry["failures"]:
                failures.append((entry["word"], field_name, error))
            if entry["fields"]:
                updates.append({"id": note_id, "fields": entry["fields"]})

        print(f"\n{len(updates)} notes need repair.")
        if failures:
            print(f"{len(failures)} fields could not be repaired:", file=sys.stderr)
            for word, field_name, error in failures:
                print(f"  {word} ({field_name}): {error}", file=sys.stderr)

        if not apply:
            if updates:
                print("Dry run: nothing written. Re-run with --apply to repair.")
        else:
            batches = itertools.batched(updates, batch_size)
            n_batches = -(-len(updates) // batch_size)
            for batch in tqdm.tqdm(
                batches, total=n_batches, unit="batch", desc="Applying"
            ):
                await client.update_note_info(list(batch))
            print(f"Updated {len(updates)} notes.")
            # The examination results describe pre-repair contents;
            # they are stale the moment the updates are written.
            if cache is not None:
                cache.clear()

    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--deck",
        default="Spanish::Vocab by Frequency",
        help="Name of the deck whose notes should be repaired.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the repaired fields back to Anki (default is a dry run).",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Check every note in the deck instead of only those matching "
        "the damage-signature search query. The query deliberately ignores "
        "whitespace-only and stripped-escape damage; a full scan is the "
        "ground truth.",
    )
    parser.add_argument(
        "--cache-path",
        type=pathlib.Path,
        default=DEFAULT_CACHE_PATH,
        help="Where to record examined notes so an interrupted run can "
        "resume (default: %(default)s).",
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore the local cache entirely: neither read nor write it.",
    )
    cache_group.add_argument(
        "--new-cache",
        action="store_true",
        help="Discard any existing cache and start recording a fresh one.",
    )
    args = parser.parse_args(argv)

    cache = None if args.no_cache else RepairCache(args.cache_path)
    if args.new_cache and cache is not None:
        cache.clear()

    return asyncio.run(
        repair_deck(
            args.deck, apply=args.apply, full_scan=args.full_scan, cache=cache
        )
    )


if __name__ == "__main__":
    sys.exit(main())
