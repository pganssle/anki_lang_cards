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
import sys
from collections.abc import Callable, Sequence
from typing import Final

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


async def repair_deck(
    deck_name: str, *, apply: bool, full_scan: bool = False, batch_size: int = 50
) -> int:
    async with anki_import.create_client() as client:
        query = None if full_scan else damage_query()
        note_ids = await client.get_note_ids(deck_name, query)
        print(f"Checking {len(note_ids)} notes in {deck_name!r}.")

        updates: list[dict] = []
        failures: list[tuple[str, str, str]] = []
        async for note in client.get_anki_notes(note_ids):
            new_fields: dict[str, str] = {}
            for field_name, canonicalize in FIELD_CANONICALIZERS.items():
                field = note["fields"].get(field_name)
                if field is None or not field["value"].strip():
                    continue
                try:
                    canonical = canonicalize(field["value"])
                except json.JSONDecodeError as e:
                    word = note["fields"].get("Word", {}).get("value", "<unknown>")
                    failures.append((word, field_name, str(e)))
                    continue
                if canonical != field["value"]:
                    new_fields[field_name] = canonical

            if new_fields:
                word = note["fields"].get("Word", {}).get("value", "<unknown>")
                print(f"{word}: {', '.join(sorted(new_fields))}")
                updates.append({"id": note["noteId"], "fields": new_fields})

        print(f"\n{len(updates)} notes need repair.")
        if failures:
            print(f"{len(failures)} fields could not be repaired:", file=sys.stderr)
            for word, field_name, error in failures:
                print(f"  {word} ({field_name}): {error}", file=sys.stderr)

        if not apply:
            if updates:
                print("Dry run: nothing written. Re-run with --apply to repair.")
        else:
            for batch in itertools.batched(updates, batch_size):
                await client.update_note_info(list(batch))
            print(f"Updated {len(updates)} notes.")

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
    args = parser.parse_args(argv)

    return asyncio.run(
        repair_deck(args.deck, apply=args.apply, full_scan=args.full_scan)
    )


if __name__ == "__main__":
    sys.exit(main())
