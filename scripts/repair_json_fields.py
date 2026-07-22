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
from collections.abc import Sequence

from anki_lang_cards import anki_import

# Note and Image are deliberately absent: both may contain intentional
# HTML (user-authored notes, <img> tags).
FIELD_CANONICALIZERS = {
    "Sentence Source": anki_import.canonicalize_sentence_source,
    "Extra Sentences": anki_import.canonicalize_extra_sentences,
    "Word": anki_import.canonicalize_text_field,
    "Part-of-Speech": anki_import.canonicalize_text_field,
    "Definition": anki_import.canonicalize_text_field,
    "Spanish": anki_import.canonicalize_text_field,
    "English": anki_import.canonicalize_text_field,
    "Article": anki_import.canonicalize_text_field,
    "model_used": anki_import.canonicalize_text_field,
}


async def repair_deck(deck_name: str, *, apply: bool, batch_size: int = 50) -> int:
    async with anki_import.create_client() as client:
        note_ids = await client.get_note_ids(deck_name)
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
    args = parser.parse_args(argv)

    return asyncio.run(repair_deck(args.deck, apply=args.apply))


if __name__ == "__main__":
    sys.exit(main())
