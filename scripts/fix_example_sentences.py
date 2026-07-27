import asyncio
import copy
import itertools
import json
import re

from anki_lang_cards import anki_import


async def main():
    deck_name = "Spanish::Vocab by Frequency"
    async with anki_import.create_client() as client:
        deck_stats = await client.get_deck_stats(deck_name)
    all_cards = {}

    async with anki_import.create_client() as client:
        note_ids = await client.get_note_ids(
            deck_name, f'"Extra Sentences:re:.*source.*"'
        )

    updated = {}
    async with anki_import.create_client() as client:
        async for note in client.get_anki_notes(note_ids):
            note_id = note["noteId"]
            data = note["fields"]["Extra Sentences"]["value"]
            replaced = data.replace("&quot;", '"')
            if re.match("&.+;", replaced):
                breakpoint()
            parsed = json.loads(replaced)
            fixed_sentences = []
            needed_fix: bool = False
            for sentence in parsed:
                if "source" in sentence and isinstance(sentence["source"], str):
                    needed_fix = True
                    source_obj = json.loads(sentence["source"])
                    new_sentence = copy.deepcopy(sentence)
                    new_sentence["source"] = source_obj
                    fixed_sentences.append(new_sentence)
                else:
                    fixed_sentences.append(sentence)

            if needed_fix:
                updated[note_id] = fixed_sentences

    updated_notes = []
    for note_id, new_sentences in updated.items():
        new_note_info = {
            "id": note_id,
            "fields": {
                "Extra Sentences": json.dumps(new_sentences),
            },
        }

        updated_notes.append(new_note_info)

    for batch in itertools.batched(updated_notes, 50):
        async with anki_import.create_client() as client:
            await client.update_note_info(batch)


if __name__ == "__main__":
    asyncio.run(main())
