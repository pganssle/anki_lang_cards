import asyncio
import itertools
import heapq

from anki_lang_cards import anki_import


async def main():
    deck_name = "Spanish::Vocab by Frequency"
    async with anki_import.create_client() as client:
        deck_stats = await client.get_deck_stats(deck_name)
    num_cards = deck_stats["total_in_deck"]
    tidx = 5
    missing_numbers = []
    highest_card_num = 0
    smallest_missing_number = 0
    all_cards = {}

    while True:
        async with anki_import.create_client() as client:
            second_digit = "[0-4]" if tidx % 2 == 0 else "[5-9]"
            second_digit = "[0-9]"
            note_match = f"{tidx}{second_digit}[0-9][0-9]"
            print(f"Fetching note ids matching {note_match}")
            note_ids = await client.get_note_ids(deck_name, f'"Rank:re:{note_match}"')
        if len(note_ids) == 0:
            break

        start = tidx * 1000
        end = start + 1000
        if highest_card_num > 0 and highest_card_num < start - 1:
            if not missing_numbers:
                smallest_missing_number = highest_card_num + 1
            for num in range(highest_card_num+1, start):
                heapq.heappush(missing_numbers, num)
        print(f"Retrieving {len(note_ids)} notes")

        async with anki_import.create_client() as client:
            cards = {}
            async for note in client.get_anki_notes(note_ids):
                card_num = int(note["fields"]["Rank"]["value"])
                cards[card_num] = note["noteId"]
                highest_card_num = max(highest_card_num, card_num)

            all_cards |= cards

        cards_seen = 0
        total_cards = len(cards)

        for i in range(start, end):
            if i not in cards:
                if not missing_numbers:
                    smallest_missing_number = i
                heapq.heappush(missing_numbers, i)
            else:
                cards_seen += 1
                if cards_seen == total_cards:
                    break
        tidx += 1

    end_numbers = []
    # We know we don't have to move anything earlier than the first missing
    # number, even if there are more spots than numbers between the highest
    # number and the first missing number, if we exhaust all the sparse numbers
    # we can ignore all remaining missing numbers.
    for number in range(highest_card_num, smallest_missing_number-1, -1):
        if number in all_cards:
            end_numbers.append(number)


    # Just move stuff from the end to the missing chunks. Not the ideal way to
    # do this, but it involves the least changes
    out_map = {}
    print(f"Found {len(missing_numbers)} missing numbers.")
    for end_number in reversed(end_numbers):
        missing_number = heapq.heapreplace(missing_numbers, end_number)
        # If the numbers at the end get higher than the missing numbers, we are
        # done.
        if missing_number > end_number:
            break
        out_map[end_number] = missing_number
        if not missing_numbers:
            break

    updated_notes = []
    for source_note, dest_num in out_map.items():
        source_note_id = all_cards[source_note]
        new_note_info = {
            "id": source_note_id,
            "fields": {
                "Rank": str(dest_num),
            },
        }

        updated_notes.append(new_note_info)

    print(f"Moving {len(updated_notes)} notes")

    for batch in itertools.batched(updated_notes, 50):
        async with anki_import.create_client() as client:
            await client.update_note_info(batch)


if __name__ == "__main__":
    asyncio.run(main())
