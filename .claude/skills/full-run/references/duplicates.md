# Answering duplicate groups

`python -m anki_lang_cards --no-ui` writes the groups it cannot settle to
`output/merge_candidates.json` and exits with status 3. Each entry is one
group of notes that might be the same word: some from the deck (they have
ranks), some from this run's inputs. The answer goes in the entry itself,
then the file is passed back with `--decision-file`.

## What an entry contains

- `summary`: one line per note, the quickest way to see the group.
- `words`: the notes, as CSV text. Deck notes have a rank; inputs do not.
- `reason`: why they were grouped, which says a lot about the right answer:
  - `compatible` — same headword, compatible parts of speech. These belong
    on one card; what differs is usually the definition. Merge (improving the
    wording) or split; keeping them separate leaves two identical card
    fronts.
  - `characteristics` — same headword and base part of speech, differing in
    the details (`nm/f` against `nf`, `vi` against `vtr`). Almost always one
    word labelled two ways between runs; usually a merge.
  - `gender` — the same noun with two stated genders. `el cura` and `la
    cura` are two words, and the definitions cannot tell you (`el orden` and
    `la orden` both gloss as "order"), so think about the actual words.
  - `base-part-of-speech` — same headword, different part of speech.
    English often glosses both the same (`segundo` the adjective and the
    noun), so agreeing definitions are not evidence of a duplicate.
- `score`, `shared_glosses`: how alike the definitions are — meaningful for
  the first two reasons, misleading for the last two.
- `proposed`: the merged note that accepting a merge would record, as CSV
  text. Read it before accepting; it is a mechanical combination. Groups
  grouped by `gender` or `base-part-of-speech` never get one, so a merge
  there has to spell out `replacements`.
- `disposition`: your answer. It starts as `review`.

## The answers

- **`merge`** — the notes are one word with one card. Leaving out
  `replacements` accepts `proposed` as it is. To improve the wording, set
  `replacements` to the one merged note as CSV text (start from `proposed`
  and edit the meaning, part of speech or headword). The merged note keeps
  the lowest rank, so it updates the existing deck note rather than adding
  one.
- **`keep-separate`** — genuinely different words or senses that deserve
  separate cards: `despellejar` (to skin) and `despellejarse` (to scrape
  one's own skin), or a word that happens to share a spelling with an
  unrelated one. This is remembered, so the group stops coming back.
- **`split`** — the group is tangled: say a deck note conflates two senses
  that the input separates. `replacements` lists two or more notes as CSV
  text, and `sentence_allocation` (one entry per sentence across the whole
  group, in order) says which card each sentence goes to: `"1"`, `"2"`, or
  `"1,2"` to share one.
- **`review`** — you are not sure. Leave it, and ask the user.

## When to answer and when to ask

Answer on your own when the call is clear from the definitions and
sentences: an input that is the same word and sense as a deck note (merge),
spelling variants of one headword (merge, keeping the correct spelling),
plainly different words that only collided on spelling (keep separate).

Ask when it depends on the user's taste or on facts you cannot check: two
senses close enough that they could go either way, a deck note whose
definition looks wrong (the merge would carry the error forward — say so), a
reflexive/base pair where both cards might be wanted, or anything where the
merge would lose information you think matters. When you ask, give your
recommendation and the reason in a sentence or two; the user should be able
to answer with "yes" or a short correction.

## What an answer does

An answer is not a note to the user; it is acted on. Answers are recorded
for good in `.cache/merges/decisions.json`, shared by every tool, so the
same group is never asked about again, and a `merge` puts the merged
wording into the diff as an update of the existing deck note: importing the
diff rewrites that note's headword, part of speech and meaning (and
combines its sentences with the input's). The user sees none of that
happen, so the report has to show it. For every group you answered
yourself, `output/runs/<run>/duplicates.md` and the report's "Duplicates
you answered" section list the disposition and, for a merge or split, the
existing note's headword, part of speech and meaning before and after.

## Changing an answer

Do not edit `.cache/merges/decisions.json` by hand: its entries are keyed
on the notes' exact CSV text, and a mis-edit either leaves the old answer in
place or breaks every tool that reads the file. The supported way to change
an answer is the same decision file: take your copy of the answered file
(`output/runs/<run>/merge_candidates.answered.json`), change the entry's
`disposition` (or `replacements`), and pass the file to the pipeline again
with `--decision-file`. Recording a group that already has an answer
replaces the old answer with the new one (the run reports it as "0 new
answers", which is right), and the diff is regenerated from the new
decisions. Nothing in the deck has changed until the user imports, so until
then this is a full undo; after an import, a changed answer is a new update
of the note, not a restoration of the old one — say so if it comes to that.

## Mechanics

- If the run rejects the file (exit status 2), it lists every problem at
  once and records nothing; fix them and pass the file again.
- Write each answer and its reason to `output/runs/<run>/duplicates.md` as
  you go, and keep a copy of the answered file as
  `output/runs/<run>/merge_candidates.answered.json`: the pending file is
  rewritten on each run, so it does not keep a history, and the copy is
  what changing an answer starts from.
- Get the answers right rather than fast; a wrong `keep-separate` leaves two
  cards for one word in the deck and is never asked again, and a wrong
  `merge` rewrites a note the user may have been happy with.
