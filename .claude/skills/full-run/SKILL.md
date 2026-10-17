---
name: full-run
description: Run the anki_lang_cards pipeline end to end from an agent harness, from KOReader highlights (or an existing word CSV) to a reviewed diff ready to import into Anki, with the harness model reviewing the output along the way — headwords, parts of speech, articles, meanings, duplicates against the deck and within the run, and the generated example sentences. Use this whenever the user asks to do a run, process new highlights or annotations, make new cards, update or add to the Spanish deck, or review a generated words CSV or diff, even if they don't say "full run".
---

# Full pipeline run

This skill drives the whole card pipeline that `docs/workflow.md` describes,
and adds a review pass that a person used to do by hand. The pipeline's own
LLM calls go to the API (batched and prompt-cached, so cheap); the review is
done by you, the harness model, which costs nothing per token. That is the
point of the skill: spend your judgment where the pipeline's cheaper model is
weakest — choosing the dictionary form, the sense used in context, spotting
duplicates the string matching missed, and writing sentences that actually
teach the word.

What you may do on your own, and what you hand back, was settled with the
user up front:

- **Correct the word CSV and the diff yourself**, through
  `scripts/review_words.py`, keeping the original files and writing a
  changelog with a reason for every edit.
- **Answer the duplicate groups you are sure about**; leave the ones you are
  not as `review` and ask the user, with your reasoning.
- **Replace weak example sentences** with ones you write, attributed to your
  own model name.
- **Stop before importing into Anki.** The import changes the user's deck;
  your run ends with a reviewed diff and a report.

## Before starting

Work from the repository root: the tools read `config.toml` and write
`input/`, `output/` and `.cache/` relative to the working directory, and
running from elsewhere silently starts with empty caches (including the
duplicate decisions). The pipeline warns "no recorded decisions" whenever
the decision cache is empty: harmless if this checkout has never had a
duplicate answered, and a sign of the wrong directory if it has (look for
`.cache/merges/decisions.json`). Use
the project's Python environment (e.g. `.venv/bin/python`).

Check, and tell the user about anything missing rather than working around it:

1. `config.toml` exists and names a model (`[models] default_model`, or the
   older `[anthropic] default_model`). Note the `[sentences] dialect`; your
   sentences must match it.
2. Anki is running with Anki Connect, so that the review and the generation
   step see the live deck:
   `curl -s localhost:8765 -X POST -d '{"action":"version","version":6}'`.
   If it is not, ask the user to start Anki, or for a fresh text export of
   the deck. The commands below are written for the live deck; with an
   export instead, pass `--exported-deck <file>` in place of `--anki` to
   `review_words.py export`, and add it to every run of the main pipeline
   (`python -m anki_lang_cards`, not `annotations_to_csv`, which never reads
   the deck), which otherwise tries to reach Anki Connect.
3. Where to start. Ask unless the user said: from a Calibre library (steps 1–2
   below; `python -m anki_lang_cards.find_libraries ~ -l es` finds it), from
   an existing `output/combined_annotations.json`, or from a word CSV they
   already have (e.g. `input/words_from_org_mode.csv`), which starts at step 4.

Make a directory for this run's review artifacts, `output/runs/<YYYY-MM-DD>/`
(add a suffix if it exists), and note the start time (`date -Iseconds`); the
cost report filters on it.

**Cost and time.** With an `anthropic/...` model, the two LLM steps run as
batch jobs by default: half price, but a batch can take anywhere from a few
minutes to an hour or more. Before each one, look at what the previous
comparable run cost:

```console
python scripts/usage_report.py
```

and tell the user what to expect (if the cache is empty, there is nothing to
compare against; say so and give the request count instead). If the user is
not there to tell, put the estimate in the report next to the actual cost. Run the long
steps in the background and check on them rather than blocking. If a
batched step is interrupted, re-run the same command: the submitted job is
recorded under `.cache/` and is picked back up rather than paid for again.

## 1–2. Extract and combine the annotations

```console
python -m anki_lang_cards.annotation_extractor -i <library> -l es --context-sentences 2 --output output --json
python -m anki_lang_cards.combine_annotations -i output/annotations.json -o output/combined_annotations.json
```

No LLM involved. `input/book_skip_list.json` excludes books or page ranges
that were already processed; if the extractor reports far more annotations
than the user expects, that file is the first suspect.

## 3. Generate the word CSV

```console
python -m anki_lang_cards.annotations_to_csv
```

This writes `output/words_from_annotations.csv`, and
`output/processed_pages.json`, the pages this run covered — mention that one
in the report, since it is the starting point for updating
`input/book_skip_list.json` so the next run skips them. It records one range
per book, from the first highlighted page to the last, so copying it in also
skips any later highlights on the pages in between; say so. `--limit N` processes
only the first N entries if the user wants a trial run first, and a limited
run does not write `processed_pages.json` at all — so after one, any
`processed_pages.json` lying around is from an earlier run, and must not be
reported as this one's.

## 4. Review the word CSV

This is the step the user used to do by hand and the cheapest place to catch
mistakes: everything downstream — duplicate detection, the generated
sentences — works from these headwords and meanings.

```console
cp output/words_from_annotations.csv output/runs/<run>/words_from_annotations.original.csv
python scripts/review_words.py export output/words_from_annotations.csv \
    --out-dir output/runs/<run>/words --anki \
    --annotations output/combined_annotations.json
```

Every row in the chunks lists `deck_matches` — existing notes that are
probably the same word, found by loose matching against the whole deck
(`--anki` fetches every note, through `.cache/anki_notes.json`, so the first
export after a deck change takes a moment) — and `csv_matches`, other rows
of this run that probably are, and `--annotations` adds the reading passages its sentences
came from (leave it out if the run started from a word CSV), which is what
you need to judge the sense used in context. Read
`references/review-checklist.md` before reviewing; it is what to check and
what the pipeline has historically got wrong.

With more than a couple of chunks, review them in parallel if you can hand
work to subagents: give each one a chunk, the checklist, the dialect, and the
instructions in the checklist's "Reviewing a chunk" section, and have it
write `chunk-NNN.corrections.json` and `chunk-NNN.notes.md` next to the chunk
rather than touching any CSV. Otherwise work through the chunks yourself the
same way.

Then apply all the corrections at once:

```console
python scripts/review_words.py apply output/words_from_annotations.csv \
    output/runs/<run>/words/chunk-*.corrections.json \
    --index output/runs/<run>/words/index.json \
    --reviewer-model <your model id> \
    --out output/words_from_annotations.csv \
    --changelog output/runs/<run>/words-changelog.md
```

`apply` checks every correction before writing anything and lists all the
problems if there are any, so a failed apply leaves the file as it was: fix
the corrections and run it again. `--index` makes it refuse if the CSV
changed since the export.

Do not merge duplicates at this step, even obvious ones. The generation step
has proper merge machinery that keeps both rows' sentences and sources;
dropping a row here would lose them. Instead, make sure duplicate rows use
the same headword, so that machinery recognizes them.

This review is also the last look anyone gets at a word that matches a deck
note exactly: the next step folds an input with the same headword and part of
speech into the existing note as an update without asking. So a wrong
headword or part of speech here either hides a duplicate or merges two words
that should not be merged.

## 5. Resolve duplicates and generate the cards

```console
python -m anki_lang_cards --inputs output/words_from_annotations.csv --no-ui
```

Always pass `--no-ui`; the terminal UI is useless to you. Before generating
anything, this matches the inputs against the deck and itself. If nothing
needs an answer, it carries straight on and generates the sentences (exit
status 0); write "none" in `output/runs/<run>/duplicates.md` and go on to
step 6. Groups it cannot settle from earlier answers stop the run with **exit
status 3**, and it writes them to `output/merge_candidates.json`. Answer them
following `references/duplicates.md`, then:

```console
python -m anki_lang_cards --inputs output/words_from_annotations.csv --no-ui \
    --decision-file output/merge_candidates.json
```

Answers go into the shared decision cache, so they are never asked again by
any tool. Groups you left as `review` come straight back with exit status 3,
and the run cannot generate anything until every group has an answer — so
ask the user about those now, with your reasoning for each, fold their
answers into the file, and re-run. Record what you decided and why in
`output/runs/<run>/duplicates.md`, and keep a copy of the answered file as
`output/runs/<run>/merge_candidates.answered.json`, since the pending file
is rewritten on every run and the copy is how an answer gets changed later
(see the "Mechanics" section of `references/duplicates.md`).

When it gets past the duplicates, it generates sentences (a second batch job)
and writes `output/diff_from_existing.csv` (new and updated notes — the file
the user imports) and adds them to `output/output.csv`, which accumulates
every row this and earlier runs emitted.

## 6. Review the diff

Same mechanics as step 4, on `output/diff_from_existing.csv`. The diff's rows
all have ranks, so key corrections by `rank` and apply the same corrections
to `output/output.csv` too, which is how later runs know what was already
emitted (`apply` refuses `row` keys without `--index`, since a row number
means nothing in a second file; a rank that recurs in `output.csv` names
its last row, the one this diff was made from):

```console
cp output/diff_from_existing.csv output/output.csv output/runs/<run>/
python scripts/review_words.py export output/diff_from_existing.csv \
    --out-dir output/runs/<run>/diff --anki \
    --annotations output/combined_annotations.json
# after reviewing:
python scripts/review_words.py apply output/diff_from_existing.csv \
    output/runs/<run>/diff/chunk-*.corrections.json \
    --index output/runs/<run>/diff/index.json \
    --reviewer-model <your model id> --out output/diff_from_existing.csv \
    --changelog output/runs/<run>/diff-changelog.md
python scripts/review_words.py apply output/output.csv \
    output/runs/<run>/diff/chunk-*.corrections.json \
    --reviewer-model <your model id> --out output/output.csv
```

The focus here shifts to the sentences, which are new since step 4; the
checklist covers both. Replace a sentence when it is too short to show the
meaning, uses a different sense than the card, or is wrong; fix it in place
when the problem is a typo or a slip in an otherwise good sentence (an edit
keeps the original source, which matters for book quotes). A row that
duplicates a deck note at a *different* rank is not something to fix here —
changing ranks would overwrite or orphan a note — so put it in the report for
the user.

Rows that update an existing note carry the note's whole content, old
sentences included, and importing replaces the note with them. The same
rules apply to those sentences, but list any change you make to an existing
note's original content separately in the report, so the user knows their
deck changed beyond the additions. (The pipeline itself also restamps an
updated note's `model_used`; that is expected and not worth reporting.)

## 7. Report and hand off

Write `output/runs/<run>/report.md` and show the user the parts that need
them:

```markdown
# Run <date>

<One paragraph: how many annotations and words, how many corrections and
dropped rows, duplicate groups answered (and how many by the user), new and
updated notes in the diff, and what the LLM steps cost.>

## Needs your decision
<Anything you were not confident enough to change: suspect definitions,
duplicates of deck notes at other ranks, content the user may not want in
their deck. Most important first; say what you would do.>

## Duplicates you answered
<One item per group you answered without asking, as duplicates.md
describes: the disposition, and for a merge the existing note's headword,
part of speech and meaning before and after. These are recorded for good
and a merge rewrites the deck note on import, so the user needs to see each
one here, not a count.>

## What changed
<Counts by kind, with links to words-changelog.md, diff-changelog.md and
duplicates.md rather than repeating them.>

## Cost
<The table from `python scripts/usage_report.py --since <start time>`.>

## Next step
Import `output/diff_from_existing.csv` into Anki (deck "Spanish::Vocab by
Frequency"); the file's directives configure the importer. Originals are in
this directory if anything needs to be undone. <If the run included step 3:
`output/processed_pages.json` lists the pages covered, for updating
`input/book_skip_list.json` (with the caveat about ranges).>
```

Stop there. Do not import into Anki, run the maintenance scripts, or commit
anything; those are the user's calls.
