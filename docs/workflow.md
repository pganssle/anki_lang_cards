# End-to-end workflow

This is the "what do I actually run" document. The [README](../README.md)
documents each tool's flags in detail; this file is the sequence of steps, in
order, for going from highlights in KOReader (or a word list in org mode) to
new cards in Anki.

There are two entry paths that converge on the same card-generation step:

```
KOReader highlights ──sync──> Calibre library (#raw_sidecar column)
                                     │
                                     ▼
                     anki_lang_cards.annotation_extractor
                                     │  output/annotations.json
                                     ▼
                     anki_lang_cards.combine_annotations
                                     │  output/combined_annotations.json
                                     ▼
                     anki_lang_cards.annotations_to_csv  (LLM)
                                     │  output/words_from_annotations.csv
                                     ▼
 org-mode word list ──> org_to_csv ──┤
                                     ▼
                        python -m anki_lang_cards  (LLM + dedup/merge)
                                     │  output/diff_from_existing.csv
                                     ▼
                        manual import into Anki
                                     │
                                     ▼
              scripts/find_duplicates.py, scripts/compact_ranks.py
```

## The short version

With everything set up (see prerequisites below), a full annotation run is:

```console
python -m anki_lang_cards.annotation_extractor -i ~/Calibre/SpanishLibrary -l es --context-sentences 2 --output output --json
python -m anki_lang_cards.combine_annotations -i output/annotations.json -o output/combined_annotations.json
python -m anki_lang_cards.annotations_to_csv --model-service anthropic
# review output/words_from_annotations.csv by hand
python -m anki_lang_cards --model-service anthropic --inputs output/words_from_annotations.csv
# import output/diff_from_existing.csv into Anki
```

Everything below is the detail on each of those steps.

## Prerequisites

Most of this only needs to be done once:

- **Config file**: a `config.toml` in the working directory (see
  [`examples/example_config.toml`](../examples/example_config.toml)) with at
  least an Anthropic API key, and optionally `annotations.dictionaries`.
  `sentences.dialect` controls which variety of Spanish generated sentences
  are written in (default "international Spanish"); note that changing it
  changes the prompts, which invalidates any cached LLM responses.
- **Dictionaries** (annotation path only): a hunspell dictionary for the
  target language (default location `/usr/share/hunspell/es_ES.dic`) and one
  or more bilingual STARDICT dictionaries, e.g. from
  [this list of Wiktionary dictionaries](https://github.com/Vuizur/Wiktionary-Dictionaries).
- **Calibre sidecar column** (annotation path only): the Calibre library
  needs a custom column holding the raw KOReader sidecar JSON. The extractor
  looks for a column labeled `raw_sidecar` by default; you can create one
  with:

  ```console
  calibredb add_custom_column --library-path <library> raw_sidecar "Raw sidecar JSON" comments
  ```

- **Anki Connect**: for the card-generation step, either have Anki running
  with the [Anki Connect](https://ankiweb.net/shared/info/2055492159) add-on,
  or export the deck as TSV and pass `--exported-deck`. The maintenance
  scripts require Anki Connect.
- The tools write to `input/`, `output/` and `.cache/` relative to the
  current working directory, so run everything from the repository root (or
  at least from a consistent directory, so the caches actually get reused).

## Step 0: find the Calibre library

If you don't remember which Calibre library has the annotated books in it:

```console
python -m anki_lang_cards.find_libraries ~ -l es
```

This walks the given directories looking for Calibre libraries (anything with
a `metadata.db`) and reports how many annotated books and annotations each
one has, e.g.:

```
/home/me/Calibre/SpanishLibrary: 12 annotated books, 345 annotations
/home/me/Calibre/Manuals: no 'raw_sidecar' column
```

It opens the databases read-only, so it's fine to run while Calibre is open.
Use `--annotated-only` to hide the empty ones.

## Step 1: get the annotations into Calibre

Highlight words in KOReader as you read. KOReader stores the highlights in a
sidecar file next to the book; sync them into the Calibre library however you
normally do — the rest of the pipeline only cares that the raw sidecar JSON
ends up in the `#raw_sidecar` column for each book.

## Step 2: extract annotations from the library

```console
python -m anki_lang_cards.annotation_extractor -i <calibre-library> -l es --context-sentences 2 --output output --json
```

This reads the sidecar column from the Calibre database, finds each highlight
in the book's EPUB, and pulls out the highlighted word along with
`--context-sentences` sentences of surrounding context (the context is what
the LLM later turns into example sentences). Books not in the target language
are skipped. Output lands in `output/annotations.json` (there's also `--csv`
if you want to eyeball it in a spreadsheet).

## Step 3: combine overlapping annotations

```console
python -m anki_lang_cards.combine_annotations -i output/annotations.json -o output/combined_annotations.json
```

Consecutive highlights with overlapping context (i.e. several words
highlighted in the same passage) get merged into a single combined annotation
so that the next step generates one query per passage instead of one per
word. This step is fast and has no external dependencies; there's no reason
to skip it even if you don't think you have overlapping highlights.

## Step 4: turn annotations into a word CSV

```console
python -m anki_lang_cards.annotations_to_csv --model-service anthropic
```

The defaults line up with the previous step (`-i
output/combined_annotations.json`, `-o output/words_from_annotations.csv`),
so usually no other flags are needed. For each annotation this looks up the
highlighted word in the STARDICT dictionaries (using hunspell to find the
headword from an inflected form) and asks the LLM to determine the
dictionary form of the word as used, a definition for that specific sense,
and a pithy example sentence extracted from the context.

Things to know:

- **Skip list**: `input/book_skip_list.json` (see the README for the DSL)
  excludes whole books or page/chapter ranges — useful for books you've
  already processed, or for things highlighted for reasons other than
  vocabulary. `--processed-pages-output` writes `output/processed_pages.json`,
  which is handy as a starting point for updating the skip list after a run.
- **Caching**: all LLM responses are cached in `.cache/annotations`, so
  re-running after an interruption (or after fixing something downstream) is
  cheap and doesn't re-query the model. `--invalidate-response-cache` and
  `--skip-response-cache` override this when you actually want fresh
  responses.
- **Trial runs**: `--limit N` processes only the first N annotation entries,
  which is a cheap way to sanity-check the output format before committing
  to a full run (the responses are cached, so the full run reuses them). A
  limited run doesn't write the processed-pages file, since it would
  otherwise claim pages were processed that never were.

After this step, review `output/words_from_annotations.csv` by hand — fix up
meanings, drop junk annotations, etc. This is the cheapest place in the
pipeline to catch mistakes.

## Alternative entry: org-mode word list

For words collected in an org-mode file rather than from books, copy the
relevant list and run:

```console
xclip -selection c -o | python -m anki_lang_cards.org_to_csv
```

This converts entries like `- [ ] rehen = hostage` into the same CSV format,
writing `input/words_from_org_mode.csv` by default (`--if-exists append` is
the default, so repeated runs accumulate). Either side of the `=` can be `?`
and the LLM will fill it in.

## Step 5: generate the cards

Have Anki running with Anki Connect (or pass `--exported-deck` with a TSV
export of the deck), then:

```console
python -m anki_lang_cards --model-service anthropic --inputs output/words_from_annotations.csv
```

(The default `--inputs` is `input/words.csv`, so point it at whichever CSV
the earlier steps produced. `--limit N` processes only the first N input
words — useful for a trial run, ideally with `--output` and `--diff-output`
pointed somewhere disposable.)

This generates additional example sentences for each word, assigns ranks, and
merges the input against the existing deck: words already in the deck are
matched up and updated rather than duplicated. When an input word conflicts
with an existing card, a terminal UI pops up to let you resolve the merge;
those decisions are cached in `.cache/merges/merged.json`, so a re-run
doesn't ask again (`--conflict-cache-behavior` controls this).

Two files come out:

- `output/output.csv` — everything;
- `output/diff_from_existing.csv` — only new and updated words. **This is
  the one to import.**

## Step 6: import into Anki

This part is still manual: in Anki, import `output/diff_from_existing.csv`
into the deck (default deck name is `Spanish::Vocab by Frequency`). The
first field is the rank, so Anki's default duplicate handling (match on
first field) makes updated words overwrite the existing notes rather than
creating duplicates.

The CSVs start with Anki file directives (`#separator:Comma`,
`#html:true`, `#deck:...`, `#columns:...`) that configure the importer
from the file itself — in particular "Allow HTML in fields" is forced on.
Importing with it off makes Anki HTML-escape every field (`'` becomes
`&#x27;`, `"` becomes `&quot;`), mangling both the visible text and the
JSON fields; with the directives in place the fields survive
byte-for-byte and there is no GUI setting to get wrong. If you'd rather
have a spreadsheet-style header row (no directives), pass `--csv-header`
to the previous step.

Should a misconfigured import happen anyway, the JSON-bearing fields are
written with `<`, `>` and `&` as `\uXXXX` escapes, so the only damage is
`&quot;`-style entities — which the repair tooling (see step 7) undoes.

## Step 7: periodic maintenance

Not part of every run, but worth doing occasionally, with Anki Connect
running:

- **`python scripts/find_duplicates.py`** — finds notes in the deck that are
  duplicates of each other, walks you through merging them in the same
  terminal merge UI (decisions cached in `.cache/merges/duplicates.json`),
  and applies the merges through Anki Connect: the surviving note is updated,
  its cards are unsuspended, and the losers are deleted. `--accept` takes all
  proposed merges without review.
- **`python scripts/compact_ranks.py`** — finds gaps and duplicates in the
  `Rank` field above `--start-at` (default 5000, i.e. the words added by this
  pipeline rather than the original frequency deck) and renumbers to close
  them. Supports `--dry-run`.
- **`python scripts/repair_json_fields.py`** — re-parses the `Sentence
  Source` and `Extra Sentences` fields on every note (repairing historical
  mangling: HTML entities from imports, hand-doubled or stripped `\"`
  escapes, sources stored as nested JSON strings) and rewrites them in the
  canonical HTML-inert form; also unescapes the HTML entities that imports
  with "allow HTML" disabled left in the plain-text fields (`Spanish`,
  `English`, `Definition`, ...). Dry run by default; pass `--apply` to
  write.

Both scripts take a `--cache-file` argument to avoid re-fetching the whole
deck from Anki Connect on repeated runs.
