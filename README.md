# Spanish Anki Card Creator

This is a set of mostly janky scripts that makes Anki cards for my Anki deck. I have created a modified version of [this deck](https://ankiweb.net/shared/info/1350487717). The way I use it is that I suspended *all* cards, and then I started by unsuspending vocab items as I came across them (except ones that were extremely obvious cognates that I didn't need to memorize).

I now keep an org-mode file of new vocabulary I encounter, and use this script to generate new example sentences and cards for it using an LLM of my choice.

The full pipeline is a sequence of several tools run one after the other; [`docs/workflow.md`](docs/workflow.md) documents the whole thing end-to-end, in order. The sections below are the per-tool reference documentation.

## Development and tests

The fast test suite runs in an isolated Python environment and does not require
Calibre or Docker:

```console
tox -e unit
```

The annotation-extractor end-to-end test builds a fresh Calibre library from
the source EPUB and KOReader sidecar fixtures. It runs in a locked, non-root
Docker image so it cannot accidentally use a local Calibre database or
configuration:

```console
tox -e e2e
```

The equivalent Docker commands are:

```console
docker build --network host --target calibre-e2e --tag anki-lang-cards-calibre-e2e .
docker run --network none --rm anki-lang-cards-calibre-e2e
```

Use `docker build --no-cache ...` when validating the build from scratch, and
`docker image inspect anki-lang-cards-calibre-e2e --format '{{.Size}}'` to watch
for unexpected image growth. The verified image is approximately 343 MB; most
of that is Calibre and its system libraries.

The expected annotation output is stored in
`tests/data/annotations.json`. Update it only after reviewing an intentional
extractor behavior change. Run all non-containerized quality checks with:

```console
tox -e py312,py313,lint,format,type
```

You can find a copy of my modified version of the deck (the fields are slightly different in mine than the upstream version) in the [`deck` folder](deck/). The `.apkg` file is the first 5000 words, mostly unmodified, but with the correct fields. Additionally, I have exported in plain text format a large number of additional words and terms that I have come across. You can import these directly into the deck if so desired.

If you want to separately see the code for the cards, it is in [`decks/spanish_vocab_templates`](decks/spanish_vocab_templates/), though keep in mind that if you modify them there, you must manually import your changes in Anki.

## How to use

### CSV input to importable Anki card CSV

The main meat of this script is `python -m anki_lang_cards`, which takes as input a CSV of new words to add and generates example sentences as an importable CSV.

The header of the CSV should look like this:

```
rank,word,meaning,part_of_speech,article,spanish_sentence,english_sentence,example_sentences,model_used
```

Where the fields are:

1. `rank`: the order in the list of most common Spanish vocabulary words. This is a hold-over from the fact that the original deck is "5000 most common Spanish vocabulary words", and they have a sort order to them. This can be empty and a new number will be assigned. The script will attempt to find a matching word and merge any words that already exist in the deck, and adding the `rank` explicitly will make that less likely to fail if you know that there is a match.
2. `word`: the word in Spanish
3. `meaning`: the definition of the word in English
4. `part_of_speech`: The part of speech of the word.
5. `article`: If the word is a noun, add the article that goes with it. This is for stuff like "el agua" where it might not be straightforward to derive the article used from the gender of the noun. It's also useful for stuff like "las tijeras", which are plural, or you can use "un/una" in place of "el/la" in the event that there is a noun where it is way more natural and common to use "un X" rather than "el X".
6. `spanish_sentence`: A sentence in Spanish that uses the relevant word. This is the default sentence displayed if one is not being randomly selected from `example_sentences`. This will generally be empty unless you have generated one or more sentences by another means and want to add additional sentences using this script.
6. `english_sentence`: The English translation of the `spanish_sentence` field.
7. `example_sentences`: A field containing JSON representing a list of objects with `english` and `spanish` attributes, which represents a `spanish`/`english` pair. As with `example_sentences`, this will generally be left blank unless you have generated sentences by another means and want to add more using this script.
8. `model_used`: The model used to generate the sentences (again, left blank, because this script fills it in).

If you do not want to manually construct the input CSVs, you can use either the org mode to CSV converter or the annotation extractor or both (documentation below).

The full invocation is:
```
$ python -m anki_lang_cards --help
usage: __main__.py [-h] [--inputs INPUTS] [--output OUTPUT] [--model-name MODEL_NAME]
                   [--model-service {anthropic,tgw,llm}] [--batch | --no-batch]
                   [--prompt-caching | --no-prompt-caching] [--structured-output | --no-structured-output]
                   [--max-simultaneous-queries MAX_SIMULTANEOUS_QUERIES] [--config CONFIG]
                   [--start-rank-at START_RANK_AT] [--max-retries MAX_RETRIES] [--max-sentences MAX_SENTENCES]
                   [--exported-deck EXPORTED_DECK] [--diff-output DIFF_OUTPUT]
                   [--conflict-cache-path CONFLICT_CACHE_PATH] [--decision-cache DECISION_CACHE]
                   [--note-cache NOTE_CACHE] [--pending-file PENDING_FILE] [--decision-file DECISION_FILE]
                   [--superseded-output SUPERSEDED_OUTPUT] [--no-ui] [--deck-name DECK_NAME] [--csv-header]
                   [--limit LIMIT]

Anki Language Card Creator

options:
  -h, --help            show this help message and exit
  --inputs INPUTS       Path to a CSV containing words and their meanings
  --output OUTPUT       Path to an output CSV file
  --config CONFIG       Config file location
  --start-rank-at START_RANK_AT
  --max-retries MAX_RETRIES
                        How many times to retry a prompt whose response could not be used before giving up on it.
  --max-sentences MAX_SENTENCES
                        If adding a word that is already in the deck and it has greater than this many sentences, do
                        not generate new sentences.
  --exported-deck EXPORTED_DECK
                        Path to a text export of the deck.
  --diff-output DIFF_OUTPUT
                        Path to output a CSV only including new and updated words.
  --conflict-cache-path CONFLICT_CACHE_PATH
                        Legacy merge-wording cache, consulted for merges decided before decisions carried their
                        wording.
  --decision-cache DECISION_CACHE
                        Record of duplicate-group answers; shared with the merge scripts so a decision is only ever
                        made once.
  --note-cache NOTE_CACHE
                        Fetched Anki notes, reused while their mod times are unchanged.
  --pending-file PENDING_FILE
                        Where unanswered duplicate groups are written when no UI can ask about them (exit code 3).
  --decision-file DECISION_FILE
                        An answered pending file (see --pending-file) to fold into the decision cache before
                        duplicates are resolved. Groups still marked "review" stay unanswered and are written back to
                        the pending file.
  --superseded-output SUPERSEDED_OUTPUT
                        Where to list existing notes a merge replaced, which importing cannot delete; written only
                        when there are any.
  --no-ui               Never open the duplicate-resolution screen; write unanswered groups to the pending file and
                        exit 3 instead.
  --deck-name DECK_NAME
  --csv-header          Include a header row when creating the output CSV. Note that Anki's CSV importer has no
                        header-skipping option, so a header row imports as a junk note; it is useful mainly for
                        matching up columns in the import dialog.
  --limit LIMIT         Process at most this many input words; useful for a trial run over a subset before committing
                        to a full one.

model selection:
  --model-name MODEL_NAME, --model MODEL_NAME
                        The model to use, e.g. 'anthropic/claude-sonnet-5'. A model prefixed with a backend's name
                        goes to that backend if it takes prefixes; anything else goes to the 'llm' backend. Defaults
                        to [models] default_model from the config; one of the two is required.
  --model-service {anthropic,tgw,llm}
                        Force a backend instead of choosing one from the model name.
  --batch, --no-batch   Send requests as a discounted batch job, which can take a while to come back. On by default
                        where the backend supports it.
  --prompt-caching, --no-prompt-caching
                        Cache the shared system prompt between requests. On by default where the backend supports it.
  --structured-output, --no-structured-output
                        Have the model's API enforce the response format, so responses always parse. On by default
                        where the model supports it.
  --max-simultaneous-queries MAX_SIMULTANEOUS_QUERIES
                        Maximum number of requests in flight at once when not batching.
```

To better merge with the existing deck, you should either be running [Anki Connect](https://ankiweb.net/shared/info/2055492159) or export the deck from Anki in text format (TSV) and provide the path to the exported deck with the `--exported-deck` option. If you are running Anki Connect, it will automatically find any existing cards to be updated and the rank for the next card. For the moment, you still need to manually import the CSV that is produced by this script.

### Org mode to CSV converter

If you are using org mode like I am, keep a file that looks like this:

```
   - [ ] rehen = hostage
   - [ ] escudriñar = scrutinize
   - [ ] borla = tassel, pompom
   - [ ] crocante = crunchy
   - [ ] chamba = daily grind (Mexican slang), fluke, chance
   - [ ] cianuro = cyanide
   - [ ] ? = to button a shirt
   - [ ] descuajarse = ?
```

It works better if you have the right and left sides filled out for the specific meaning you care about, but I find that Claude Sonnet 3.5 works pretty well for this in most cases even if you don't.

You then have to pipe the input into `anki_lang_cards.org_to_csv`, which will create `input/words_from_org_mode.csv` (or whatever you call the output). I usually copy the input and pipe in like this:

```
xclip -selection c -o | python -m anki_lang_cards.org_to_csv
```

### Cards from annotations

If you use `calibre` and `KOReader`, you can sync annotations to your `calibre` database and automatically extract the sentences they are part of. This will only extract annotations from books matching the target language.

#### Finding libraries with annotations

If you don't remember which of your Calibre libraries contains the annotated books, `anki_lang_cards.find_libraries` will scan one or more directory trees for Calibre libraries and report how many annotated books and annotations each one has:

```
$ python -m anki_lang_cards.find_libraries ~ -l es
/home/me/Calibre/SpanishLibrary: 12 annotated books, 345 annotations
/home/me/Calibre/Manuals: no 'raw_sidecar' column

Found 2 Calibre libraries (1 with annotations).
```

The databases are opened read-only, so this is safe to run while Calibre is open. `--annotated-only` hides libraries without annotations, and `--sidecar-column` changes which custom column it looks for (same default as the annotation extractor).

#### Annotation extractor

The annotation extraction occurs in two phases. The first phase runs `anki_lang_cards.annotation_extractor`, which extracts annotations from the calibre database, it is invoked as such:

```
usage: annotation_extractor.py [-h] -i CALIBRE_DIRECTORY [-l LANGUAGE]
                               [--context-sentences CONTEXT_SENTENCES]
                               --output OUTPUT
                               [--sidecar-column SIDECAR_COLUMN] [--csv]
                               [--json]
                               [--base-output-filename BASE_OUTPUT_FILENAME]

Extract Calibre annotations

options:
  -h, --help            show this help message and exit
  -i CALIBRE_DIRECTORY, --calibre-directory CALIBRE_DIRECTORY
  -l LANGUAGE, --language LANGUAGE
  --context-sentences CONTEXT_SENTENCES
  --output OUTPUT
  --sidecar-column SIDECAR_COLUMN
  --csv
  --json
  --base-output-filename BASE_OUTPUT_FILENAME
```

Once that is done, you should have in your output directory a CSV or JSON or both containing extracted annotations. It will skip anything not in the target language. `--context-sentences` tells the script to extract a certain number of sentences for context (in the next stage, these will be sent to an LLM to make them pithier and more suitable as example sentences).

#### Annotations to CSV

You then want to invoke `anki_lang_cards.annotations_to_csv` to generate cards for these. It is invoked as such:

```
$ python -m anki_lang_cards.annotations_to_csv --help
usage: annotations_to_csv.py [-h] [-i INPUT] [-o OUTPUT] [--dictionary DICTIONARY]
                             [--hunspell-dict-path HUNSPELL_DICT_PATH] [--model-name MODEL_NAME]
                             [--model-service {anthropic,tgw,llm}] [--batch | --no-batch]
                             [--prompt-caching | --no-prompt-caching] [--structured-output | --no-structured-output]
                             [--max-simultaneous-queries MAX_SIMULTANEOUS_QUERIES] [--max-retries MAX_RETRIES]
                             [--config CONFIG] [--skip-list SKIP_LIST]
                             [--processed-pages-output PROCESSED_PAGES_OUTPUT] [--invalidate-response-cache]
                             [--skip-response-cache] [--csv-header] [--limit LIMIT]
                             [--prompt-token-budget PROMPT_TOKEN_BUDGET]

Convert extracted and combined annotations into Anki cards

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
  -o OUTPUT, --output OUTPUT
  --dictionary DICTIONARY
                        Path to a dictionary in STARDICT format. This can also be specified in the config file under
                        annotations.dictionaries.
  --hunspell-dict-path HUNSPELL_DICT_PATH
                        Path to the hunspell dictionary for headword stemming
  --max-retries MAX_RETRIES
                        How many times to retry a prompt whose response could not be used before giving up on it.
  --config CONFIG       Config file location
  --skip-list SKIP_LIST
                        List of books to skip, or page ranges within books
  --processed-pages-output PROCESSED_PAGES_OUTPUT
                        Path to output a JSON file containing processed pages
  --invalidate-response-cache
                        If specified, delete any existing caches for this given response and replace them with the new
                        query response.
  --skip-response-cache
                        If specified, don't use existing caches, but also don't replace them.
  --csv-header          Include a header row in the output CSV. Note that Anki's CSV importer has no header-skipping
                        option, so a header row imports as a junk note; it is useful mainly for matching up columns in
                        the import dialog.
  --limit LIMIT         Process at most this many annotation entries; useful for a trial run over a subset before
                        committing to a full one. The processed-pages output is not written for a limited run.
  --prompt-token-budget PROMPT_TOKEN_BUDGET
                        Pack annotation entries into prompts of at most roughly this many input tokens (estimated as
                        characters / 4, system prompt included); an entry too big to share a prompt still gets one to
                        itself. Bigger prompts mean fewer requests and less repeated instruction text, but more
                        entries for the model to handle in each response. Default: 8192.

model selection:
  --model-name MODEL_NAME, --model MODEL_NAME
                        The model to use, e.g. 'anthropic/claude-sonnet-5'. A model prefixed with a backend's name
                        goes to that backend if it takes prefixes; anything else goes to the 'llm' backend. Defaults
                        to [models] default_model from the config; one of the two is required.
  --model-service {anthropic,tgw,llm}
                        Force a backend instead of choosing one from the model name.
  --batch, --no-batch   Send requests as a discounted batch job, which can take a while to come back. On by default
                        where the backend supports it.
  --prompt-caching, --no-prompt-caching
                        Cache the shared system prompt between requests. On by default where the backend supports it.
  --structured-output, --no-structured-output
                        Have the model's API enforce the response format, so responses always parse. On by default
                        where the model supports it.
  --max-simultaneous-queries MAX_SIMULTANEOUS_QUERIES
                        Maximum number of requests in flight at once when not batching.
```

This script uses `hunspell` to find all potential stems for the library (default location of the Hunspell dictionary is `/usr/share/hunspell/es_ES.dic`, but this is something you need to install separately for each language. You may specify multiple dictionaries as long as they are in the STARDICT format. One source of bilingual stardict dictionaries is [this list of Wiktionary dictionaries](https://github.com/Vuizur/Wiktionary-Dictionaries). The value passed to `--dictionary` should be the root name of all the dictionary files. For example, if I have a dictionary in `input/dicts/sp-en-wik/` with files `sp-en-wik.dict.dz`, `sp-en-wik.idx`, `sp-en-wik.ifo`, `sp-en-wik.syn`, I would specify `--dictionary inputs/dict/sp-en-wik/sp-en-wik`. These can also be configured in the config file (see example config in the `examples` folder).

To make it easier to resume broken pipelines, all LLM responses are cached in `.cache/annotations`. If you change the prompt or the words that go into it, it will invalidate the cache. You can also manually invalidate the cache or skip the cache using the `--invalidate-cache` and `--skip-cache` options. The first one will overwrite any existing cache files, and `--skip-cache` will ignore the cache and not create an entry.

There is also an option to specify a "skip-list" with a domain-specific language in JSON that will let you skip entire books or subsets of the books. Currently this DSL needs to be specified by hand. The skip list is a JSON list of mappings. Each mapping is a rule that is applied, and anything matching any of these rules is skipped.

The top-level keys are `path`, `author`, `language` and `title`, and if no second-level patterns are found, this excludes the entire book of any book specified. `path` is relative to the base of the calibre directory, and you should be able to find it in the output of `annotation_extractor`. Within a book (second level), you can also filter by `page_numbers` or `chapters`. If the second-level patterns are present, they apply only to books matching the top-level patterns. Here are some examples:

```json
[
    {"path": "Frank Herbert/Dune (563)"},
    {
        "title": "The Hobbit",
        "author": "J. R. R. Tolkein",
        "page_numbers": [1, {"range": [10, 20]}, 25]
    }
]
```

This filters out the book *Dune*, and it skips any annotation on pages 1, 10-20 and 25.

```json
[
  {
    "title": {"pattern": "Harry Potter.*"},
    "chapters": {"pattern": "Acknowledgements|Prologue|Dedication"}
  }
]
```

This filters out only the "Acknowledgements", "Dedication" and "Prologue" chapters of all Harry Potter books.
