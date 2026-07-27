import argparse
import asyncio
import csv
import itertools
import operator
import pathlib
import sys
from collections.abc import Callable, Sequence
from typing import Any, Final

import attrs

from . import _types as types
from . import (
    anki_import,
    apply_merges,
    caches,
    config,
    conflict_resolver,
    dedup_flow,
    prompts,
    query,
    triage,
)

try:
    import textual  # noqa: F401
except ImportError:
    _textual_import_successful = False
else:
    _textual_import_successful = True

UI_AVAILABLE: Final[bool] = _textual_import_successful
del _textual_import_successful

EXIT_DECISIONS_NEEDED: Final[int] = 3
"""The run stopped before the LLM because duplicates need answers; the
pending file says which. Distinct from 1 so a harness can tell "answer and
re-run" from "look at a traceback"."""


def _unique(s: Sequence[str], key: Callable[[str], Any] | None = None) -> Sequence[str]:
    seen: dict = {}
    out: list[str] = []
    if key is None:

        def key(x: str) -> str:
            return x

    for entry in s:
        if (entry_key := key(entry)) not in seen:
            out.append(entry)
        else:
            existing_entry = seen[entry_key]
            existing_entry.update(entry)

    return out


def _make_api_obj(
    conf_obj: config.Config, service: str, model_name: str | None
) -> query.ModelAPIBase:
    if service == "anthropic":
        if conf_obj.anthropic is None:
            raise ValueError("Anthropic model selected but no config found.")
        if model_name is None and conf_obj.anthropic.default_model is not None:
            model_name = conf_obj.anthropic.default_model
        query_cacher = caches.QueryCacher(
            cache_type=caches.CacheTypes.SENTENCES,
            model_name=model_name or query.AnthropicAPI.DEFAULT_MODEL,
            model_service=service,
        )
        model_api = query.AnthropicAPI(
            api_key=conf_obj.anthropic.api_key,
            query_cacher=query_cacher,
            model_name=model_name,
            max_connections=1,
        )
    elif service == "tgw":
        if conf_obj.tgw is None:
            raise ValueError("tgw model selected but no config found.")

        model_api = query.TGWModelAPI(
            api_base_url=conf_obj.tgw.api_base_url,
            request_timeout=conf_obj.tgw.request_timeout,
            max_connections=conf_obj.tgw.max_connections,
            max_tokens=conf_obj.tgw.max_tokens,
            model_name=model_name,
        )
    else:
        raise ValueError(f"Unknown model service: {service}")

    return model_api


@attrs.frozen
class Classification:
    """Where each word the run touched goes next."""

    to_llm: tuple[types.Word, ...]
    """Under the sentence target; sent for generation."""

    include_directly: tuple[types.Word, ...]
    """Changed and already full of sentences; written straight to the output."""

    skipped: tuple[types.Word, ...]
    """Already emitted by an earlier run with nothing new to add."""


def _decided_ranks(outcome: apply_merges.Outcome) -> frozenset[int]:
    """Ranks changed by an actual answer, as opposed to mechanical dedup.

    A deck note merged with an equal input copy is bookkeeping, not an update
    — the note reads the same — so automatic merges do not count. Decided
    merges, rewrites and splits all changed what a card says.
    """
    ranks: set[int] = {w.rank for w in outcome.revised if w.rank is not None}
    for merge in outcome.merges:
        if merge.decided:
            ranks |= {
                w.rank for w in (merge.result, *merge.sources) if w.rank is not None
            }
    for split in outcome.splits:
        ranks |= {w.rank for w in (*split.cards, *split.sources) if w.rank is not None}
    return frozenset(ranks)


def _classify_for_generation(
    selected: Sequence[types.Word],
    *,
    inputs_wl: conflict_resolver.WordLookup,
    deck_wl: conflict_resolver.WordLookup,
    output_wl: conflict_resolver.WordLookup,
    decided_ranks: frozenset[int],
    max_sentences: int,
) -> Classification:
    """Sort the deduplicated words into what happens to each next.

    The reasoning is the old classification's, applied after merging instead
    of interleaved with it: a word already emitted to the output CSV is only
    interesting again when it gained sentences (and then it keeps its new
    sentences first, with the output's extras appended); a word under the
    sentence target goes to the LLM whether or not anything else changed; a
    word at the target is written directly only when something about it
    actually changed — an answered decision, sentences the deck lacks, being
    new, or the input carrying sentences of its own.
    """
    to_llm: list[types.Word] = []
    include: list[types.Word] = []
    skipped: list[types.Word] = []

    for word in selected:
        prior = output_wl.lookup(word)
        if prior:
            prior_merged = conflict_resolver.merge_word_set(tuple(prior))
            if not set(word.sentences) - set(prior_merged.sentences):
                skipped.append(word)
                continue
            sentences = list(word.sentences)
            seen = set(sentences)
            for sentence in itertools.chain.from_iterable(w.sentences for w in prior):
                if sentence not in seen:
                    sentences.append(sentence)
                    seen.add(sentence)
            word = attrs.evolve(word, sentences=tuple(sentences))
            updated = True
        else:
            in_deck = deck_wl.lookup(word)
            updated = (
                (word.rank is not None and word.rank in decided_ranks)
                or not in_deck
                or bool(
                    set(word.sentences)
                    - set(conflict_resolver.merge_word_set(tuple(in_deck)).sentences)
                )
            )
        if any(input_word.sentences for input_word in inputs_wl.lookup(word)):
            updated = True

        if len(word.sentences) < max_sentences:
            to_llm.append(word)
        elif updated:
            include.append(word)
        else:
            skipped.append(word)

    return Classification(
        to_llm=tuple(to_llm),
        include_directly=tuple(include),
        skipped=tuple(skipped),
    )


async def main():
    parser = argparse.ArgumentParser(description="Anki Language Card Creator")
    parser.add_argument(
        "--inputs",
        type=pathlib.Path,
        default="input/words.csv",
        help="Path to a CSV containing words and their meanings",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default="output/output.csv",
        help="Path to an output CSV file",
    )
    parser.add_argument(
        "--model-service",
        type=str,
        default="tgw",
        help="Which model service to use. Options are:\n"
        "    tgw: Text Generation Web\n"
        "    anthropic: Anthropic",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="The specific model to use. If unspecified, the default value "
        "for the chosen model is selected, which is:\n"
        f"    tgw: {query.TGWModelAPI.DEFAULT_MODEL}\n"
        + f"    anthropic: {query.AnthropicAPI.DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default="config.toml",
        help="Config file location",
    )
    parser.add_argument(
        "--start-rank-at",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of failed responses from the LLM before giving up.",
    )

    parser.add_argument(
        "--max-sentences",
        type=int,
        default=7,
        help="If adding a word that is already in the deck and it has greater "
        "than this many sentences, do not generate new sentences.",
    )

    parser.add_argument(
        "--exported-deck",
        type=pathlib.Path,
        default=None,
        help="Path to a text export of the deck.",
    )
    parser.add_argument(
        "--diff-output",
        type=pathlib.Path,
        default="output/diff_from_existing.csv",
        help="Path to output a CSV only including new and updated words.",
    )

    parser.add_argument(
        "--conflict-cache-path",
        type=pathlib.Path,
        default=pathlib.Path(".cache/merges/merged.json"),
        help="Legacy merge-wording cache, consulted for merges decided "
        "before decisions carried their wording.",
    )

    parser.add_argument(
        "--decision-cache",
        type=pathlib.Path,
        default=pathlib.Path(".cache/merges/decisions.json"),
        help="Record of duplicate-group answers; shared with the merge "
        "scripts so a decision is only ever made once.",
    )

    parser.add_argument(
        "--pending-file",
        type=pathlib.Path,
        default=pathlib.Path("output/merge_candidates.json"),
        help="Where unanswered duplicate groups are written when no UI can "
        "ask about them (exit code 3).",
    )

    parser.add_argument(
        "--superseded-output",
        type=pathlib.Path,
        default=pathlib.Path("output/superseded_notes.csv"),
        help="Where to list existing notes a merge replaced, which importing "
        "cannot delete; written only when there are any.",
    )

    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Never open the duplicate-resolution screen; write unanswered "
        "groups to the pending file and exit 3 instead.",
    )

    parser.add_argument("--deck-name", type=str, default="Spanish::Vocab by Frequency")
    parser.add_argument(
        "--csv-header",
        action="store_true",
        help="Include a header row when creating the output CSV. Note that "
        "Anki's CSV importer has no header-skipping option, so a header row "
        "imports as a junk note; it is useful mainly for matching up "
        "columns in the import dialog.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many input words; useful for a trial "
        "run over a subset before committing to a full one.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    deck_name = args.deck_name

    word_getter = operator.attrgetter("word")

    words = types.Word.words_from_csv(args.inputs)
    if args.limit is not None and args.limit < len(words):
        print(f"Limiting to the first {args.limit} of {len(words)} input words.")
        words = words[: args.limit]
    if args.output.exists():
        output_words = types.Word.words_from_csv(args.output)
    else:
        output_words = []

    if args.exported_deck is not None:
        current_deck = anki_import.load_anki_words(args.exported_deck)
        start_rank_at = args.start_rank_at
        if start_rank_at is None:
            start_rank_at = (
                max(
                    itertools.chain(
                        (word.rank for word in current_deck),
                        (word.rank for word in output_words),
                    )
                )
                + 1
            )
    else:
        print("Loading words from Anki Connect")
        async with anki_import.create_client() as client:
            current_deck_notes = await client.retrieve_matching_words(deck_name, words)
            current_deck = list(current_deck_notes.values())
        print(f"Loaded {len(current_deck)} words")

        async with anki_import.create_client() as client:
            if not args.start_rank_at:
                print("Determining next rank")
                start_rank_at = (
                    await client.find_highest_rank(deck_name, max_range=len(words)) + 1
                )
                print(f"Starting with {start_rank_at}")
            else:
                start_rank_at = args.start_rank_at

    config_file = pathlib.Path(args.config)
    if not config_file.exists():
        print(f"Config file {config.abs()} does not exist.")
        raise OSError("At least one model config is required.")
    else:
        conf_obj = config.Config.from_file(config_file)

    # Words from the input CSV
    inputs_wl = conflict_resolver.WordLookup(words)

    # Words from Anki Connect or the exported deck
    current_deck_wl = conflict_resolver.WordLookup(current_deck)

    # If the output CSV already exists, this contains those words, which
    # may or may not have been imported to the deck already.
    output_wl = conflict_resolver.WordLookup(output_words)

    decisions = triage.DecisionCache(args.decision_cache)
    merge_cache = conflict_resolver.MergeCache(args.conflict_cache_path)

    result = await dedup_flow.resolve_duplicates(
        words,
        current_deck,
        decisions=decisions,
        merge_cache=merge_cache,
        interactive=UI_AVAILABLE and not args.no_ui,
    )
    outcome = result.outcome

    if result.rank_only_decisions:
        print(
            f"WARNING: {len(result.rank_only_decisions)} recorded decisions "
            "match this run's notes by rank only — the notes' text has "
            "changed since they were answered. Look before trusting these:"
        )
        for group in result.rank_only_decisions:
            for w in sorted(group):
                print(f"  {w.rank} {w.word} ({w.part_of_speech}) {w.meaning}")

    if result.interrupted:
        # Nothing reaches the LLM until every duplicate this import raises
        # has an answer; a silent skip here used to hide unresolved groups.
        print(f"{len(result.pending)} duplicate groups still need an answer:")
        for candidate in result.pending:
            for line in candidate.summarize():
                print(f"  {line}")
        if UI_AVAILABLE and not args.no_ui:
            print("Answers made so far were saved; re-run to continue.")
            sys.exit(1)
        dedup_flow.write_pending_file(args.pending_file, result.pending)
        print(
            f'Edit {args.pending_file} (set "disposition"; a merge with no '
            '"replacements" accepts the proposal) and re-run.'
        )
        sys.exit(EXIT_DECISIONS_NEEDED)

    # What this run is about: everything an answer or a merge touched, plus
    # the inputs themselves. Untouched deck notes stay out of the output, and
    # rank-less words can only have come from the inputs or a split.
    wanted = outcome.touched_ranks() | {w.rank for w in words if w.rank is not None}
    split_cards = {id(card) for split in outcome.splits for card in split.cards}
    selected = [
        w
        for w in outcome.words
        if w.rank is None or w.rank in wanted or id(w) in split_cards
    ]

    classified = _classify_for_generation(
        selected,
        inputs_wl=inputs_wl,
        deck_wl=current_deck_wl,
        output_wl=output_wl,
        decided_ranks=_decided_ranks(outcome),
        max_sentences=args.max_sentences,
    )
    diff_words = list(classified.to_llm)

    rank = start_rank_at or 5001

    # TODO: Check size here as well
    # By default new CSVs start with Anki file directives, which configure
    # the GUI importer (separator, allow HTML, deck, column mapping) from
    # the file itself; --csv-header replaces them with a spreadsheet-style
    # header row instead.
    needs_header = args.csv_header and not args.output.exists()
    needs_directives = not args.csv_header and not args.output.exists()
    prompt_handler = query.MultiSentencePromptHandler(
        prompts.MULTI_SENTENCE_PROMPT_TEMPLATE,
        system_prompt=prompts.MULTI_SENTENCE_SYSTEM_PROMPT
        % {"dialect": conf_obj.sentences.dialect},
    )

    # Directly-included rows are written with require_rank; a rank-less one —
    # two annotation words merging while already at the sentence target, a
    # split's extra card — gets a fresh rank here instead of crashing there.
    include_without_new_sentences: list[types.Word] = []
    for word in classified.include_directly:
        if word.rank is None:
            word = attrs.evolve(word, rank=rank)
            rank += 1
        include_without_new_sentences.append(word)
    diff_words_lookup = conflict_resolver.WordLookup(diff_words)
    diff_out: list[types.Word] = []
    if diff_words:
        print(f"Generating sentences for {len(diff_words)} words")
        model_api = _make_api_obj(
            conf_obj=conf_obj,
            service=args.model_service,
            model_name=args.model_name,
        )

        async with model_api as api:
            query_results = api.make_queries(
                diff_words,
                prompt=prompt_handler,
                max_invalid_output_retries=args.max_retries,
            )

            with open(args.output, "at") as f:
                if needs_directives:
                    f.write(
                        "\n".join(types.Word.make_anki_directives(deck_name)) + "\n"
                    )
                writer = csv.writer(f)

                if needs_header:
                    writer.writerow(types.Word.make_csv_header())

                for word in include_without_new_sentences:
                    diff_out.append(word)
                    writer.writerow(word.to_csv_line(require_rank=True))
                    f.flush()

                async for word in query_results:
                    print(f"Adding {word.word}")
                    diff_words_lookup.add(word)
                    word = conflict_resolver.merge_word_set(
                        diff_words_lookup.lookup(word)
                    )
                    if not word.rank:
                        word.rank = rank
                        rank += 1
                        diff_words_lookup.add(word)

                    diff_out.append(word)
                    writer.writerow(word.to_csv_line(require_rank=True))
                    f.flush()

    else:
        print("No new words found.")

    print(f"Found {len(diff_out)} new or updated words.")
    print("\n".join(map(word_getter, diff_out)))
    if diff_words and diff_out:
        with open(args.diff_output, "wt") as f:
            if args.csv_header:
                csv.writer(f).writerow(types.Word.make_csv_header())
            else:
                f.write("\n".join(types.Word.make_anki_directives(deck_name)) + "\n")
            writer = csv.writer(f)
            for word in diff_out:
                writer.writerow(word.to_csv_line(require_rank=True))

    superseded = apply_merges.superseded_notes(outcome, current_deck)
    if superseded:
        print(
            "\nNotes Anki will still hold that nothing will update — "
            "delete these in Anki:"
        )
        for word in superseded:
            print(f"  {word.rank} {word.word} ({word.part_of_speech}) {word.meaning}")
        if not args.superseded_output.parent.exists():
            args.superseded_output.parent.mkdir(parents=True, exist_ok=True)
        with args.superseded_output.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("rank", "word", "part_of_speech", "meaning"))
            writer.writerows(
                (word.rank, word.word, str(word.part_of_speech), word.meaning)
                for word in superseded
            )
        print(f"  (also written to {args.superseded_output})")

    discarded = apply_merges.discarded_imports(outcome, words, current_deck)
    if discarded:
        print(
            f"\nWARNING: {len(discarded)} notes came out exactly as the deck "
            "already has them, so the imported version went nowhere:"
        )
        for word in discarded:
            print(f"  {word.rank} {word.word} ({word.part_of_speech}) {word.meaning}")


if __name__ == "__main__":
    asyncio.run(main())
