import argparse
import asyncio
import csv
import itertools
import operator
import pathlib
from collections.abc import Callable, Sequence
from typing import Any, Final

import attrs

from . import _types as types
from . import anki_import, caches, config, conflict_resolver, prompts, query

try:
    from . import word_merger
except ImportError:
    _word_merger_import_successful = False
else:
    _word_merger_import_successful = True

WORD_MERGER_AVAILABLE: Final[bool] = _word_merger_import_successful
del _word_merger_import_successful


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
        "--conflict-cache-behavior",
        type=str,
        default="SKIP",
        help="When resolving conflicts, how to handle cache hits. Options:\n"
        "  SKIP: Use cached value without review.\n"
        "  REVIEW: Load cached values into the UI for review.\n"
        "  IGNORE: Ignore cached values.",
    )

    parser.add_argument(
        "--conflict-cache-path",
        type=pathlib.Path,
        default=pathlib.Path(".cache/merges/merged.json"),
        help="Location of caches for resolving word conflicts.",
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

    if WORD_MERGER_AVAILABLE:
        match args.conflict_cache_behavior:
            case "SKIP":
                conflict_cache_behavior = word_merger.CacheHitBehavior.SKIP
            case "REVIEW":
                conflict_cache_behavior = word_merger.CacheHitBehavior.REVIEW
            case "IGNORE":
                conflict_cache_behavior = word_merger.CacheHitBehavior.IGNORE
            case _:
                raise ValueError(
                    f"Unknown conflict cache behavior: {args.conflict_cache_behavior}"
                )

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

    has_sentences: set[types.Word] = {word for word in words if len(word.sentences)}

    config_file = pathlib.Path(args.config)
    if not config_file.exists():
        print(f"Config file {config.abs()} does not exist.")
        raise OSError("At least one model config is required.")
    else:
        conf_obj = config.Config.from_file(config_file)

    # This seems like it can be dramatically simplified, but we need to make it
    # so that after we have placed a resolved conflict into the output, we can
    # detect that the output is the cache we've already resolved. HOWEVER, we
    # also want to be able to handle the situation where there is a conflict
    # between an updated word in the current output and a NEW word that was not
    # included in the merge resolution.

    # Words from the input CSV
    new_words_wl = conflict_resolver.WordLookup(words)

    # Words from Anki Connect or the exported deck
    current_deck_wl = conflict_resolver.WordLookup(current_deck)

    # If the output CSV already exists, this contains those words, which
    # may or may not have been imported to the deck already.
    output_wl = conflict_resolver.WordLookup(output_words)

    new_and_existing = conflict_resolver.WordLookup(
        itertools.chain(*new_words_wl, current_deck)
    )

    # This gets a single exemplar `Word` object for each word in the new words
    # imported from the CSV (a given word may appear more than once, and
    # it would show up in the `new_words_wl` as a set of words).
    unique_new_words = [next(iter(s)) for s in new_words_wl]

    # Reads a cache from disk mapping sets of words to a single merged word
    merge_cache = conflict_resolver.MergeCache(args.conflict_cache_path)

    # This finds all words which should map to the same card, but cannot
    # because they are not exactly equal (usually the definition is slightly
    # different). These may need manual merging; it is a lazy iterable.
    conflict_sets = conflict_resolver.find_conflicting_word_sets(
        unique_new_words, new_and_existing
    )

    # There are two ways that a word in the conflict set may already have been
    # resolved: 1. it could be in the merge_cache or 2. it could already be
    # in the output CSV. In both of these cases, we don't need to show it to
    # the user for manual merge resolution, so we filter these out.
    to_merge = list(
        conflict_resolver.filter_merged_outputs(conflict_sets, output_wl, merge_cache)
    )

    if to_merge:
        # `textual` is an optional depdendency; if it isn't installed, you cannot
        # use the TUI to do merge resolution.
        if WORD_MERGER_AVAILABLE:
            merge_results = await word_merger.merge_words(
                to_merge,
                cache=merge_cache,
                cache_behavior=conflict_cache_behavior,
            )
            if merge_results is None:
                # The merge UI died or was dismissed without returning
                # results. Merges accepted before that point were saved to
                # the merge cache, so a re-run picks up where this left off.
                raise RuntimeError(
                    "The merge UI exited without returning results; "
                    "re-run to continue from the merges cached so far."
                )

            conflicting_sets = dict(zip(to_merge, merge_results, strict=True))
        else:
            conflict_resolver.print_conflicts(to_merge)

            import sys

            print(
                f"Found {len(to_merge)} unresolved conflicts, cannot proceed.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        conflicting_sets = {}

    # TODO: Handle the case where the word itself is changed during the merging
    # process and this results in a new conflict between the (updated) merged
    # word and the word that you've added.
    # TODO: Add a test of some sort for this.

    diff_words = []
    updated: set[types.Word] = set()
    # This is the logic for which things get sent to the LLM and which don't.
    # At this point we have a list of unique words, we have them deduplicated,
    # and we also have resolved all conflicts or have chosen not to resolve
    # some conflicts. This leaves us with a list of unique words from the input
    # set, and we need to sort them into several buckets:
    #
    # 1. Things that are already in the output set because they were generated
    #   from an earlier run of this program that didn't (e.g. one with fewer
    #   conflicts resolved).
    # 2. Words that are in the output set but have unique new sentences that
    #    need to be merged into the Anki deck.
    # 3. Words are in the existing deck and have a full complement of example
    #    sentences, but need updating because the input set has new example
    #    sentences (e.g. as a result of annotation extraction).
    # 4. Words that are in the existing deck and don't have unique new
    #    sentences.
    # 5. Words that needed manual conflict resolution.
    # 6. Words that need conflict resolution but it hasn't been done.
    # 7. Words that only exist in the inputs.
    #
    # Group 1 shouldn't be sent to the LLM or added to the output. Group 2 should
    # also not be sent to the LLM (unless the earlier run generated fewer sentences
    # than the current maximum), but should be added to the output (and possibly
    # removed from the existing output file?). Group 3 should not be sent to the
    # LLM but should definitely be in the output set. Group 4 should be sent to
    # the LLM if they don't have enough sentences, but otherwise not.
    #
    # Group 5 (which may overlap with other groups) will always need to be
    # added to the output (and possibly earlier versions of it should be
    # removed as well). Group 6 should be excluded entirely, since it needs to
    # be handled after conflicts have been resolved.
    for word in unique_new_words:
        # Groups 1 and 2: Already in the output
        word_set = frozenset(new_and_existing.lookup(word))
        ew = output_wl.lookup(word)

        if ew:
            if word_set in conflicting_sets:
                resolved = conflicting_sets[word_set]

                existing_resolved = conflict_resolver.merge_word_set(ew)
                if set(resolved.sentences) - set(existing_resolved.sentences):
                    # Group 2 - New sentences
                    merged_word = resolved
                    updated.add(merged_word)
                else:
                    # Group 1 - No new sentences
                    continue
            else:
                # Group 1 - Only in output
                continue

        if (resolved := conflicting_sets.get(word_set)) is not None:
            if ew:
                existing_resolved = conflict_resolver.merge_word_set(ew)
                if set(resolved.sentences) - set(existing_resolved.sentences):
                    # Group 2 - New sentences, so we need to update it
                    # Since the values in `ew` haven't already been harmonized
                    # with those in `resolved`, we are going to just update
                    # the sentences part.
                    new_sentences = list(resolved.sentences)
                    sentence_set = set(new_sentences)
                    for sentence in itertools.chain.from_iterable(
                        w.sentences for w in ew
                    ):
                        if sentence not in sentence_set:
                            new_sentences.append(sentence)
                            sentence_set.add(sentence)

                    merged_word = attrs.evolve(resolved, sentences=new_sentences)
                    updated.add(merged_word)
                else:
                    # Group 1: No new sentences and already in output
                    continue
            else:
                # Group 5: We have a resolution, so we know it has been updated.
                merged_word = conflicting_sets[word_set]
                updated.add(merged_word)
        elif word_set in conflicting_sets:
            # Group 6
            w = next(iter(word_set))
            print(f"Skipping {w.word}: {w.meaning} as unresolved conflict.")
            continue
        elif ew:
            # Group 1 - Only exists in output
            continue
        else:
            # No conflicts with the existing deck or among new words, but
            # we can be in groups 3, 4 or 7 here. Merge over the full lookup
            # group rather than `word_set`: the frozenset collapses Word
            # objects that compare equal, which would throw away the deck
            # copy's rank and sentences and produce a duplicate card.
            merged_word = conflict_resolver.merge_word_set(
                tuple(new_and_existing.lookup(word))
            )
            if word not in current_deck_wl or (
                set(word.sentences)
                - set(
                    conflict_resolver.merge_word_set(
                        current_deck_wl.lookup(word)
                    ).sentences
                )
            ):
                # If it is not in the current deck, that is group 7
                # If the merged word has sentences not in the existing deck, that is group 3.
                updated.add(merged_word)

        if len(merged_word.sentences) < args.max_sentences:
            diff_words.append(merged_word)

        if word in has_sentences:
            updated.add(merged_word)

    rank = start_rank_at or 5001

    # TODO: Check size here as well
    needs_header = args.csv_header and not args.output.exists()
    prompt_handler = query.MultiSentencePromptHandler(
        prompts.MULTI_SENTENCE_PROMPT_TEMPLATE,
        system_prompt=prompts.MULTI_SENTENCE_SYSTEM_PROMPT
        % {"dialect": conf_obj.sentences.dialect},
    )

    diff_word_set = set(diff_words)
    include_without_new_sentences = updated - diff_word_set
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
            writer = csv.writer(f)
            for word in diff_out:
                writer.writerow(word.to_csv_line(word))


if __name__ == "__main__":
    asyncio.run(main())
