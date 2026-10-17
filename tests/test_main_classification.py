"""Tests for the pipeline's post-merge classification.

`_classify_for_generation` decides what happens to each word after
deduplication: LLM, straight to the output, or nothing. Each case here is one
of the old inline classification's groups, preserved across the rewrite.
"""

from collections.abc import Sequence

import pytest

from anki_lang_cards import __main__ as main_module
from anki_lang_cards import _types as types
from anki_lang_cards import conflict_resolver

MAX_SENTENCES = 3


def make_word(
    word: str,
    meaning: str = "meaning",
    rank: int | None = None,
    sentence_count: int = 0,
    part_of_speech: str = "nm",
) -> types.Word:
    return types.Word(
        rank=rank,
        word=word,
        meaning=meaning,
        part_of_speech=part_of_speech,
        article=None,
        sentences=tuple(
            types.Sentence(spanish=f"{word} frase {i}.", english=f"EN{i}", source=None)
            for i in range(sentence_count)
        ),
    )


def classify(
    selected: Sequence[types.Word],
    *,
    inputs: Sequence[types.Word] = (),
    deck: Sequence[types.Word] = (),
    output: Sequence[types.Word] = (),
    decided_ranks: frozenset[int] = frozenset(),
) -> main_module.Classification:
    return main_module._classify_for_generation(
        selected,
        inputs_wl=conflict_resolver.WordLookup(inputs),
        deck_wl=conflict_resolver.WordLookup(deck),
        output_wl=conflict_resolver.WordLookup(output),
        decided_ranks=decided_ranks,
        max_sentences=MAX_SENTENCES,
    )


def test_a_word_already_emitted_is_skipped() -> None:
    """Old group 1: in the output CSV with nothing new to add."""
    emitted = make_word("perro", rank=10, sentence_count=MAX_SENTENCES)
    result = classify([emitted], output=[emitted])

    assert result.skipped == (emitted,)
    assert result.to_llm == ()


def test_new_sentences_reopen_an_emitted_word() -> None:
    """Old group 2: the union keeps the new sentences first, output extras after."""
    old = make_word("perro", rank=10, sentence_count=1)
    fresh_sentence = types.Sentence(
        spanish="Frase nueva del perro.", english="New.", source=None
    )
    incoming = types.Word(
        rank=10,
        word="perro",
        meaning="meaning",
        part_of_speech="nm",
        article=None,
        sentences=(fresh_sentence,),
    )

    result = classify([incoming], output=[old])

    (word,) = result.to_llm
    assert word.sentences[0] == fresh_sentence
    assert len(word.sentences) == 2


def test_a_full_emitted_word_with_new_sentences_goes_straight_out() -> None:
    old = make_word("perro", rank=10, sentence_count=1)
    incoming = make_word("perro", rank=10, sentence_count=MAX_SENTENCES)

    result = classify([incoming], output=[old])

    assert len(result.include_directly) == 1
    assert result.to_llm == ()


def test_a_deck_word_under_the_target_is_topped_up() -> None:
    """Old group 4: nothing new, but short of sentences — LLM, not output."""
    deck_word = make_word("perro", rank=10, sentence_count=1)
    result = classify([deck_word], deck=[deck_word])

    assert result.to_llm == (deck_word,)
    assert result.include_directly == ()


def test_a_full_deck_word_with_nothing_new_is_skipped() -> None:
    deck_word = make_word("perro", rank=10, sentence_count=MAX_SENTENCES)
    result = classify([deck_word], deck=[deck_word])

    assert result.skipped == (deck_word,)


def test_new_deck_sentences_mean_an_update() -> None:
    """Old group 3: the merged word carries sentences the deck lacks."""
    deck_word = make_word("perro", rank=10, sentence_count=2)
    merged = make_word("perro", rank=10, sentence_count=MAX_SENTENCES)

    result = classify([merged], deck=[deck_word])

    assert result.include_directly == (merged,)


def test_a_decided_merge_is_an_update() -> None:
    """Old group 5: an answered decision always reaches the output."""
    merged = make_word("perro", rank=10, sentence_count=MAX_SENTENCES)
    deck_word = make_word("perro", rank=10, sentence_count=MAX_SENTENCES)

    result = classify([merged], deck=[deck_word], decided_ranks=frozenset({10}))

    assert result.include_directly == (merged,)


def test_a_brand_new_word_goes_to_the_llm() -> None:
    """Old group 7."""
    new = make_word("gato", rank=None)
    result = classify([new])

    assert result.to_llm == (new,)


def test_input_sentences_force_inclusion() -> None:
    """Old has_sentences: annotation sentences always reach the output."""
    input_word = make_word("perro", rank=10, sentence_count=1)
    deck_word = make_word("perro", rank=10, sentence_count=MAX_SENTENCES)
    merged = make_word("perro", rank=10, sentence_count=MAX_SENTENCES)

    result = classify([merged], inputs=[input_word], deck=[deck_word])

    assert result.include_directly == (merged,)


@pytest.mark.parametrize("sentence_count", [0, MAX_SENTENCES - 1])
def test_the_gate_is_strictly_less_than(sentence_count: int) -> None:
    word = make_word("gato", sentence_count=sentence_count)
    assert classify([word]).to_llm == (word,)


def test_at_the_gate_means_no_generation() -> None:
    word = make_word("gato", sentence_count=MAX_SENTENCES)
    assert classify([word]).to_llm == ()


def test_an_untouched_rank_less_deck_note_is_skipped() -> None:
    """A deck note with an empty Rank field passes the rank-based selection
    (rank-less words are assumed to be inputs or split cards), so it has to
    be caught here rather than sent off for sentences."""
    deck_note = make_word("gato", rank=None, sentence_count=1)
    result = classify([deck_note], deck=[deck_note])

    assert result.skipped == (deck_note,)
    assert result.to_llm == ()
    assert result.include_directly == ()


def test_a_rank_less_input_matching_a_deck_note_is_not_mistaken_for_it() -> None:
    """Only the deck's own object is left alone: an input that merely looks
    like a deck note (a declined duplicate, say) is still generated for."""
    deck_note = make_word("gato", rank=None, sentence_count=1)
    incoming = make_word("gato", rank=None, sentence_count=1)
    result = classify([incoming], deck=[deck_note], inputs=[incoming])

    assert result.to_llm == (incoming,)
    assert result.skipped == ()
