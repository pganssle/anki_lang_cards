"""Tests for detecting words that only occur inside a fixed phrase."""

from collections.abc import Sequence

import hypothesis
import pytest
from hypothesis import strategies as st

from anki_lang_cards import _types as types
from anki_lang_cards import phrase_context


def make_word(
    word: str,
    sentences: Sequence[str],
    part_of_speech: str = "nm",
    meaning: str = "meaning",
    rank: int | None = None,
) -> types.Word:
    return types.Word(
        rank=rank,
        word=word,
        meaning=meaning,
        part_of_speech=part_of_speech,
        article=None,
        sentences=tuple(
            types.Sentence(spanish=spanish, english="", source=None)
            for spanish in sentences
        ),
    )


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("Lo miró de reojo.", ("lo", "miró", "de", "reojo")),
        ("¿Qué pasa?", ("qué", "pasa")),
        ("—Vale —dijo, sonriendo.", ("vale", "dijo", "sonriendo")),
        # Punctuation is dropped rather than splitting the phrase around it.
        ("por arte, de birlibirloque", ("por", "arte", "de", "birlibirloque")),
        ("", ()),
    ],
)
def test_tokenize(sentence: str, expected: tuple[str, ...]) -> None:
    assert phrase_context.tokenize(sentence) == expected


def test_tokenize_is_case_insensitive() -> None:
    assert phrase_context.tokenize("De Reojo") == phrase_context.tokenize("de reojo")


class TestRecurringContext:
    def test_a_fixed_phrase_is_found_whole(self) -> None:
        sentences = [
            "Lo consiguió por arte de birlibirloque, sin esfuerzo.",
            "Todo apareció por arte de birlibirloque.",
            "Y por arte de birlibirloque se resolvió el problema.",
        ]
        (run,) = phrase_context.recurring_context("birlibirloque", sentences)

        assert run.phrase == "por arte de birlibirloque"
        assert run.coverage == 1.0
        assert run.sentence_count == 3

    def test_a_two_word_phrase_is_found(self) -> None:
        sentences = [
            "Me miró de reojo.",
            "Lo observaba de reojo mientras hablaba.",
            "De reojo vi que se marchaba.",
        ]
        (run,) = phrase_context.recurring_context("reojo", sentences)

        assert run.phrase == "de reojo"
        assert run.coverage == 1.0

    def test_a_word_that_stands_alone_has_no_recurring_context(self) -> None:
        """ "genio" belongs in the deck on its own, phrase or no phrase."""
        sentences = [
            "Tiene un genio terrible por las mañanas.",
            "Era un genio de las matemáticas.",
            "El genio salió de la lámpara.",
            "Genio y figura hasta la sepultura.",
        ]
        assert phrase_context.recurring_context("genio", sentences) == ()

    def test_fragments_of_a_longer_run_are_dropped(self) -> None:
        """ "de birlibirloque" says less than "por arte de birlibirloque"."""
        sentences = [
            "Apareció por arte de birlibirloque.",
            "Desapareció por arte de birlibirloque.",
        ]
        runs = phrase_context.recurring_context("birlibirloque", sentences)

        assert [run.phrase for run in runs] == ["por arte de birlibirloque"]

    def test_a_shorter_run_survives_when_it_covers_more(self) -> None:
        """The longer phrase is not always the one that explains the evidence."""
        sentences = [
            "Le dio con la culata de la escopeta.",
            "Le dio con la culata de la pistola.",
            "Agarró la culata de la escopeta.",
        ]
        runs = phrase_context.recurring_context("culata", sentences, min_coverage=0.6)
        phrases = [run.phrase for run in runs]

        assert "la culata de la escopeta" in phrases
        assert any(run.coverage == 1.0 for run in runs)

    def test_context_spanning_both_sides_is_found(self) -> None:
        """The run is not anchored to either side of the headword."""
        sentences = [
            "Vamos a por ello ahora.",
            "Fue a por ello sin dudarlo.",
            "Iré a por ello mañana.",
        ]
        (run,) = phrase_context.recurring_context("por", sentences)

        assert run.phrase == "a por ello"

    def test_sentences_without_the_headword_are_ignored(self) -> None:
        sentences = [
            "Me miró de reojo.",
            "Lo observaba de reojo.",
            "Una frase que no contiene la palabra.",
        ]
        (run,) = phrase_context.recurring_context("reojo", sentences)

        assert run.coverage == 1.0
        assert run.sentence_count == 2

    def test_a_plural_form_still_counts(self) -> None:
        sentences = [
            "Le di balas de fogueo.",
            "Cargó una bala de fogueo.",
            "Solo tenía balas de fogueos falsos.",
        ]
        runs = phrase_context.recurring_context("fogueo", sentences)

        assert runs
        assert runs[0].phrase.endswith("de fogueo")

    def test_no_sentences_yields_nothing(self) -> None:
        assert phrase_context.recurring_context("reojo", []) == ()

    def test_a_multi_word_headword_is_skipped(self) -> None:
        """The question only makes sense for a single word."""
        assert phrase_context.recurring_context("de reojo", ["Lo miró de reojo."]) == ()

    def test_runs_are_ordered_longest_first(self) -> None:
        sentences = [
            "Apareció por arte de birlibirloque.",
            "Se fue por arte de birlibirloque.",
            "Otra cosa de birlibirloque distinta.",
        ]
        runs = phrase_context.recurring_context(
            "birlibirloque", sentences, min_coverage=0.6
        )
        lengths = [run.length for run in runs]
        assert lengths == sorted(lengths, reverse=True)


class TestFindPhraseBoundWords:
    def test_a_phrase_bound_word_is_reported(self) -> None:
        word = make_word(
            "reojo",
            [
                "Me miró de reojo.",
                "Lo observaba de reojo mientras hablaba.",
                "De reojo vi que se marchaba.",
            ],
        )
        (found,) = phrase_context.find_phrase_bound_words([word])

        assert found.word is word
        assert found.context.phrase == "de reojo"
        assert not found.is_redundant

    def test_an_independent_word_is_not_reported(self) -> None:
        word = make_word(
            "genio",
            [
                "Tiene un genio terrible.",
                "Era un genio de las matemáticas.",
                "El genio salió de la lámpara.",
            ],
        )
        assert phrase_context.find_phrase_bound_words([word]) == ()

    def test_an_existing_phrase_note_is_linked(self) -> None:
        fragment = make_word(
            "reojo",
            [
                "Me miró de reojo.",
                "Lo observaba de reojo.",
                "De reojo vi que se marchaba.",
            ],
        )
        phrase = make_word("de reojo", [], part_of_speech="adv")

        (found,) = phrase_context.find_phrase_bound_words([fragment, phrase])

        assert found.existing_phrase is phrase
        assert found.is_redundant

    def test_a_word_appearing_in_a_phrase_note_is_still_independent(self) -> None:
        """Having a phrase card is not itself evidence against the word card."""
        word = make_word(
            "sepultura",
            [
                "Lo llevaron a la sepultura.",
                "Cavaron una sepultura poco profunda.",
                "Genio y figura hasta la sepultura.",
            ],
        )
        phrase = make_word("genio y figura hasta la sepultura", [], part_of_speech="nm")

        assert phrase_context.find_phrase_bound_words([word, phrase]) == ()

    def test_too_few_examples_says_nothing(self) -> None:
        word = make_word("reojo", ["Me miró de reojo.", "Lo miró de reojo."])
        assert phrase_context.find_phrase_bound_words([word]) == ()

    def test_multi_word_notes_are_skipped(self) -> None:
        word = make_word(
            "de reojo",
            ["Me miró de reojo.", "Lo miró de reojo.", "De reojo lo vio."],
        )
        assert phrase_context.find_phrase_bound_words([word]) == ()

    def test_a_word_with_no_sentences_is_skipped(self) -> None:
        assert phrase_context.find_phrase_bound_words([make_word("reojo", [])]) == ()

    def test_results_are_ordered_by_strength_of_evidence(self) -> None:
        strong = make_word(
            "reojo",
            ["Miró de reojo.", "Lo vio de reojo.", "De reojo lo notó."],
        )
        weaker = make_word(
            "fogueo",
            [
                "Le di balas de fogueo.",
                "Cargó balas de fogueo.",
                "Compró munición corriente.",
                "Tenía balas de fogueo.",
            ],
        )
        found = phrase_context.find_phrase_bound_words(
            [weaker, strong], min_coverage=0.5
        )
        coverages = [f.context.coverage for f in found]
        assert coverages == sorted(coverages, reverse=True)


@hypothesis.given(st.text(max_size=60))
def test_tokenize_never_produces_empty_tokens(sentence: str) -> None:
    assert all(phrase_context.tokenize(sentence))


@hypothesis.given(
    st.text(min_size=1, max_size=15).filter(lambda s: s.strip()),
    st.lists(st.text(max_size=60), max_size=5),
)
def test_recurring_context_never_exceeds_full_coverage(
    word: str, sentences: list[str]
) -> None:
    for run in phrase_context.recurring_context(word, sentences):
        assert 0.0 < run.coverage <= 1.0
        assert run.length >= 2


class TestDeterminerNoise:
    def test_a_bare_determiner_is_not_a_phrase(self) -> None:
        """Every Spanish noun takes a determiner, so "la X" says nothing."""
        sentences = [
            "Encendió la lumbre con cuidado.",
            "Se sentó junto a la lumbre.",
            "La lumbre iluminaba la sala.",
        ]
        assert phrase_context.recurring_context("lumbre", sentences) == ()

    def test_a_preposition_before_the_word_is_a_phrase(self) -> None:
        """Unlike a determiner, a preposition is not obligatory."""
        sentences = [
            "Lo miró de reojo.",
            "Me observaba de reojo.",
            "De reojo lo vio marcharse.",
        ]
        (run,) = phrase_context.recurring_context("reojo", sentences)
        assert run.phrase == "de reojo"

    def test_a_determiner_still_counts_with_context_on_both_sides(self) -> None:
        sentences = [
            "Vamos a dar un garbeo por la Gran Vía.",
            "Dieron un garbeo por el parque.",
            "Nos damos un garbeo por ahí.",
        ]
        runs = phrase_context.recurring_context("garbeo", sentences)
        assert any(run.phrase.startswith("un garbeo por") for run in runs)

    def test_taking_a_complement_alone_is_not_a_phrase(self) -> None:
        """ "manojo de" recurs perfectly, but that is just what "manojo" does."""
        sentences = [
            "Cogió manojo de hierbas.",
            "Llevaba manojo de llaves.",
            "Compró manojo de flores.",
        ]
        assert phrase_context.recurring_context("manojo", sentences) == ()

    def test_a_quantifier_noun_is_a_known_false_positive(self) -> None:
        """A limitation worth stating rather than papering over.

        "un manojo de" and "un garbeo por" have the same shape — determiner,
        headword, preposition — but only the second is an idiom. Nothing in the
        surface forms separates them, so both are reported and the reader
        decides. The cost is a short list with some quantifier nouns in it,
        which is the right way round.
        """
        sentences = [
            "Cogió un manojo de hierbas.",
            "Llevaba un manojo de llaves.",
            "Compró un manojo de flores.",
        ]
        runs = phrase_context.recurring_context("manojo", sentences)
        assert [run.phrase for run in runs] == ["un manojo de"]

    def test_a_contracted_preposition_still_counts_as_context(self) -> None:
        sentences = [
            "Salió del tirón sin descansar.",
            "Lo hizo del tirón.",
            "Terminó el libro del tirón.",
        ]
        (run,) = phrase_context.recurring_context("tirón", sentences)
        assert run.phrase == "del tirón"
