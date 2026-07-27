"""Tests for carrying out triage decisions."""

import pathlib
from collections.abc import Sequence
from typing import Final

import attrs
import pytest

from anki_lang_cards import _types as types
from anki_lang_cards import apply_merges, definition_similarity, triage

CORPUS: Final[Sequence[str]] = (
    "social climber",
    "busybody",
    "meddlesome",
    "settlement, colony",
    "priest",
    "cure, remedy",
    "duty, obligation",
    "should, ought to",
    "dog",
    "cat",
)


@pytest.fixture
def comparer() -> definition_similarity.DefinitionComparer:
    return definition_similarity.DefinitionComparer(CORPUS)


@pytest.fixture
def decisions(tmp_path: pathlib.Path) -> triage.DecisionCache:
    return triage.DecisionCache(tmp_path / "decisions.json")


def make_word(
    word: str,
    part_of_speech: str,
    meaning: str,
    rank: int,
    sentences: Sequence[types.Sentence] = (),
) -> types.Word:
    return types.Word(
        rank=rank,
        word=word,
        meaning=meaning,
        part_of_speech=part_of_speech,
        article=None,
        sentences=tuple(sentences),
    )


def decide(
    cache: triage.DecisionCache,
    words: Sequence[types.Word],
    disposition: triage.Disposition,
    revisions: Sequence[types.Word] | None = None,
) -> None:
    cache.remember(
        triage.Candidate(
            words=tuple(words),
            reason=triage.Reason.COMPATIBLE,
            comparison=definition_similarity.Comparison(1.0, frozenset()),
            disposition=disposition,
            revisions=None if revisions is None else tuple(revisions),
        )
    )


def test_nothing_decided_changes_nothing(comparer, decisions) -> None:
    words = [make_word("perro", "nm", "dog", 1), make_word("gato", "nm", "cat", 2)]
    outcome = apply_merges.apply_decisions(words, decisions, comparer)

    assert outcome.merges == ()
    assert {w.rank for w in outcome.words} == {1, 2}


def test_a_decided_group_is_combined(comparer, decisions) -> None:
    group = [
        make_word("trepa", "nm/f", "social climber", 8576),
        make_word("trepa", "nm/f", "climber", 9615),
    ]
    decide(decisions, group, triage.Disposition.MERGE)

    outcome = apply_merges.apply_decisions(group, decisions, comparer)

    assert len(outcome.merges) == 1
    assert len(outcome.words) == 1
    assert outcome.words[0].rank == 8576, "the surviving note keeps the lower rank"


def test_a_group_left_for_review_is_not_combined(comparer, decisions) -> None:
    group = [
        make_word("cura", "nm", "priest", 1790),
        make_word("cura", "nm", "cure, remedy", 9000),
    ]
    decide(decisions, group, triage.Disposition.REVIEW)

    outcome = apply_merges.apply_decisions(group, decisions, comparer)
    assert outcome.merges == ()
    assert len(outcome.words) == 2


def test_identical_notes_combine_without_a_decision(comparer, decisions) -> None:
    """Nothing is left to decide, so nothing is asked."""
    group = [
        make_word("grelos", "nm pl", "turnip greens", 8225),
        make_word("grelos", "nm pl", "turnip greens", 8554),
    ]
    outcome = apply_merges.apply_decisions(group, decisions, comparer)

    assert len(outcome.merges) == 1
    assert outcome.merges[0].decided is False


class TestRevisions:
    def test_a_rewritten_note_is_replaced(self, comparer, decisions) -> None:
        original = make_word("segundo", "adj", "second", 223)
        revised = make_word("segundo/a", "adj", "second", 223)
        decide(
            decisions,
            [original, make_word("segundo", "nm", "second", 1693)],
            triage.Disposition.KEEP_SEPARATE,
            revisions=[revised, make_word("segundo", "nm", "second", 1693)],
        )

        outcome = apply_merges.apply_decisions([original], decisions, comparer)
        assert [w.word for w in outcome.words] == ["segundo/a"]
        assert len(outcome.revised) == 1

    def test_a_rewrite_keeps_the_sentences(self, comparer, decisions) -> None:
        """The editor never showed the examples, so it cannot have removed them."""
        sentence = types.Sentence(
            spanish="Un segundo.", english="A second.", source=None
        )
        original = make_word("segundo", "adj", "second", 223, [sentence])
        decide(
            decisions,
            [original],
            triage.Disposition.KEEP_SEPARATE,
            revisions=[make_word("segundo/a", "adj", "second", 223)],
        )

        outcome = apply_merges.apply_decisions([original], decisions, comparer)
        assert outcome.words[0].sentences == (sentence,)

    def test_rewriting_into_identical_notes_then_merges_them(
        self, comparer, decisions
    ) -> None:
        """The case that makes this iterative rather than a single sweep."""
        noun = make_word("metomentodo", "nm/f", "busybody", 5433)
        adjective = make_word("metomentodo", "adj", "meddlesome", 9609)
        decide(
            decisions,
            [noun, adjective],
            triage.Disposition.KEEP_SEPARATE,
            revisions=[
                make_word("metomentodo", "nm/f", "busybody, meddlesome", 5433),
                make_word("metomentodo", "nm/f", "busybody, meddlesome", 9609),
            ],
        )

        outcome = apply_merges.apply_decisions([noun, adjective], decisions, comparer)

        assert len(outcome.words) == 1
        assert outcome.words[0].rank == 5433
        assert outcome.merges[0].decided is False
        assert outcome.rounds >= 2, "the merge could not have been seen in one pass"


class TestDecisionMatching:
    def test_a_decision_survives_its_notes_being_rewritten(
        self, comparer, decisions
    ) -> None:
        """The decision is about which notes, not about what they said."""
        first = make_word("nodriza", "nf", "wet nurse", 8825)
        second = make_word("nodriza", "nf", "mother ship", 9570)
        decide(decisions, [first, second], triage.Disposition.MERGE)
        # Now rewrite one of them, as the editor would.
        decide(
            decisions,
            [first],
            triage.Disposition.MERGE,
            revisions=[make_word("nodriza", "nf", "wet nurse; mother ship", 8825)],
        )

        outcome = apply_merges.apply_decisions([first, second], decisions, comparer)
        assert len(outcome.words) == 1, "the merge decision still applied"


class TestHeadwords:
    def test_one_spelling_is_chosen_and_reported(self, comparer, decisions) -> None:
        group = [
            make_word("dar paso", "v phr", "to give way to", 8012),
            make_word("dar paso (a)", "v phr", "to give way to", 9592),
        ]
        decide(decisions, group, triage.Disposition.MERGE)

        outcome = apply_merges.apply_decisions(group, decisions, comparer)
        (merge,) = outcome.merges

        assert merge.result.word == "dar paso", "the established note's spelling wins"
        assert merge.dropped_headwords == ("dar paso (a)",)

    def test_agreeing_spellings_report_nothing(self, comparer, decisions) -> None:
        group = [
            make_word("trepa", "nm/f", "social climber", 1),
            make_word("trepa", "nm/f", "climber", 2),
        ]
        decide(decisions, group, triage.Disposition.MERGE)

        (merge,) = apply_merges.apply_decisions(group, decisions, comparer).merges
        assert merge.dropped_headwords == ()


class TestReporting:
    def test_superseded_deck_notes_are_listed(self, comparer, decisions) -> None:
        """Importing updates the surviving rank; the other note stays behind."""
        deck = [
            make_word("botarate", "nm/f", "spendthrift", 7383),
            make_word("botarate", "nm/f", "fool", 7608),
        ]
        decide(decisions, deck, triage.Disposition.MERGE)

        outcome = apply_merges.apply_decisions(deck, decisions, comparer)
        (orphan,) = apply_merges.superseded_notes(outcome, deck)
        assert orphan.rank == 7608

    def test_a_new_note_being_superseded_is_not_listed(
        self, comparer, decisions
    ) -> None:
        """Anki has never seen it, so there is nothing to delete."""
        deck = [make_word("asentamiento", "nm", "settlement", 4594)]
        added = [make_word("asentamiento", "nm", "settlement, colony", 8821)]
        decide(decisions, [*deck, *added], triage.Disposition.MERGE)

        outcome = apply_merges.apply_decisions([*deck, *added], decisions, comparer)
        assert apply_merges.superseded_notes(outcome, deck) == ()

    def test_only_collisions_touching_the_import_are_reported(
        self, comparer, decisions
    ) -> None:
        """Groups that have disagreed in the deck for a year are not this run's doing."""
        deck = [
            make_word("cura", "nm", "priest", 1790),
            make_word("cura", "nm", "cure, remedy", 1791),
        ]
        added = [make_word("perro", "nm", "dog", 9000)]

        outcome = apply_merges.apply_decisions([*deck, *added], decisions, comparer)
        assert outcome.unresolved, "the deck's own collision is still found"
        assert apply_merges.new_collisions(outcome, added) == ()

    def test_the_surviving_rank_is_written_even_when_it_came_from_the_deck(
        self, comparer, decisions
    ) -> None:
        deck = [make_word("asentamiento", "nm", "settlement", 4594)]
        added = [make_word("asentamiento", "nm", "settlement, colony", 8821)]
        decide(decisions, [*deck, *added], triage.Disposition.MERGE)

        outcome = apply_merges.apply_decisions([*deck, *added], decisions, comparer)
        ranks = {w.rank for w in apply_merges.words_to_import(outcome, added)}
        assert ranks == {4594}

    def test_untouched_deck_notes_are_not_written(self, comparer, decisions) -> None:
        deck = [make_word("gato", "nm", "cat", 5)]
        added = [make_word("perro", "nm", "dog", 9000)]

        outcome = apply_merges.apply_decisions([*deck, *added], decisions, comparer)
        ranks = {w.rank for w in apply_merges.words_to_import(outcome, added)}
        assert ranks == {9000}


def test_a_merge_chain_runs_to_completion(comparer, decisions) -> None:
    """Three notes converging on one, which needs more than a single pass."""
    a = make_word("trepa", "nm/f", "social climber", 1)
    b = make_word("trepa", "nm/f", "climber", 2)
    c = make_word("trepa", "nm/f", "social climber", 3)
    decide(decisions, [a, b, c], triage.Disposition.MERGE)

    outcome = apply_merges.apply_decisions([a, b, c], decisions, comparer)
    assert len(outcome.words) == 1
    assert outcome.words[0].rank == 1


def test_the_round_cap_is_respected(comparer, decisions) -> None:
    group = [
        make_word("trepa", "nm/f", "social climber", 1),
        make_word("trepa", "nm/f", "climber", 2),
    ]
    decide(decisions, group, triage.Disposition.MERGE)

    outcome = apply_merges.apply_decisions(group, decisions, comparer, max_rounds=1)
    assert outcome.rounds == 1


def test_summarize_mentions_what_happened(comparer, decisions) -> None:
    group = [
        make_word("trepa", "nm/f", "social climber", 1),
        make_word("trepa", "nm/f", "climber", 2),
    ]
    decide(decisions, group, triage.Disposition.MERGE)

    outcome = apply_merges.apply_decisions(group, decisions, comparer)
    text = " ".join(apply_merges.summarize(outcome))
    assert "1 groups merged" in text


def test_sources_are_not_mutated(comparer, decisions) -> None:
    """Merging builds new notes; the originals must survive for reporting."""
    first = make_word("trepa", "nm/f", "social climber", 1)
    second = make_word("trepa", "nm/f", "climber", 2)
    decide(decisions, [first, second], triage.Disposition.MERGE)

    before = (attrs.evolve(first), attrs.evolve(second))
    apply_merges.apply_decisions([first, second], decisions, comparer)

    assert (first.word, first.meaning) == (before[0].word, before[0].meaning)
    assert (second.word, second.meaning) == (before[1].word, before[1].meaning)


class TestRegressions:
    def test_two_notes_sharing_a_rank_are_rewritten_separately(
        self, comparer, decisions
    ) -> None:
        """The deck's copy and the imported copy of one note both have its rank.

        Keying rewrites on the rank alone let one displace the other, so a typo
        the rewrite had fixed came back in the merged output.
        """
        from_deck = make_word("prenda", "nf", "piece of clothing", 3467)
        from_import = make_word("prenda", "nf", "piece of glothing, garment", 3467)
        decide(
            decisions,
            [from_deck, from_import],
            triage.Disposition.MERGE,
            revisions=[
                make_word("prenda", "nf", "piece of clothing", 3467),
                make_word("prenda", "nf", "piece of clothing, garment", 3467),
            ],
        )

        outcome = apply_merges.apply_decisions(
            [from_deck, from_import], decisions, comparer
        )
        assert "glothing" not in outcome.words[0].meaning

    def test_an_unchanged_rewrite_counts_as_no_rewrite(
        self, comparer, decisions
    ) -> None:
        """Opening the editor and saving is not an edit."""
        word = make_word("trepa", "nm/f", "social climber", 1)
        decide(
            decisions,
            [word],
            triage.Disposition.MERGE,
            revisions=[make_word("trepa", "nm/f", "social climber", 1)],
        )

        assert apply_merges.apply_decisions([word], decisions, comparer).revised == ()

    def test_a_verb_merged_with_a_preposition_is_not_given_an_infinitive(
        self, comparer, decisions
    ) -> None:
        """ "de no ser por" was labelled both, and produced "to if it weren't for"."""
        group = [
            make_word("de no ser por", "prep", "if it weren't for", 8692),
            make_word("de no ser por", "v phr", "were it not for", 9505),
        ]
        decide(decisions, group, triage.Disposition.MERGE)

        (merge,) = apply_merges.apply_decisions(group, decisions, comparer).merges
        assert not merge.result.meaning.startswith("to ")
        assert "if it weren't for" in merge.result.meaning

    def test_a_merged_noun_keeps_its_gender(self, comparer, decisions) -> None:
        """ "nm/f" and "nf" used to collapse to a bare "n", losing the gender
        the card front shows."""
        group = [
            make_word("trepa", "nm/f", "social climber", 8576),
            make_word("trepa", "nf", "climber", 9615),
        ]
        decide(decisions, group, triage.Disposition.MERGE)

        (merge,) = apply_merges.apply_decisions(group, decisions, comparer).merges
        assert str(merge.result.part_of_speech) == "nm/f"

    def test_a_real_verb_keeps_its_infinitive(self, comparer, decisions) -> None:
        group = [
            make_word("percatarse", "vtr phr", "to notice, realize", 5676),
            make_word("percatarse", "vi", "to notice", 9595),
        ]
        decide(decisions, group, triage.Disposition.MERGE)

        (merge,) = apply_merges.apply_decisions(group, decisions, comparer).merges
        assert merge.result.meaning.startswith("to ")
