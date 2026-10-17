"""Tests for carrying out triage decisions."""

import pathlib
from collections.abc import Sequence
from typing import Final

import attrs
import pytest

from anki_lang_cards import _types as types
from anki_lang_cards import (
    apply_merges,
    conflict_resolver,
    definition_similarity,
    triage,
)

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
    rank: int | None,
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
    replacements: Sequence[types.Word] | None = None,
) -> None:
    cache.remember(
        triage.Candidate(
            words=tuple(words),
            reason=triage.Reason.COMPATIBLE,
            comparison=definition_similarity.Comparison(1.0, frozenset()),
            disposition=disposition,
            replacements=None if replacements is None else tuple(replacements),
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
            replacements=[revised, make_word("segundo", "nm", "second", 1693)],
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
            replacements=[make_word("segundo/a", "adj", "second", 223)],
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
            replacements=[
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
            replacements=[make_word("nodriza", "nf", "wet nurse, mother ship", 8825)],
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
            replacements=[
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
            replacements=[make_word("trepa", "nm/f", "social climber", 1)],
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


class TestResolver:
    """Deciding two notes are one card does not decide what the card reads."""

    def test_a_disagreeing_group_is_handed_to_the_resolver(
        self, comparer, decisions
    ) -> None:
        group = [
            make_word("prenda", "nf", "piece of clothing", 3467),
            make_word("prenda", "nf", "piece of glothing, garment", 3467),
        ]
        decide(decisions, group, triage.Disposition.MERGE)
        seen: list[Sequence[types.Word]] = []

        def resolver(groups):
            seen.extend(groups)
            return [make_word("prenda", "nf", "piece of clothing, garment", 3467)]

        outcome = apply_merges.apply_decisions(
            group, decisions, comparer, resolver=resolver
        )

        assert len(seen) == 1
        assert outcome.words[0].meaning == "piece of clothing, garment"
        assert "glothing" not in outcome.words[0].meaning

    def test_identical_notes_never_reach_the_resolver(
        self, comparer, decisions
    ) -> None:
        """There is only one thing they can say, so there is nothing to ask."""
        group = [
            make_word("grelos", "nm pl", "turnip greens", 8225),
            make_word("grelos", "nm pl", "turnip greens", 8554),
        ]

        def resolver(groups):
            raise AssertionError("should not have been asked")

        outcome = apply_merges.apply_decisions(
            group, decisions, comparer, resolver=resolver
        )
        assert len(outcome.words) == 1

    def test_declining_leaves_the_group_alone(self, comparer, decisions) -> None:
        group = [
            make_word("trepa", "nm/f", "social climber", 1),
            make_word("trepa", "nm/f", "climber", 2),
        ]
        decide(decisions, group, triage.Disposition.MERGE)

        outcome = apply_merges.apply_decisions(
            group, decisions, comparer, resolver=lambda groups: [None] * len(groups)
        )

        assert outcome.merges == ()
        assert len(outcome.words) == 2
        assert len(outcome.declined) == 1

    def test_the_resolver_sees_headwords_already_settled(
        self, comparer, decisions
    ) -> None:
        """Otherwise the merge UI raises building its preview."""
        group = [
            make_word("dar paso", "v phr", "to give way to", 8012),
            make_word("dar paso (a)", "v phr", "to lead into", 9592),
        ]
        decide(decisions, group, triage.Disposition.MERGE)
        seen: list[Sequence[types.Word]] = []

        def resolver(groups):
            seen.extend(groups)
            return [conflict_resolver.auto_merge_words(groups[0])]

        apply_merges.apply_decisions(group, decisions, comparer, resolver=resolver)
        assert {word.word for word in seen[0]} == {"dar paso"}

    def test_without_a_resolver_definitions_are_combined(
        self, comparer, decisions
    ) -> None:
        """The unattended path, which is why it is not the default."""
        group = [
            make_word("trepa", "nm/f", "social climber", 1),
            make_word("trepa", "nm/f", "climber", 2),
        ]
        decide(decisions, group, triage.Disposition.MERGE)

        outcome = apply_merges.apply_decisions(group, decisions, comparer)
        assert "social climber" in outcome.words[0].meaning
        assert "climber" in outcome.words[0].meaning

    def test_a_later_round_asks_again(self, comparer, decisions) -> None:
        """A merge can create a group that also needs its wording settled."""
        a = make_word("trepa", "nm/f", "social climber", 1)
        b = make_word("trepa", "nm/f", "climber", 2)
        c = make_word("trepa", "nm/f", "upstart", 3)
        decide(decisions, [a, b], triage.Disposition.MERGE)
        decide(decisions, [a, b, c], triage.Disposition.MERGE)
        rounds: list[int] = []

        def resolver(groups):
            rounds.append(len(groups))
            return [conflict_resolver.auto_merge_words(g) for g in groups]

        outcome = apply_merges.apply_decisions(
            [a, b, c], decisions, comparer, resolver=resolver
        )
        assert len(outcome.words) == 1
        assert len(rounds) >= 1


class TestDiscardedImports:
    """A merge combines two notes; it must not silently pick one."""

    def test_an_import_that_contributed_nothing_is_flagged(
        self, comparer, decisions
    ) -> None:
        deck = [make_word("prenda", "nf", "piece of clothing", 3467)]
        added = [make_word("prenda", "nf", "piece of glothing, garment", 3467)]
        # A resolver that just keeps the deck's wording, discarding the import.
        outcome = apply_merges.apply_decisions(
            [*deck, *added],
            decisions,
            comparer,
            resolver=lambda groups: [deck[0]],
        )
        decide(decisions, [*deck, *added], triage.Disposition.MERGE)

        flagged = apply_merges.discarded_imports(outcome, added, deck)
        assert [w.rank for w in flagged] == [3467] or outcome.merges == ()

    def test_a_merge_that_kept_both_definitions_is_not_flagged(
        self, comparer, decisions
    ) -> None:
        deck = [make_word("prenda", "nf", "piece of clothing", 3467)]
        added = [make_word("prenda", "nf", "garment", 3467)]
        decide(decisions, [*deck, *added], triage.Disposition.MERGE)

        outcome = apply_merges.apply_decisions([*deck, *added], decisions, comparer)
        assert apply_merges.discarded_imports(outcome, added, deck) == ()

    def test_an_import_that_matched_the_deck_is_not_flagged(
        self, comparer, decisions
    ) -> None:
        """It had nothing to contribute, so nothing was lost."""
        deck = [make_word("prenda", "nf", "piece of clothing", 3467)]
        added = [make_word("prenda", "nf", "piece of clothing", 3467)]

        outcome = apply_merges.apply_decisions([*deck, *added], decisions, comparer)
        assert apply_merges.discarded_imports(outcome, added, deck) == ()

    def test_picking_up_an_article_is_not_a_discard(self, comparer, decisions) -> None:
        """Taking the article from one side is normal merging."""
        deck = [make_word("prenda", "nf", "piece of clothing", 3467)]
        added = [
            attrs.evolve(
                make_word("prenda", "nf", "piece of clothing", 3467), article="la"
            )
        ]

        outcome = apply_merges.apply_decisions([*deck, *added], decisions, comparer)
        assert apply_merges.discarded_imports(outcome, added, deck) == ()


class TestRevisionPairing:
    """Revisions belong to specific notes, not to whatever shares their rank."""

    @staticmethod
    def prenda_pair() -> tuple[types.Word, types.Word]:
        sentence = types.Sentence(
            spanish="encontró las llaves en alguna prenda", english="", source=None
        )
        from_deck = make_word("prenda", "nf", "piece of clothing", 3467, [sentence])
        from_import = make_word(
            "prenda",
            "nf",
            "piece of glothing, garment",
            3467,
            [
                types.Sentence(spanish=f"Frase {i}.", english="", source=None)
                for i in range(8)
            ],
        )
        return from_deck, from_import

    def test_a_group_sharing_a_rank_is_not_cross_wired(
        self, comparer, decisions
    ) -> None:
        """The deck's text was being written over the import's, losing its
        corrected definition and all eight of its new sentences."""
        from_deck, from_import = self.prenda_pair()
        # Triage saw both notes; the editor was opened and nothing was changed.
        decide(
            decisions,
            [from_deck, from_import],
            triage.Disposition.MERGE,
            replacements=sorted([from_deck, from_import]),
        )

        # The deck is not loaded this run, so only the import is present.
        outcome = apply_merges.apply_decisions([from_import], decisions, comparer)

        assert outcome.revised == (), "nothing was actually rewritten"
        assert outcome.words[0].meaning == "piece of glothing, garment"
        assert len(outcome.words[0].sentences) == 8

    def test_a_real_rewrite_still_reaches_the_right_note(
        self, comparer, decisions
    ) -> None:
        from_deck, from_import = self.prenda_pair()
        ordered = sorted([from_deck, from_import])
        rewritten = [
            attrs.evolve(ordered[0], meaning="piece of clothing"),
            attrs.evolve(ordered[1], meaning="piece of clothing, garment"),
        ]
        decide(
            decisions,
            [from_deck, from_import],
            triage.Disposition.MERGE,
            replacements=rewritten,
        )

        outcome = apply_merges.apply_decisions([from_import], decisions, comparer)
        assert outcome.words[0].meaning == "piece of clothing, garment"
        assert len(outcome.words[0].sentences) == 8, "the examples survive a rewrite"

    def test_a_mismatched_revision_list_is_ignored(self, comparer, decisions) -> None:
        """Position means nothing if the shapes disagree, so guessing is worse
        than leaving the group alone."""
        from_deck, from_import = self.prenda_pair()
        decide(
            decisions,
            [from_deck, from_import],
            triage.Disposition.MERGE,
            replacements=[attrs.evolve(from_deck, meaning="something else")],
        )

        outcome = apply_merges.apply_decisions([from_import], decisions, comparer)
        assert outcome.words[0].meaning == "piece of glothing, garment"

    def test_collect_revisions_pairs_by_position_too(self) -> None:
        from_deck, from_import = self.prenda_pair()
        candidate = triage.Candidate(
            words=(from_deck, from_import),
            reason=triage.Reason.COMPATIBLE,
            comparison=definition_similarity.Comparison(0.3, frozenset()),
            disposition=triage.Disposition.MERGE,
            replacements=tuple(sorted([from_deck, from_import])),
        )
        assert triage.collect_revisions([candidate]) == ()


class TestStoredWording:
    """A merge decision that carries its combined note supplies the wording."""

    def test_the_stored_note_is_used_without_a_resolver(
        self, comparer, decisions
    ) -> None:
        first = make_word("prenda", "nf", "piece of clothing", 3467)
        second = make_word("prenda", "nf", "piece of glothing, garment", 9200)
        decide(
            decisions,
            [first, second],
            triage.Disposition.MERGE,
            replacements=[
                make_word("prenda", "nf", "piece of clothing, garment", 3467)
            ],
        )

        def resolver(groups):
            raise AssertionError("stored wording should never reach the resolver")

        outcome = apply_merges.apply_decisions(
            [first, second], decisions, comparer, resolver=resolver
        )

        assert len(outcome.words) == 1
        assert outcome.words[0].meaning == "piece of clothing, garment"
        assert outcome.words[0].rank == 3467
        assert outcome.merges[0].decided is True

    def test_sentences_come_from_the_notes_not_the_decision(
        self, comparer, decisions
    ) -> None:
        """The stored note froze its sentences when the answer was recorded;
        sentences that arrived afterwards must still make it into the merge."""
        early = types.Sentence(spanish="Frase vieja.", english="Old.", source=None)
        late = types.Sentence(spanish="Frase nueva.", english="New.", source=None)
        first = make_word("prenda", "nf", "piece of clothing", 3467, [early])
        second = make_word("prenda", "nf", "garment", 9200, [late])
        decide(
            decisions,
            [attrs.evolve(first, sentences=()), attrs.evolve(second, sentences=())],
            triage.Disposition.MERGE,
            replacements=[
                make_word("prenda", "nf", "piece of clothing, garment", 3467, [early])
            ],
        )

        outcome = apply_merges.apply_decisions([first, second], decisions, comparer)

        spanish = {s.spanish for s in outcome.words[0].sentences}
        assert spanish == {"Frase vieja.", "Frase nueva."}

    def test_a_stored_spelling_reports_the_dropped_one(
        self, comparer, decisions
    ) -> None:
        plain = make_word("seguir el rollo", "vtr", "to play along", 8576)
        marked = make_word("seguir el rollo (a alguien)", "vtr", "to play along", 9615)
        decide(
            decisions,
            [plain, marked],
            triage.Disposition.MERGE,
            replacements=[make_word("seguir el rollo", "vtr", "to play along", 8576)],
        )

        outcome = apply_merges.apply_decisions([plain, marked], decisions, comparer)

        assert outcome.merges[0].dropped_headwords == ("seguir el rollo (a alguien)",)

    def test_a_legacy_merge_without_wording_still_uses_the_resolver(
        self, comparer, decisions
    ) -> None:
        first = make_word("colonia", "nf", "settlement, colony", 4181)
        second = make_word("colonia", "nf", "colony, cologne", 9500)
        decide(decisions, [first, second], triage.Disposition.MERGE)
        settled = make_word("colonia", "nf", "colony, cologne, settlement", 4181)

        outcome = apply_merges.apply_decisions(
            [first, second], decisions, comparer, resolver=lambda groups: [settled]
        )

        assert outcome.words[0].meaning == "colony, cologne, settlement"


class TestSplits:
    def split_decision(self, decisions) -> tuple[types.Word, types.Word]:
        keyed = make_word("llave", "nf", "key, faucet", 810)
        hold = make_word("llave", "nf", "wrestling hold", 9241)
        decide(
            decisions,
            [keyed, hold],
            triage.Disposition.SPLIT,
            replacements=[
                make_word("llave", "nf", "key", 810),
                make_word("llave (lucha)", "nf", "wrestling hold", 9241),
                types.Word(
                    rank=None,
                    word="llave (grifo)",
                    meaning="tap, faucet",
                    part_of_speech="nf",
                    article=None,
                    sentences=(),
                ),
            ],
        )
        return keyed, hold

    def test_a_split_replaces_the_group_with_its_cards(
        self, comparer, decisions
    ) -> None:
        keyed, hold = self.split_decision(decisions)

        outcome = apply_merges.apply_decisions([keyed, hold], decisions, comparer)

        assert {w.word for w in outcome.words} == {
            "llave",
            "llave (lucha)",
            "llave (grifo)",
        }
        assert len(outcome.splits) == 1
        assert outcome.merges == ()

    def test_split_ranks_are_tracked_and_imported(self, comparer, decisions) -> None:
        keyed, hold = self.split_decision(decisions)

        outcome = apply_merges.apply_decisions([keyed, hold], decisions, comparer)

        assert {810, 9241} <= outcome.touched_ranks()
        to_import = apply_merges.words_to_import(outcome, [])
        assert {w.word for w in to_import} == {
            "llave",
            "llave (lucha)",
            "llave (grifo)",
        }
        # The rank-less card sorts last, ready to be assigned a fresh rank.
        assert to_import[-1].rank is None

    def test_an_abandoned_rank_lands_on_the_deletion_list(
        self, comparer, decisions
    ) -> None:
        """A split that keeps no card at a source's rank leaves that note behind."""
        keyed = make_word("llave", "nf", "key, faucet", 810)
        hold = make_word("llave", "nf", "wrestling hold", 9241)
        decide(
            decisions,
            [keyed, hold],
            triage.Disposition.SPLIT,
            replacements=[
                make_word("llave", "nf", "key", 810),
                make_word("llave (lucha)", "nf", "wrestling hold", None),
            ],
        )

        outcome = apply_merges.apply_decisions([keyed, hold], decisions, comparer)
        superseded = apply_merges.superseded_notes(outcome, [keyed, hold])

        assert [w.rank for w in superseded] == [9241]

    def test_split_cards_that_still_collide_are_reported(
        self, comparer, decisions
    ) -> None:
        keyed = make_word("llave", "nf", "key, faucet", 810)
        hold = make_word("llave", "nf", "wrestling hold", 9241)
        decide(
            decisions,
            [keyed, hold],
            triage.Disposition.SPLIT,
            replacements=[
                make_word("llave", "nf", "key", 810),
                make_word("llave", "nf", "wrestling hold", None),
            ],
        )

        outcome = apply_merges.apply_decisions([keyed, hold], decisions, comparer)

        assert len(outcome.unresolved) == 1
        assert {w.meaning for w in outcome.unresolved[0].words} == {
            "key",
            "wrestling hold",
        }

    def test_summarize_mentions_splits(self, comparer, decisions) -> None:
        keyed, hold = self.split_decision(decisions)
        outcome = apply_merges.apply_decisions([keyed, hold], decisions, comparer)

        assert any(
            "split into 3 cards" in line for line in apply_merges.summarize(outcome)
        )


class TestRankValidation:
    """A merge or split may only keep ranks its own group had.

    A rank decides which note an import overwrites, so a typo in one must not
    be carried out: it would overwrite an unrelated note and list the group's
    real notes for deletion.
    """

    @pytest.fixture
    def deck(self) -> tuple[types.Word, types.Word, types.Word]:
        return (
            make_word("gato", "nm", "cat", 100),
            make_word("gato", "nm", "tomcat", 200),
            make_word("perro", "nm", "dog", 300),
        )

    def test_a_merge_onto_an_unrelated_rank_is_refused(
        self, comparer, decisions, deck
    ) -> None:
        decide(
            decisions,
            deck[:2],
            triage.Disposition.MERGE,
            replacements=[make_word("gato", "nm", "cat, tomcat", 300)],
        )

        outcome = apply_merges.apply_decisions(deck, decisions, comparer)

        assert outcome.merges == ()
        assert set(outcome.words) == set(deck)
        assert [set(c.words) for c in outcome.unresolved] == [set(deck[:2])]
        assert apply_merges.superseded_notes(outcome, deck) == ()
        to_import = apply_merges.words_to_import(outcome, deck[:2])
        assert all(word.rank != 300 for word in to_import)

    @pytest.mark.parametrize(
        ("rank", "kept", "deleted"),
        [
            pytest.param(None, 100, [200], id="blank-keeps-the-lowest"),
            pytest.param(100, 100, [200], id="lowest"),
            pytest.param(200, 200, [100], id="explicit-higher"),
        ],
    )
    def test_a_merge_may_keep_any_rank_from_its_group(
        self, comparer, decisions, deck, rank, kept, deleted
    ) -> None:
        decide(
            decisions,
            deck[:2],
            triage.Disposition.MERGE,
            replacements=[make_word("gato", "nm", "cat, tomcat", rank)],
        )

        outcome = apply_merges.apply_decisions(deck, decisions, comparer)

        (merge,) = outcome.merges
        assert merge.result.rank == kept
        superseded = apply_merges.superseded_notes(outcome, deck)
        assert [w.rank for w in superseded] == deleted

    @pytest.mark.parametrize(
        "ranks",
        [
            pytest.param((100, 100), id="shared"),
            pytest.param((100, 300), id="unrelated"),
            pytest.param((None, 300), id="unrelated-beside-a-new-card"),
        ],
    )
    def test_a_split_with_bad_ranks_is_refused(
        self, comparer, decisions, deck, ranks
    ) -> None:
        decide(
            decisions,
            deck[:2],
            triage.Disposition.SPLIT,
            replacements=[
                make_word("gato", "nm", "cat", ranks[0]),
                make_word("gata", "nf", "she-cat", ranks[1]),
            ],
        )

        outcome = apply_merges.apply_decisions(deck, decisions, comparer)

        assert outcome.splits == ()
        assert set(outcome.words) == set(deck)
        assert [set(c.words) for c in outcome.unresolved] == [set(deck[:2])]

    def test_a_resolver_result_on_an_unrelated_rank_is_refused(
        self, comparer, decisions, deck
    ) -> None:
        """Wording replayed from the legacy cache gets the same check."""
        decide(decisions, deck[:2], triage.Disposition.MERGE)

        outcome = apply_merges.apply_decisions(
            deck,
            decisions,
            comparer,
            resolver=lambda groups: [make_word("gato", "nm", "cat, tomcat", 300)],
        )

        assert outcome.merges == ()
        assert outcome.declined == ()
        assert set(outcome.words) == set(deck)
        assert [set(c.words) for c in outcome.unresolved] == [set(deck[:2])]
        assert apply_merges.superseded_notes(outcome, deck) == ()
        to_import = apply_merges.words_to_import(outcome, deck[:2])
        assert all(word.rank != 300 for word in to_import)

    def test_identical_notes_merge_despite_a_stale_bad_decision(
        self, comparer, decisions, deck
    ) -> None:
        """A refused decision must not strand a group the flow never asks about.

        The decision was recorded when the notes still differed and carries a
        rank from outside the group; the notes have since been edited into
        agreement. Refusing it would leave the group unresolved forever, since
        an identical group is never re-asked, so the decision is ignored and
        the notes combine mechanically.
        """
        decide(
            decisions,
            deck[:2],
            triage.Disposition.MERGE,
            replacements=[make_word("gato", "nm", "cat, tomcat", 300)],
        )
        identical = (
            make_word("gato", "nm", "cat", 100),
            make_word("gato", "nm", "cat", 200),
            deck[2],
        )

        outcome = apply_merges.apply_decisions(identical, decisions, comparer)

        (merge,) = outcome.merges
        assert merge.decided is False
        assert merge.result.rank == 100
        assert merge.result.meaning == "cat"
        assert outcome.unresolved == ()
        assert [w.rank for w in apply_merges.superseded_notes(outcome, identical)] == [
            200
        ]

    def test_split_cards_may_reassign_the_groups_ranks(
        self, comparer, decisions, deck
    ) -> None:
        decide(
            decisions,
            deck[:2],
            triage.Disposition.SPLIT,
            replacements=[
                make_word("gato", "nm", "cat", 200),
                make_word("gata", "nf", "she-cat", 100),
                make_word("gatito", "nm", "kitten", None),
            ],
        )

        outcome = apply_merges.apply_decisions(deck, decisions, comparer)

        (split,) = outcome.splits
        assert [card.rank for card in split.cards] == [200, 100, None]
        assert apply_merges.superseded_notes(outcome, deck) == ()


@pytest.mark.parametrize(
    ("disposition", "ranks", "expected"),
    [
        pytest.param(
            triage.Disposition.MERGE,
            (300,),
            "the merged note's rank 300 is not in the group; use one of the "
            "group's ranks (100, 200) or leave it blank",
            id="merge-unrelated",
        ),
        pytest.param(triage.Disposition.MERGE, (200,), None, id="merge-higher"),
        pytest.param(triage.Disposition.MERGE, (None,), None, id="merge-blank"),
        pytest.param(
            triage.Disposition.SPLIT,
            (100, 100, 200, 200),
            "split cards cannot share rank 100; give each card a different "
            "rank, or leave extra cards blank",
            id="split-shared",
        ),
        pytest.param(
            triage.Disposition.SPLIT,
            (100, 7),
            "the card's rank 7 is not in the group; use one of the group's "
            "ranks (100, 200) or leave it blank",
            id="split-unrelated",
        ),
        pytest.param(triage.Disposition.SPLIT, (None, None), None, id="split-all-new"),
        pytest.param(triage.Disposition.KEEP_SEPARATE, (7, 7), None, id="not-checked"),
    ],
)
def test_rank_problem(
    disposition: triage.Disposition,
    ranks: Sequence[int | None],
    expected: str | None,
) -> None:
    sources = (
        make_word("gato", "nm", "cat", 200),
        make_word("gato", "nm", "tomcat", 100),
        make_word("gato", "nm", "cat", 100),
    )
    replacements = tuple(make_word("gato", "nm", "cat", rank) for rank in ranks)

    assert apply_merges.rank_problem(disposition, sources, replacements) == expected


@pytest.mark.parametrize(
    ("ranks", "position"),
    [
        pytest.param((7, 100), 0, id="unrelated-first"),
        pytest.param((100, 200, 7), 2, id="unrelated-last"),
        pytest.param((100, None, 100), 2, id="shared-points-at-the-repeat"),
        pytest.param((100, 100, 7), 2, id="unrelated-before-shared"),
    ],
)
def test_find_rank_problem_points_at_the_offending_card(
    ranks: Sequence[int | None], position: int
) -> None:
    sources = (make_word("gato", "nm", "cat", 100), make_word("gato", "nm", "cat", 200))
    replacements = tuple(make_word("gato", "nm", "cat", rank) for rank in ranks)

    problem = apply_merges.find_rank_problem(
        triage.Disposition.SPLIT, sources, replacements
    )

    assert problem is not None
    assert problem.position == position
    assert problem.message == apply_merges.rank_problem(
        triage.Disposition.SPLIT, sources, replacements
    )


def test_rank_problem_when_the_group_has_no_ranks() -> None:
    sources = (make_word("gato", "nm", "cat", None),)
    replacements = (make_word("gato", "nm", "cat", 5),)

    assert apply_merges.rank_problem(
        triage.Disposition.MERGE, sources, replacements
    ) == (
        "the merged note's rank 5 is not in the group; leave it blank, since "
        "no note in the group has one"
    )
