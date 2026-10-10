"""Tests for pre-merge duplicate triage."""

import json
import pathlib
from collections.abc import Sequence
from typing import Final

import attrs
import pytest

from anki_lang_cards import _types as types
from anki_lang_cards import definition_similarity, triage

CORPUS: Final[Sequence[str]] = (
    "social climber",
    "old fogey",
    "priest",
    "cure, remedy",
    "helmet",
    "hull of a ship",
    "settlement, colony",
    "dog",
    "cat",
    "house",
)


@pytest.fixture
def comparer() -> definition_similarity.DefinitionComparer:
    return definition_similarity.DefinitionComparer(CORPUS)


def make_word(
    word: str,
    part_of_speech: str,
    meaning: str,
    rank: int | None = None,
) -> types.Word:
    return types.Word(
        rank=rank,
        word=word,
        meaning=meaning,
        part_of_speech=part_of_speech,
        article=None,
        sentences=(),
    )


def test_words_with_no_overlap_produce_no_candidates(comparer) -> None:
    words = [
        make_word("perro", "nm", "dog", rank=1),
        make_word("gato", "nm", "cat", rank=2),
    ]
    assert triage.find_candidates(words, comparer) == ()


def test_compatible_conflict_is_flagged(comparer) -> None:
    words = [
        make_word("trepa", "nm/f", "social climber", rank=8576),
        make_word("trepa", "nm/f", "social climber", rank=9615),
    ]
    (candidate,) = triage.find_candidates(words, comparer)

    assert candidate.reason is triage.Reason.COMPATIBLE
    assert candidate.disposition is triage.Disposition.MERGE
    assert candidate.comparison.score == pytest.approx(1.0)


def test_incompatible_part_of_speech_is_flagged_when_definitions_agree(
    comparer,
) -> None:
    """The "nm/f" against "nf" case, which no part-of-speech rule can catch."""
    words = [
        make_word("trepa", "nm/f", "social climber", rank=8576),
        make_word("trepa", "nf", "social climber", rank=9615),
    ]
    (candidate,) = triage.find_candidates(words, comparer)

    assert candidate.reason is triage.Reason.CHARACTERISTICS
    assert candidate.disposition is triage.Disposition.MERGE


def test_genuine_gender_split_is_listed_but_not_proposed(comparer) -> None:
    """ "el cura"/"la cura" must never be suggested for merging."""
    words = [
        make_word("cura", "nm", "priest", rank=1790),
        make_word("cura", "nf", "cure, remedy", rank=9000),
    ]
    (candidate,) = triage.find_candidates(words, comparer)

    assert candidate.reason is triage.Reason.GENDER
    assert candidate.disposition is triage.Disposition.REVIEW


def test_gender_splits_are_never_proposed_even_when_definitions_agree(
    comparer,
) -> None:
    """ "el orden" (sequence) and "la orden" (religious order) both gloss as
    "order", so the definitions cannot settle a gender split."""
    words = [
        make_word("orden", "nm", "order, sequence", rank=421),
        make_word("orden", "nf", "religious order", rank=6022),
    ]
    (candidate,) = triage.find_candidates(words, comparer)

    assert candidate.reason is triage.Reason.GENDER
    assert candidate.disposition is triage.Disposition.REVIEW


def test_unrestricted_gender_against_a_specific_one_may_be_proposed(
    comparer,
) -> None:
    """ "nm/f" claims both forms exist, so it does not contradict "nf"."""
    words = [
        make_word("trepa", "nm/f", "social climber", rank=1),
        make_word("trepa", "nf", "social climber", rank=2),
    ]
    (candidate,) = triage.find_candidates(words, comparer)

    assert candidate.reason is triage.Reason.CHARACTERISTICS
    assert candidate.disposition is triage.Disposition.MERGE


def test_differing_base_part_of_speech_is_never_proposed(comparer) -> None:
    """ "segundo" is "second" as both adjective and noun, and they are two
    cards; identical definitions must not propose merging them."""
    words = [
        make_word("segundo", "adj", "second", rank=223),
        make_word("segundo", "nm", "second", rank=1693),
    ]
    (candidate,) = triage.find_candidates(words, comparer)

    assert candidate.reason is triage.Reason.BASE_PART_OF_SPEECH
    assert candidate.comparison.score == pytest.approx(1.0)
    assert candidate.disposition is triage.Disposition.REVIEW


def test_verb_transitivity_disagreement_may_be_proposed(comparer) -> None:
    """Transitivity is not on the front of the card, so unlike gender it never
    distinguishes two cards for the same verb."""
    words = [
        make_word("percatarse", "vtr phr", "to notice, realize", rank=5676),
        make_word("percatarse (de)", "vi", "to notice, realize", rank=9595),
    ]
    (candidate,) = triage.find_candidates(words, comparer)

    assert candidate.reason is triage.Reason.CHARACTERISTICS
    assert candidate.disposition is triage.Disposition.MERGE


def test_sense_split_across_parts_of_speech_is_not_proposed(comparer) -> None:
    words = [
        make_word("casco", "nm", "helmet", rank=3098),
        make_word("casco (de buque)", "nm", "hull of a ship", rank=9441),
    ]
    # A sense marker keeps the headwords apart entirely, so nothing is flagged.
    assert triage.find_candidates(words, comparer) == ()


def test_nothing_is_ever_suggested_as_keep_separate(comparer) -> None:
    words = [
        make_word("cura", "nm", "priest", rank=1),
        make_word("cura", "nf", "cure, remedy", rank=2),
        make_word("trepa", "nm/f", "social climber", rank=3),
        make_word("trepa", "nf", "social climber", rank=4),
    ]
    dispositions = {c.disposition for c in triage.find_candidates(words, comparer)}
    assert triage.Disposition.KEEP_SEPARATE not in dispositions


def test_undecided_candidates_come_first(comparer) -> None:
    """The file is read top-down, so put the questions before the answers."""
    words = [
        make_word("cura", "nm", "priest", rank=1),
        make_word("cura", "nf", "cure, remedy", rank=2),
        make_word("trepa", "nm/f", "social climber", rank=3),
        make_word("trepa", "nf", "social climber", rank=4),
    ]
    candidates = triage.find_candidates(words, comparer)

    assert [c.disposition for c in candidates] == [
        triage.Disposition.REVIEW,
        triage.Disposition.MERGE,
    ]


def test_candidates_are_ordered_by_agreement_within_a_disposition(comparer) -> None:
    words = [
        make_word("cura", "nm", "priest", rank=1),
        make_word("cura", "nf", "cure, remedy", rank=2),
        make_word("orden", "nm", "order, sequence", rank=3),
        make_word("orden", "nf", "religious order", rank=4),
    ]
    candidates = triage.find_candidates(words, comparer)
    scores = [c.comparison.score for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_partition_splits_by_disposition(comparer) -> None:
    words = [
        make_word("trepa", "nm/f", "social climber", rank=1),
        make_word("trepa", "nf", "social climber", rank=2),
        make_word("cura", "nm", "priest", rank=3),
        make_word("cura", "nf", "cure, remedy", rank=4),
    ]
    partitioned = triage.partition(triage.find_candidates(words, comparer))

    assert len(partitioned[triage.Disposition.MERGE]) == 1
    assert len(partitioned[triage.Disposition.REVIEW]) == 1
    assert partitioned[triage.Disposition.KEEP_SEPARATE] == ()


def test_to_merge_sets_excludes_only_the_separated(comparer) -> None:
    merged = triage.Candidate(
        words=(make_word("a", "nm", "x", rank=1),),
        reason=triage.Reason.COMPATIBLE,
        comparison=definition_similarity.Comparison(1.0, frozenset()),
        disposition=triage.Disposition.MERGE,
    )
    separate = triage.Candidate(
        words=(make_word("b", "nm", "y", rank=2),),
        reason=triage.Reason.CHARACTERISTICS,
        comparison=definition_similarity.Comparison(0.0, frozenset()),
        disposition=triage.Disposition.KEEP_SEPARATE,
    )
    assert triage.to_merge_sets([merged, separate]) == (frozenset(merged.words),)


class TestTriageFile:
    def test_round_trip_preserves_candidates(
        self, tmp_path: pathlib.Path, comparer
    ) -> None:
        words = [
            make_word("trepa", "nm/f", "social climber", rank=8576),
            make_word("trepa", "nf", "social climber", rank=9615),
        ]
        candidates = triage.find_candidates(words, comparer)
        path = tmp_path / "triage.json"
        triage.write_triage_file(path, candidates)

        restored = triage.read_triage_file(path)
        assert [c.disposition for c in restored] == [c.disposition for c in candidates]
        assert [c.reason for c in restored] == [c.reason for c in candidates]
        assert [sorted(w.word for w in c.words) for c in restored] == [
            sorted(w.word for w in c.words) for c in candidates
        ]

    def test_edited_disposition_is_honored(
        self, tmp_path: pathlib.Path, comparer
    ) -> None:
        words = [
            make_word("cura", "nm", "priest", rank=1790),
            make_word("cura", "nf", "cure, remedy", rank=9000),
        ]
        path = tmp_path / "triage.json"
        triage.write_triage_file(path, triage.find_candidates(words, comparer))

        payload = json.loads(path.read_text())
        payload[0]["disposition"] = "keep-separate"
        path.write_text(json.dumps(payload))

        (restored,) = triage.read_triage_file(path)
        assert restored.disposition is triage.Disposition.KEEP_SEPARATE
        assert triage.to_merge_sets([restored]) == ()

    def test_file_is_readable(self, tmp_path: pathlib.Path, comparer) -> None:
        """The point of the file is that it can be skimmed without tooling."""
        words = [
            make_word("trepa", "nm/f", "social climber", rank=8576),
            make_word("trepa", "nf", "social climber", rank=9615),
        ]
        path = tmp_path / "triage.json"
        triage.write_triage_file(path, triage.find_candidates(words, comparer))

        entry = json.loads(path.read_text())[0]
        assert entry["disposition"] == "merge"
        assert entry["reason"] == "characteristics"
        assert entry["summary"] == [
            "8576 trepa (nm/f) social climber",
            "9615 trepa (nf) social climber",
        ]


class TestDecisionCache:
    def test_answers_persist_across_runs(self, tmp_path: pathlib.Path) -> None:
        candidate = triage.Candidate(
            words=(
                make_word("cura", "nm", "priest", rank=1790),
                make_word("cura", "nf", "cure, remedy", rank=9000),
            ),
            reason=triage.Reason.GENDER,
            comparison=definition_similarity.Comparison(0.0, frozenset()),
            disposition=triage.Disposition.KEEP_SEPARATE,
        )
        path = tmp_path / "decisions.json"

        cache = triage.DecisionCache(path)
        assert candidate.words not in cache
        cache.remember(candidate)
        cache.save()

        assert candidate.words in triage.DecisionCache(path)

    def test_a_separated_group_stops_being_offered(
        self, tmp_path: pathlib.Path, comparer
    ) -> None:
        words = [
            make_word("cura", "nm", "priest", rank=1790),
            make_word("cura", "nf", "cure, remedy", rank=9000),
        ]
        cache = triage.DecisionCache(tmp_path / "decisions.json")
        (candidate,) = triage.find_candidates(words, comparer)
        cache.remember(
            attrs.evolve(candidate, disposition=triage.Disposition.KEEP_SEPARATE)
        )

        assert triage.find_candidates(words, comparer, decisions=cache) == ()

    def test_a_merge_answer_comes_back_answered(
        self, tmp_path: pathlib.Path, comparer
    ) -> None:
        """The bug this fixes: only separations were remembered, so a group
        answered "merge" was asked about again on the next run."""
        words = [
            make_word("no obstante", "conj", "nevertheless", rank=9087),
            make_word("no obstante", "adv", "nevertheless, however", rank=9265),
        ]
        cache = triage.DecisionCache(tmp_path / "decisions.json")
        (candidate,) = triage.find_candidates(words, comparer)
        assert candidate.disposition is triage.Disposition.REVIEW

        cache.remember(attrs.evolve(candidate, disposition=triage.Disposition.MERGE))
        (again,) = triage.find_candidates(words, comparer, decisions=cache)

        assert again.disposition is triage.Disposition.MERGE

    def test_revisions_are_remembered_too(
        self, tmp_path: pathlib.Path, comparer
    ) -> None:
        words = [
            make_word("segundo", "adj", "second", rank=223),
            make_word("segundo", "nm", "second", rank=1693),
        ]
        cache = triage.DecisionCache(tmp_path / "decisions.json")
        (candidate,) = triage.find_candidates(words, comparer)
        revised = attrs.evolve(
            candidate,
            disposition=triage.Disposition.MERGE,
            replacements=(
                make_word("segundo/a", "adj", "second", rank=223),
                make_word("segundo", "nm", "second", rank=1693),
            ),
        )
        cache.remember(revised)
        cache.save()

        (again,) = triage.find_candidates(
            words, comparer, decisions=triage.DecisionCache(cache.cache_path)
        )
        assert again.has_replacements
        assert {w.word for w in again.replacements} == {"segundo/a", "segundo"}

    def test_an_undecided_group_is_not_remembered(
        self, tmp_path: pathlib.Path, comparer
    ) -> None:
        """Leaving something for later is not an answer."""
        words = [
            make_word("cura", "nm", "priest", rank=1),
            make_word("cura", "nf", "cure, remedy", rank=2),
        ]
        cache = triage.DecisionCache(tmp_path / "decisions.json")
        candidates = triage.find_candidates(words, comparer)

        assert triage.record_decisions(candidates, cache) == 0
        assert len(cache) == 0

    def test_an_edit_is_remembered_even_when_undecided(
        self, tmp_path: pathlib.Path, comparer
    ) -> None:
        words = [
            make_word("segundo", "adj", "second", rank=223),
            make_word("segundo", "nm", "second", rank=1693),
        ]
        cache = triage.DecisionCache(tmp_path / "decisions.json")
        (candidate,) = triage.find_candidates(words, comparer)
        revised = attrs.evolve(
            candidate,
            replacements=(
                make_word("segundo/a", "adj", "second", rank=223),
                make_word("segundo", "nm", "second", rank=1693),
            ),
        )

        assert triage.record_decisions([revised], cache) == 1

    def test_recording_the_same_answer_twice_counts_once(
        self, tmp_path: pathlib.Path, comparer
    ) -> None:
        words = [
            make_word("cura", "nm", "priest", rank=1),
            make_word("cura", "nf", "cure, remedy", rank=2),
        ]
        cache = triage.DecisionCache(tmp_path / "decisions.json")
        (candidate,) = triage.find_candidates(words, comparer)
        answered = [attrs.evolve(candidate, disposition=triage.Disposition.MERGE)]

        assert triage.record_decisions(answered, cache) == 1
        assert triage.record_decisions(answered, cache) == 0
        assert len(cache) == 1

    def test_lookup_ignores_ordering(self, tmp_path: pathlib.Path) -> None:
        first = make_word("cura", "nm", "priest", rank=1790)
        second = make_word("cura", "nf", "cure", rank=9000)
        cache = triage.DecisionCache(tmp_path / "decisions.json")
        cache.remember(
            triage.Candidate(
                words=(first, second),
                reason=triage.Reason.GENDER,
                comparison=definition_similarity.Comparison(0.0, frozenset()),
                disposition=triage.Disposition.KEEP_SEPARATE,
            )
        )
        assert (second, first) in cache

    def test_the_old_rejection_format_still_loads(self, tmp_path: pathlib.Path) -> None:
        """Earlier versions stored bare CSV strings meaning "keep separate"."""
        words = (
            make_word("cura", "nm", "priest", rank=1790),
            make_word("cura", "nf", "cure, remedy", rank=9000),
        )
        new = triage.DecisionCache(tmp_path / "new.json")
        new.remember(
            triage.Candidate(
                words=words,
                reason=triage.Reason.GENDER,
                comparison=definition_similarity.Comparison(0.0, frozenset()),
                disposition=triage.Disposition.KEEP_SEPARATE,
            )
        )
        new.save()
        payload = json.loads(new.cache_path.read_text())

        legacy = tmp_path / "legacy.json"
        legacy.write_text(json.dumps([entry["words"] for entry in payload]))
        loaded = triage.DecisionCache(legacy)

        assert len(loaded) == 1
        assert loaded.get(words).disposition is triage.Disposition.KEEP_SEPARATE

    def test_missing_cache_file_is_empty(self, tmp_path: pathlib.Path) -> None:
        assert len(triage.DecisionCache(tmp_path / "nope.json")) == 0

    def test_save_creates_missing_directories(self, tmp_path: pathlib.Path) -> None:
        cache = triage.DecisionCache(tmp_path / "deep" / "nested" / "d.json")
        cache.save()
        assert cache.cache_path.exists()

    def test_the_revisions_key_still_loads(self, tmp_path: pathlib.Path) -> None:
        """Replacements were called revisions before splits existed, and the
        decisions recorded under that name must keep replaying."""
        words = (
            make_word("segundo", "adj", "second", rank=223),
            make_word("segundo", "nm", "second", rank=1693),
        )
        rewritten = (
            make_word("segundo/a", "adj", "second", rank=223),
            make_word("segundo", "nm", "second", rank=1693),
        )
        new = triage.DecisionCache(tmp_path / "new.json")
        new.remember(
            triage.Candidate(
                words=words,
                reason=triage.Reason.BASE_PART_OF_SPEECH,
                comparison=definition_similarity.Comparison(0.0, frozenset()),
                disposition=triage.Disposition.KEEP_SEPARATE,
                replacements=rewritten,
            )
        )
        new.save()
        payload = json.loads(new.cache_path.read_text())
        for entry in payload:
            entry["revisions"] = entry.pop("replacements")

        legacy = tmp_path / "legacy.json"
        legacy.write_text(json.dumps(payload))
        decision = triage.DecisionCache(legacy).get(words)

        assert decision is not None
        assert decision.disposition is triage.Disposition.KEEP_SEPARATE
        assert decision.replacements is not None
        assert {w.word for w in decision.replacements} == {"segundo/a", "segundo"}

    @staticmethod
    def remember(
        cache: triage.DecisionCache,
        words: Sequence[types.Word],
        disposition: triage.Disposition,
    ) -> None:
        cache.remember(
            triage.Candidate(
                words=tuple(words),
                reason=triage.Reason.COMPATIBLE,
                comparison=definition_similarity.Comparison(0.0, frozenset()),
                disposition=disposition,
            )
        )

    def test_keys_differing_only_in_case_are_one_group(
        self, tmp_path: pathlib.Path
    ) -> None:
        cache = triage.DecisionCache(tmp_path / "decisions.json")
        bench = make_word("banco", "nm", "bench", rank=9)
        self.remember(
            cache,
            (make_word("Banco", "nm", "bank", rank=5), bench),
            triage.Disposition.KEEP_SEPARATE,
        )
        self.remember(
            cache,
            (make_word("banco", "nm", "Bank ", rank=5), bench),
            triage.Disposition.MERGE,
        )
        cache.save()

        assert len(cache) == 1
        reloaded = triage.DecisionCache(cache.cache_path)
        decision = reloaded.get((make_word("banco", "nm", "bank", rank=5), bench))
        assert decision is not None
        assert decision.disposition is triage.Disposition.MERGE

    def test_keys_that_sort_as_ties_are_both_saved(
        self, tmp_path: pathlib.Path
    ) -> None:
        """These keys differ only in gender, which Word ordering ignores, so
        they tie when sorted even though neither equals the other."""
        cache = triage.DecisionCache(tmp_path / "decisions.json")
        other = make_word("cura", "n", "priest", rank=1)
        self.remember(
            cache,
            (make_word("cura", "nm", "cure", rank=2), other),
            triage.Disposition.KEEP_SEPARATE,
        )
        self.remember(
            cache,
            (make_word("cura", "nf", "cure", rank=2), other),
            triage.Disposition.MERGE,
        )
        cache.save()

        assert len(triage.DecisionCache(cache.cache_path)) == 2


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (("banco", "nm", "bank"), ("Banco", "nm", "bank")),
        (("banco", "nm", "bank"), (" banco ", "nm", "BANK")),
        (("cura", "n", "priest"), ("cura", "nm", "priest")),
        (("ir", "v", "to go"), ("ir", "vi", "to go")),
        (("Ñandú", "nm", "rhea"), ("ñandú", "nm", "Rhea")),
    ],
)
def test_equal_words_hash_equal(
    first: tuple[str, str, str], second: tuple[str, str, str]
) -> None:
    a, b = make_word(*first), make_word(*second)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


class TestSplit:
    """A group can become several new cards instead of one or none."""

    def make_candidate(self) -> triage.Candidate:
        return triage.Candidate(
            words=(
                make_word("llave", "nf", "key, faucet", rank=810),
                make_word("llave", "nf", "wrestling hold", rank=9241),
            ),
            reason=triage.Reason.COMPATIBLE,
            comparison=definition_similarity.Comparison(0.1, frozenset()),
            disposition=triage.Disposition.SPLIT,
            replacements=(
                make_word("llave", "nf", "key, faucet", rank=810),
                make_word("llave (lucha)", "nf", "wrestling hold", rank=9241),
                make_word("llave (grifo)", "nf", "tap, faucet", rank=None),
            ),
        )

    def test_a_split_round_trips_through_the_cache(
        self, tmp_path: pathlib.Path
    ) -> None:
        candidate = self.make_candidate()
        cache = triage.DecisionCache(tmp_path / "decisions.json")
        cache.remember(candidate)
        cache.save()

        decision = triage.DecisionCache(cache.cache_path).get(candidate.words)
        assert decision is not None
        assert decision.disposition is triage.Disposition.SPLIT
        assert len(decision.replacements) == 3
        assert decision.replacements[2].rank is None

    def test_a_split_round_trips_through_the_triage_file(
        self, tmp_path: pathlib.Path
    ) -> None:
        candidate = self.make_candidate()
        path = tmp_path / "candidates.json"
        triage.write_triage_file(path, [candidate])
        (reloaded,) = triage.read_triage_file(path)

        assert reloaded.disposition is triage.Disposition.SPLIT
        assert reloaded.replacements == candidate.replacements

    def test_a_split_counts_as_answered(self, tmp_path: pathlib.Path) -> None:
        cache = triage.DecisionCache(tmp_path / "decisions.json")
        assert triage.record_decisions([self.make_candidate()], cache) == 1

    def test_a_split_group_stays_out_of_merge_sets(self) -> None:
        assert triage.to_merge_sets([self.make_candidate()]) == ()

    def test_a_split_never_reads_as_a_notewise_rewrite(self) -> None:
        """Split cards are not paired with the notes they came from, even when
        the counts happen to line up."""
        candidate = attrs.evolve(
            self.make_candidate(),
            replacements=(
                make_word("llave", "nf", "key, faucet", rank=810),
                make_word("llave (lucha)", "nf", "wrestling hold", rank=9241),
            ),
        )
        assert triage.collect_revisions([candidate]) == ()


class TestFocus:
    """Narrowing a run to the notes being added."""

    def test_groups_untouched_by_the_focus_are_dropped(self, comparer) -> None:
        existing = [
            make_word("segundo", "adj", "second", rank=223),
            make_word("segundo", "nm", "second", rank=1693),
        ]
        added = [
            make_word("trepa", "nm/f", "social climber", rank=8576),
            make_word("trepa", "nf", "social climber", rank=9615),
        ]
        candidates = triage.find_candidates(existing + added, comparer, focus=added)

        assert len(candidates) == 1
        assert {w.word for w in candidates[0].words} == {"trepa"}

    def test_a_group_is_kept_if_any_member_is_in_focus(self, comparer) -> None:
        deck = make_word("trepa", "nm/f", "social climber", rank=8576)
        added = make_word("trepa", "nf", "social climber", rank=9615)

        (candidate,) = triage.find_candidates([deck, added], comparer, focus=[added])
        assert len(candidate.words) == 2

    def test_no_focus_looks_at_everything(self, comparer) -> None:
        words = [
            make_word("segundo", "adj", "second", rank=223),
            make_word("segundo", "nm", "second", rank=1693),
            make_word("trepa", "nm/f", "social climber", rank=8576),
            make_word("trepa", "nf", "social climber", rank=9615),
        ]
        assert len(triage.find_candidates(words, comparer, focus=None)) == 2

    def test_an_empty_focus_finds_nothing(self, comparer) -> None:
        """Distinct from None: nothing was added, so nothing is about it."""
        words = [
            make_word("trepa", "nm/f", "social climber", rank=1),
            make_word("trepa", "nf", "social climber", rank=2),
        ]
        assert triage.find_candidates(words, comparer, focus=()) == ()
