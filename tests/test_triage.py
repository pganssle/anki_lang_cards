"""Tests for pre-merge duplicate triage."""

import json
import pathlib
from collections.abc import Sequence
from typing import Final

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


class TestRejectionCache:
    def test_rejections_persist_across_runs(self, tmp_path: pathlib.Path) -> None:
        words = (
            make_word("cura", "nm", "priest", rank=1790),
            make_word("cura", "nf", "cure, remedy", rank=9000),
        )
        path = tmp_path / "rejections.json"

        cache = triage.RejectionCache(path)
        assert words not in cache
        cache.reject(words)
        cache.save()

        assert words in triage.RejectionCache(path)

    def test_rejected_groups_stop_being_offered(
        self, tmp_path: pathlib.Path, comparer
    ) -> None:
        words = [
            make_word("cura", "nm", "priest", rank=1790),
            make_word("cura", "nf", "cure, remedy", rank=9000),
        ]
        cache = triage.RejectionCache(tmp_path / "rejections.json")

        (candidate,) = triage.find_candidates(words, comparer)
        cache.reject(candidate.words)

        assert triage.find_candidates(words, comparer, rejections=cache) == ()

    def test_rejection_ignores_ordering(self, tmp_path: pathlib.Path) -> None:
        first = make_word("cura", "nm", "priest", rank=1790)
        second = make_word("cura", "nf", "cure", rank=9000)
        cache = triage.RejectionCache(tmp_path / "rejections.json")

        cache.reject((first, second))
        assert (second, first) in cache

    def test_record_rejections_collects_from_a_triage_file(
        self, tmp_path: pathlib.Path, comparer
    ) -> None:
        words = [
            make_word("cura", "nm", "priest", rank=1790),
            make_word("cura", "nf", "cure, remedy", rank=9000),
        ]
        candidates = [
            c.__class__(
                words=c.words,
                reason=c.reason,
                comparison=c.comparison,
                disposition=triage.Disposition.KEEP_SEPARATE,
            )
            for c in triage.find_candidates(words, comparer)
        ]
        cache = triage.RejectionCache(tmp_path / "rejections.json")
        triage.record_rejections(candidates, cache)

        assert len(cache) == 1
        assert triage.find_candidates(words, comparer, rejections=cache) == ()

    def test_missing_cache_file_is_empty(self, tmp_path: pathlib.Path) -> None:
        assert len(triage.RejectionCache(tmp_path / "nope.json")) == 0

    def test_save_creates_missing_directories(self, tmp_path: pathlib.Path) -> None:
        cache = triage.RejectionCache(tmp_path / "deep" / "nested" / "r.json")
        cache.reject((make_word("cura", "nm", "priest", rank=1),))
        cache.save()
        assert cache.cache_path.exists()
