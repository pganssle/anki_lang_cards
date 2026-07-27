"""Tests for weighted definition comparison."""

from collections.abc import Sequence
from typing import Final

import hypothesis
import pytest
from hypothesis import strategies as st

from anki_lang_cards import definition_similarity as ds

# A stand-in for the deck: enough definitions that common words such as
# "person" are down-weighted relative to distinctive ones.
CORPUS: Final[Sequence[str]] = (
    "dog, animal",
    "cat, animal",
    "a person who works",
    "a person who sleeps",
    "a person who eats",
    "a person who runs",
    "social climber",
    "old fogey",
    "priest",
    "cure, remedy",
    "settlement, colony",
    "helmet",
    "hull of a ship",
    "moon",
    "key, faucet, wrench",
)


@pytest.fixture
def comparer() -> ds.DefinitionComparer:
    return ds.DefinitionComparer(CORPUS)


@pytest.mark.parametrize(
    ("definition", "expected"),
    [
        ("social climber", {"social climber"}),
        ("old fogey (colloquial, derogatory)", {"old fogey"}),
        ("(colloquial, derogatory) old fogey", {"old fogey"}),
        ("to pity, to feel sorry for", {"pity", "feel sorry for"}),
        ("nevertheless; however", {"nevertheless", "however"}),
        ("windshield/car window", {"windshield", "car window"}),
        ("", set()),
    ],
)
def test_glosses(definition: str, expected: set[str]) -> None:
    assert ds.glosses(definition) == expected


def test_glosses_ignore_leading_infinitive_marker() -> None:
    assert ds.glosses("to pity") == ds.glosses("pity")


def test_tokens_drop_function_and_register_words() -> None:
    assert ds.tokens("(colloquial, derogatory) an old fogey") == {"old", "fogey"}


def test_tokens_ignore_accents() -> None:
    assert ds.tokens("café") == ds.tokens("cafe")


@pytest.mark.parametrize(
    ("first", "second"),
    [
        # Identical, modulo ordering and register labels.
        ("social climber", "social climber"),
        ("(colloquial, derogatory) old fogey", "old fogey (colloquial, derogatory)"),
        ("nevertheless; however", "nevertheless, however"),
        # One definition is a superset of the other.
        ("trench, ditch", "ditch"),
        ("overalls, jumpsuit", "overalls, jumpsuit, boilersuit"),
        ("settlement, habitation", "settlement, colony"),
    ],
)
def test_duplicate_definitions_agree_strongly(
    comparer: ds.DefinitionComparer, first: str, second: str
) -> None:
    assert comparer.compare(first, second).is_strong


@pytest.mark.parametrize(
    ("first", "second"),
    [
        # The deliberate sense splits that must survive untouched.
        ("key, faucet, wrench", "lock, hold in wrestling"),
        ("helmet", "hull of a ship"),
        ("moon", "windshield"),
        ("horn, antler", "toggle"),
        ("temporary", "temple, side of the head"),
        ("to tell, say", "a saying, a figure of speech"),
        ("bridge", "long weekend"),
        ("ladybug", "sissy"),
        # The counterexamples that motivate keeping part of speech out of it.
        ("priest", "cure, remedy"),
        ("loss", "lost, missing"),
    ],
)
def test_distinct_senses_do_not_agree(
    comparer: ds.DefinitionComparer, first: str, second: str
) -> None:
    comparison = comparer.compare(first, second)
    assert not comparison.is_strong
    assert not comparison.shared_glosses


def test_sharing_a_whole_gloss_is_enough(comparer: ds.DefinitionComparer) -> None:
    """Even a low token overlap counts if a whole sense matches verbatim."""
    comparison = comparer.compare("trench, ditch", "ditch")
    assert "ditch" in comparison.shared_glosses
    assert comparison.is_strong


def test_rare_agreement_outweighs_common_agreement(
    comparer: ds.DefinitionComparer,
) -> None:
    """Agreeing on "climber" says more than agreeing on "person"."""
    assert comparer.weight("climber") > comparer.weight("person")

    # Same shape either way — two tokens each, one of them shared — so the only
    # thing that can move the score is how rare the shared token is.
    rare = comparer.compare("climber alpha", "climber beta")
    common = comparer.compare("person alpha", "person beta")
    assert rare.score > common.score


def test_identical_definitions_score_one(comparer: ds.DefinitionComparer) -> None:
    assert comparer.compare("social climber", "social climber").score == 1.0


def test_empty_definitions_do_not_agree(comparer: ds.DefinitionComparer) -> None:
    assert comparer.compare("", "social climber").score == 0.0
    assert comparer.compare("", "").score == 0.0


def test_comparison_is_symmetric(comparer: ds.DefinitionComparer) -> None:
    first, second = "settlement, habitation", "settlement, colony"
    assert comparer.compare(first, second) == comparer.compare(second, first)


def test_compare_all_reports_the_weakest_pair(
    comparer: ds.DefinitionComparer,
) -> None:
    """A group only agrees if every member does."""
    definitions = ["social climber", "social climber", "helmet"]
    assert comparer.compare_all(definitions).score == pytest.approx(
        comparer.compare("social climber", "helmet").score
    )
    assert not comparer.compare_all(definitions).is_strong


def test_compare_all_of_a_single_definition_is_trivially_strong(
    comparer: ds.DefinitionComparer,
) -> None:
    assert comparer.compare_all(["anything"]).is_strong


def test_empty_corpus_still_compares(comparer: ds.DefinitionComparer) -> None:
    empty = ds.DefinitionComparer(())
    assert empty.compare("social climber", "social climber").is_strong
    assert not empty.compare("helmet", "moon").is_strong


@hypothesis.given(st.text(max_size=40), st.text(max_size=40))
def test_score_is_bounded(first: str, second: str) -> None:
    comparer = ds.DefinitionComparer(CORPUS)
    assert 0.0 <= comparer.compare(first, second).score <= 1.0


@hypothesis.given(st.text(min_size=1, max_size=40))
def test_a_definition_agrees_with_itself(definition: str) -> None:
    comparer = ds.DefinitionComparer(CORPUS)
    comparison = comparer.compare(definition, definition)
    if ds.tokens(definition) or ds.glosses(definition):
        assert comparison.is_strong
