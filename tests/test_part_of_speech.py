from collections.abc import Sequence
from typing import Final

import hypothesis
import pytest
from hypothesis import strategies as st

from anki_lang_cards import part_of_speech as pos

PART_OF_SPEECH_BASES: Final[Sequence[pos.BasePoS]] = tuple(
    base_pos
    for base_pos in pos.BasePoS
    # These two are special and cannot be used with PartOfSpeech
    if base_pos not in (pos.BasePoS.NOUN, pos.BasePoS.VERB)
)


@pytest.mark.parametrize(
    ("pos1", "pos2"),
    [
        (
            pos.PartOfSpeech(base_pos, phrasal=phrasal1),
            pos.PartOfSpeech(base_pos, phrasal=phrasal2),
        )
        for base_pos in PART_OF_SPEECH_BASES
        for phrasal1 in pos.Phrasality
        for phrasal2 in pos.Phrasality
    ],
)
def test_part_of_speech_equal(pos1: pos.PartOfSpeech, pos2: pos.PartOfSpeech) -> None:
    assert pos1 == pos2


@hypothesis.given(
    # We do this "select an index" thing because we want to filter to make sure
    # these are not the same, but that relies on comparison, which is what we
    # are trying to test here.
    st.tuples(
        st.sampled_from(PART_OF_SPEECH_BASES),
        st.sampled_from(PART_OF_SPEECH_BASES),
        st.sampled_from(pos.Phrasality),
        st.sampled_from(pos.Phrasality),
    )
    .filter(lambda x: (x[0] != x[1]))
    .map(
        lambda x: (
            pos.PartOfSpeech(x[0], phrasal=x[2]),
            pos.PartOfSpeech(x[1], phrasal=x[3]),
        )
    )
)
def test_part_of_speech_unequal(
    comparands: tuple[pos.PartOfSpeech, pos.PartOfSpeech],
) -> None:
    pos1, pos2 = comparands
    assert pos1 != pos2


@hypothesis.given(
    st.tuples(st.sampled_from(PART_OF_SPEECH_BASES), st.sampled_from(pos.Phrasality))
)
def test_part_of_speech_merge_same_phrasality(
    data: tuple[pos.BasePoS, pos.Phrasality],
) -> None:
    base_pos, phrasality = data
    pos1 = pos.PartOfSpeech(base_pos, phrasal=phrasality)
    pos2 = pos.PartOfSpeech(base_pos, phrasal=phrasality)

    # TODO: Use subtests?
    merged_12 = pos1.merge(pos2)
    merged_21 = pos2.merge(pos1)

    assert merged_12 == pos1
    assert merged_21 == pos1
    assert merged_12 == pos2
    assert merged_21 == pos2
    assert merged_12 == merged_21
    assert merged_21 == pos1


@hypothesis.given(
    st.tuples(
        st.sampled_from(PART_OF_SPEECH_BASES),
        st.sampled_from(pos.Phrasality),
        st.sampled_from(pos.Phrasality),
    ).filter(lambda x: x[1] != x[2])
)
def test_part_of_speech_merge_different_phrasality(
    data: tuple[pos.BasePoS, pos.Phrasality, pos.Phrasality],
) -> None:
    base_pos, phr1, phr2 = data
    pos1 = pos.PartOfSpeech(base_pos, phrasal=phr1)
    pos2 = pos.PartOfSpeech(base_pos, phrasal=phr2)

    if pos.Phrasality.PHRASAL in (phr1, phr2):
        expected_phr = pos.Phrasality.PHRASAL
    else:
        expected_phr = pos.Phrasality.NON_PHRASAL

    expected = pos.PartOfSpeech(base_pos, phrasal=expected_phr)

    # TODO: Use subtests
    # TODO: assert that a warning is raised
    merged_12 = pos1.merge(pos2)
    merged_21 = pos2.merge(pos1)

    assert merged_12 == merged_21
    assert merged_12 == expected


@pytest.mark.parametrize(
    ("pos_str", "expected"),
    [
        (
            "nm",
            pos.Noun(
                gender=pos.Gender.MASC,
                number=pos.Number.UNSPECIFIED,
                phrasal=pos.Phrasality.NON_PHRASAL,
            ),
        ),
        (
            "nf",
            pos.Noun(
                gender=pos.Gender.FEM,
                number=pos.Number.UNSPECIFIED,
                phrasal=pos.Phrasality.NON_PHRASAL,
            ),
        ),
        (
            "nm pl",
            pos.Noun(
                gender=pos.Gender.MASC,
                number=pos.Number.PLURAL,
                phrasal=pos.Phrasality.NON_PHRASAL,
            ),
        ),
        (
            "nf pl",
            pos.Noun(
                gender=pos.Gender.FEM,
                number=pos.Number.PLURAL,
                phrasal=pos.Phrasality.NON_PHRASAL,
            ),
        ),
        (
            "nf pl phr",
            pos.Noun(
                gender=pos.Gender.FEM,
                number=pos.Number.PLURAL,
                phrasal=pos.Phrasality.PHRASAL,
            ),
        ),
        (
            "nm/f",
            pos.Noun(
                gender=pos.Gender.FORM_DEPENDENT,
                number=pos.Number.UNSPECIFIED,
                phrasal=pos.Phrasality.NON_PHRASAL,
            ),
        ),
        (
            "nc",
            pos.Noun(
                gender=pos.Gender.COMMON,
                number=pos.Number.UNSPECIFIED,
                phrasal=pos.Phrasality.NON_PHRASAL,
            ),
        ),
        (
            "nmf",
            pos.Noun(
                gender=pos.Gender.AMBIGUOUS,
                number=pos.Number.UNSPECIFIED,
                phrasal=pos.Phrasality.NON_PHRASAL,
            ),
        ),
        (
            "nc pl",
            pos.Noun(
                gender=pos.Gender.COMMON,
                number=pos.Number.PLURAL,
                phrasal=pos.Phrasality.NON_PHRASAL,
            ),
        ),
        (
            "n",
            pos.Noun(
                gender=pos.Gender.UNSPECIFIED,
                number=pos.Number.UNSPECIFIED,
                phrasal=pos.Phrasality.UNSPECIFIED,
            ),
        ),
        (
            "n pl",
            pos.Noun(
                gender=pos.Gender.UNSPECIFIED,
                number=pos.Number.PLURAL,
                phrasal=pos.Phrasality.NON_PHRASAL,
            ),
        ),
        (
            "v",
            pos.Verb(
                transitivity=pos.Transitivity.UNSPECIFIED,
            ),
        ),
        (
            "vi",
            pos.Verb(
                transitivity=pos.Transitivity.INTRANSITIVE,
                phrasal=pos.Phrasality.NON_PHRASAL,
            ),
        ),
        (
            "vtr",
            pos.Verb(
                transitivity=pos.Transitivity.TRANSITIVE,
                phrasal=pos.Phrasality.NON_PHRASAL,
            ),
        ),
        (
            "vtr phr",
            pos.Verb(
                transitivity=pos.Transitivity.TRANSITIVE, phrasal=pos.Phrasality.PHRASAL
            ),
        ),
        (
            "vi phr",
            pos.Verb(
                transitivity=pos.Transitivity.INTRANSITIVE,
                phrasal=pos.Phrasality.PHRASAL,
            ),
        ),
        (
            "adj",
            pos.PartOfSpeech(base_pos=pos.BasePoS.ADJECTIVE),
        ),
        (
            "adj phr",
            pos.PartOfSpeech(
                base_pos=pos.BasePoS.ADJECTIVE, phrasal=pos.Phrasality.PHRASAL
            ),
        ),
        (
            "prep",
            pos.PartOfSpeech(base_pos=pos.BasePoS.PREPOSITION),
        ),
        (
            "prep phr",
            pos.PartOfSpeech(
                base_pos=pos.BasePoS.PREPOSITION, phrasal=pos.Phrasality.PHRASAL
            ),
        ),
        (
            "art",
            pos.PartOfSpeech(base_pos=pos.BasePoS.ARTICLE),
        ),
        (
            "conj",
            pos.PartOfSpeech(base_pos=pos.BasePoS.CONJUNCTION),
        ),
        (
            "interj",
            pos.PartOfSpeech(base_pos=pos.BasePoS.INTERJECTION),
        ),
        (
            "num",
            pos.PartOfSpeech(base_pos=pos.BasePoS.NUMBER),
        ),
        (
            "pron",
            pos.PartOfSpeech(base_pos=pos.BasePoS.PRONOUN),
        ),
    ],
)
def test_parse_part_of_speech(pos_str: str, expected: pos.PartOfSpeech) -> None:
    actual = pos.parse_part_of_speech(pos_str)

    assert actual == expected
    assert actual.phrasal == expected.phrasal


@pytest.mark.parametrize(
    "pos_str",
    ["", " ", "zzz", "nx", "vx", "nm x", "vtr x y", "prep x"],
)
def test_parse_part_of_speech_invalid(pos_str: str) -> None:
    """Invalid inputs raise PartOfSpeechParseError, not IndexError/KeyError.

    An empty string previously raised a bare IndexError, which escaped the
    merge UI's error handling and took down the whole merge session."""
    with pytest.raises(pos.PartOfSpeechParseError):
        pos.parse_part_of_speech(pos_str)
