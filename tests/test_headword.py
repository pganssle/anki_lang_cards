"""Tests for headword canonicalization."""

import unicodedata

import hypothesis
import pytest
from hypothesis import strategies as st

from anki_lang_cards import headword


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        # No slash: the word expands to itself.
        ("perro", ("perro",)),
        ("mirón", ("mirón",)),
        ("nombre de pila", ("nombre de pila",)),
        # "o/a" and "a/o" swap the final vowel.
        ("perro/a", ("perro", "perra")),
        ("científico/a", ("científico", "científica")),
        ("modista/o", ("modista", "modisto")),
        # Plurals work the same way two characters at a time.
        ("perros/as", ("perros", "perras")),
        ("as/os", ("as", "os")),
        # A bare "/a" appends the feminine and drops the now-redundant accent.
        ("soplón/a", ("soplón", "soplona")),
        ("besucón/a", ("besucón", "besucona")),
        ("francés/a", ("francés", "francesa")),
        ("español/a", ("español", "española")),
        # Suffix notation restates part of the stem.
        ("mirón/ona", ("mirón", "mirona")),
        ("francés/esa", ("francés", "francesa")),
        # A suffix that restates nothing is a whole word.
        ("actor/actriz", ("actor", "actriz")),
        ("padre/madre", ("padre", "madre")),
    ],
)
def test_gendered_forms(word: str, expected: tuple[str, ...]) -> None:
    assert headword.gendered_forms(word) == expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        # The case this exists for: two spellings of one pair.
        ("mirón/ona", "mirón/a"),
        ("francés/esa", "francés/a"),
        ("soplón/ona", "soplón/a"),
        # Already canonical.
        ("mirón/a", "mirón/a"),
        ("perro/a", "perro/a"),
        ("soplón/a", "soplón/a"),
        # Not a suffix pair, so there is no shorthand to collapse onto.
        ("actor/actriz", "actor/actriz"),
        ("padre/madre", "padre/madre"),
        # Nothing to do.
        ("perro", "perro"),
        ("nombre de pila", "nombre de pila"),
    ],
)
def test_canonical_form(word: str, expected: str) -> None:
    assert headword.canonical_form(word) == expected


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("mirón/ona", "mirón/a"),
        ("francés/esa", "francés/a"),
        ("perro/a", "  perro/a  "),
        ("Perro/a", "perro/a"),
    ],
)
def test_equivalent_spellings_share_a_lookup_key(first: str, second: str) -> None:
    assert headword.lookup_key(first) == headword.lookup_key(second)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("perro/a", "gato/a"),
        ("actor/actriz", "actor/a"),
        ("mirón/a", "mirona"),
        ("cura", "curar"),
    ],
)
def test_distinct_headwords_keep_distinct_lookup_keys(first: str, second: str) -> None:
    assert headword.lookup_key(first) != headword.lookup_key(second)


def test_decomposed_and_composed_spellings_agree() -> None:
    composed = unicodedata.normalize("NFC", "mirón/a")
    decomposed = unicodedata.normalize("NFD", "mirón/a")

    assert composed != decomposed
    assert headword.lookup_key(composed) == headword.lookup_key(decomposed)
    assert headword.gendered_forms(decomposed) == ("mirón", "mirona")


def test_deaccenting_stops_at_the_first_unaccented_vowel() -> None:
    """Ported quirk: only the accent after the last plain vowel is removed."""
    assert headword.gendered_forms("soplón/a") == ("soplón", "soplona")
    # "u" is reached before "ó" going backwards from the end, so it stops there.
    assert headword.gendered_forms("besucón/a") == ("besucón", "besucona")


@hypothesis.given(st.text(min_size=1, max_size=20).filter(lambda s: "/" not in s))
def test_headwords_without_slashes_are_only_normalized(word: str) -> None:
    assert headword.gendered_forms(word) == (
        unicodedata.normalize("NFC", word).strip(),
    )


@hypothesis.given(st.text(min_size=1, max_size=20))
def test_canonical_form_is_idempotent(word: str) -> None:
    once = headword.canonical_form(word)
    assert headword.canonical_form(once) == once


@hypothesis.given(st.text(min_size=1, max_size=20))
def test_canonicalization_preserves_the_expansion(word: str) -> None:
    """Rewriting the notation must never change what the card displays."""
    assert headword.gendered_forms(headword.canonical_form(word)) == (
        headword.gendered_forms(word)
    )


@pytest.mark.parametrize(
    "word",
    [
        "percatarse (de)",
        "avenirse (a)",
        "dar paso (a)",
        "mofarse (de)",
        "quedarse (con algo)",
        "agarrarse (a algo)",
        "hacer caso omiso (de)",
        "hacerle la rosca (a alguien)",
        "ponérselo fácil (a alguien)",
        "quedar en (que)",
        "poder más (que)",
        "dársele algo (a alguien)",
        # An empty parenthetical cannot be picking out a sense.
        "percatarse ()",
    ],
)
def test_valency_markers_are_recognized(word: str) -> None:
    kinds = [kind for _, kind in headword.parenthetical_markers(word)]
    assert kinds == [headword.MarkerKind.VALENCY]


@pytest.mark.parametrize(
    "word",
    [
        "llave (lucha)",
        "mono (ropa)",
        "casco (de buque)",
        "cuerno (ropa)",
        "luna (de un coche)",
        "toma (de corriente)",
        "puente (periodo del tiempo)",
        "gastar (ropa, zapatos)",
        "arrastrar (las palabras)",
        "allanamiento (de morada)",
        "mariquita (dicho de un hombre)",
        "sortear (obstáculo)",
        # A content word makes this a sense marker even though it is a slot;
        # erring toward "never merge" is the safe direction.
        "llevarle (años) a alguien",
    ],
)
def test_sense_markers_are_recognized(word: str) -> None:
    kinds = [kind for _, kind in headword.parenthetical_markers(word)]
    assert kinds == [headword.MarkerKind.SENSE]


def test_headword_without_parentheses_has_no_markers() -> None:
    assert headword.parenthetical_markers("percatarse") == ()


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("percatarse (de)", "percatarse"),
        ("dar paso (a)", "dar paso"),
        ("hacer caso omiso (de)", "hacer caso omiso"),
        ("tocarle (a alguien) con", "tocarle con"),
        # Sense markers survive untouched.
        ("llave (lucha)", "llave (lucha)"),
        ("gastar (ropa, zapatos)", "gastar (ropa, zapatos)"),
        # Nothing to strip.
        ("percatarse", "percatarse"),
    ],
)
def test_strip_valency_markers(word: str, expected: str) -> None:
    assert headword.strip_valency_markers(word) == expected


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("percatarse (de)", "percatarse"),
        ("avenirse (a)", "avenirse"),
        ("dar paso (a)", "dar paso"),
        ("sortear (obstáculo)", "sortear (obstáculo)"),
    ],
)
def test_valency_variants_share_a_lookup_key(first: str, second: str) -> None:
    assert headword.lookup_key(first) == headword.lookup_key(second)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        # The whole point of a sense marker is that it keeps cards apart.
        ("llave (lucha)", "llave"),
        ("mono (ropa)", "mono"),
        ("casco (de buque)", "casco"),
        ("luna (de un coche)", "luna"),
        ("sortear (obstáculo)", "sortear"),
        ("mariquita (dicho de un hombre)", "mariquita"),
    ],
)
def test_sense_variants_keep_distinct_lookup_keys(first: str, second: str) -> None:
    assert headword.lookup_key(first) != headword.lookup_key(second)


@hypothesis.given(st.text(min_size=1, max_size=30))
def test_stripping_valency_markers_is_idempotent(word: str) -> None:
    once = headword.strip_valency_markers(word)
    assert headword.strip_valency_markers(once) == once


@hypothesis.given(st.text(min_size=1, max_size=30))
def test_stripping_never_removes_a_sense_marker(word: str) -> None:
    stripped = headword.strip_valency_markers(word)
    sense_markers = [
        contents
        for contents, kind in headword.parenthetical_markers(word)
        if kind is headword.MarkerKind.SENSE
    ]
    for contents in sense_markers:
        assert f"({contents})" in stripped
