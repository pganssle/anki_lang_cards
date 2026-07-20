"""Module for representing parts of speech."""

import enum
import functools
import warnings
from collections.abc import Mapping
from typing import Any, Literal, Self, TypeAlias

import attrs


class BasePoS(enum.Enum):
    """Represents the base part of speech, with no modifiers"""

    NOUN = "n"
    VERB = "v"
    ADJECTIVE = "adj"
    ADVERB = "adv"
    ARTICLE = "art"
    CONJUNCTION = "conj"
    PREPOSITION = "prep"
    INTERJECTION = "interj"
    NUMBER = "num"
    PRONOUN = "pron"
    UNKNOWN = "unk"


class Gender(enum.Enum):
    """Noun characteristic

    Note that this only represents noun gender; since we are dealing with
    dictionary headwords, we do not consider adjectives to have gender, and
    the differences between something lie "adj mf" and "adj" are encoded in
    the headword itself, e.g. "enfermo/a" rather than "enfermo, adj mf".
    """

    MASC = "m"
    FEM = "f"
    COMMON = "c"
    FORM_DEPENDENT = "m/f"
    AMBIGUOUS = "mf"
    UNSPECIFIED = ""

    @classmethod
    @functools.cache
    def from_str(cls, s: str) -> Self:
        for potential in cls:
            if s == potential.value:
                return potential
        raise EnumParseError(f"Unknown string value {s} for {cls}")


class Number(enum.Enum):
    """Noun characteristic — whether it represents one or multiple things."""

    SINGULAR = enum.auto()
    PLURAL = enum.auto()
    UNSPECIFIED = enum.auto()


class Transitivity(enum.Enum):
    """Verb characteristic — whether it is transitive or intransitive."""

    TRANSITIVE = "tr"
    INTRANSITIVE = "i"
    UNSPECIFIED = ""

    @classmethod
    @functools.cache
    def from_str(cls, s: str) -> Self:
        for potential in cls:
            if s == potential.value:
                return potential
        raise EnumParseError(f"Unknown string value {s} for {cls}")


class Phrasality(enum.Enum):
    """Whether a word is a phrase or a single term"""

    PHRASAL = "phr"
    NON_PHRASAL = ""
    UNSPECIFIED = "unk"


_WordCharacteristic: TypeAlias = Gender | Number | Transitivity | Phrasality


def _compare_characteristic(c1: _WordCharacteristic, c2: _WordCharacteristic) -> bool:
    if type(c1) is not type(c2):
        raise TypeError(f"Cannot compare {c1} and {c2}")

    unspecified = type(c1).UNSPECIFIED
    return c1 == unspecified or c2 == unspecified or c1 == c2  # noqa: SIM102


def _merge_phrasality(
    p1: Phrasality,
    p2: Phrasality,
    *,
    on_mismatch: Literal["warn"] | Literal["error"] | Literal["ignore"] = "warn",
) -> Phrasality:
    if p1 == Phrasality.PHRASAL or p2 == Phrasality.PHRASAL:
        phrasality = Phrasality.PHRASAL
    elif p1 != Phrasality.UNSPECIFIED and p2 == Phrasality.UNSPECIFIED:
        phrasality = p1
    elif p2 != Phrasality.UNSPECIFIED and p1 == Phrasality.UNSPECIFIED:
        phrasality = p2
    else:
        assert p1 == p2
        phrasality = p1

    if p1 != p2 and p1 != Phrasality.UNSPECIFIED and p2 != Phrasality.UNSPECIFIED:
        match on_mismatch:
            case "warn":
                warnings.warn(
                    f"Phrasality mismatch: {p1} != {p2}. Defaulting to {phrasality}",
                    RuntimeWarning,
                )
            case "error":
                raise ValueError(f"Phrasality mismatch: {p1} != {p2}.")
            case "ignore":
                pass
            case _:
                raise ValueError(f"Unknown setting for on_mismatch: {on_mismatch}")
    return phrasality


@attrs.frozen
class PartOfSpeech:
    """Represents a full part of speech, with modifiers.

    This class exists to make it easier to have more complicated equality
    relationships between parts of speech. For example, "v" and "vtr" should
    have the same hash and compare equal, but "vi" and "vtr" should not,
    because "v" has the transitivity /unspecified/ and thus admits an
    interpretation where it is either "vi" or "vtr".

    Since some characteristics are unique to specific parts of speech (e.g.
    only nouns have grammatical gender), but most do not, the base class is
    used for anything without unique characteristics, and subclasses are
    used to represent parts of speech with other characteristics.
    """

    base_pos = attrs.field(
        type=BasePoS,
        validator=attrs.validators.not_(
            attrs.validators.in_({BasePoS.NOUN, BasePoS.VERB})
        ),
    )
    phrasal = attrs.field(
        type=Phrasality, default=Phrasality.UNSPECIFIED, eq=False, hash=False, repr=True
    )

    def __str__(self) -> str:
        if self.phrasal == Phrasality.PHRASAL:
            return f"{self.base_pos.value} phr"
        else:
            return self.base_pos.value

    def merge(self, other: Self) -> Self:
        """Create a new instance of the same type with compatible characteristics merged.

        Only parts of speech of the same base type (noun, verb, etc) can be
        merged, and among those, they must be compatible. The general rule here
        is that if a characteristics is unspecified in one of the two instances,
        the merged result will have the version that has been specified.

        So for example:

            "nm phr" + "n" -> "nm phr"
            "nf phr" + "nf" -> "nf phr" (with warning)
            "nf" + "nm" -> error
            "adj" + "n" -> error
        """
        if self.base_pos != other.base_pos:
            raise ValueError(
                f"Cannot merge incompatible parts of speech: '{self}' and '{other}'"
            )

        phrasality = _merge_phrasality(self.phrasal, other.phrasal, on_mismatch="warn")

        return type(self)(base_pos=self.base_pos, phrasal=phrasality)


@attrs.frozen
class Noun(PartOfSpeech):
    """Represents types of noun and noun phrases."""

    gender: Gender
    number: Number
    phrasal: Phrasality = Phrasality.UNSPECIFIED
    base_pos: BasePoS = attrs.field(
        init=False, eq=True, hash=True, repr=False, default=BasePoS.NOUN
    )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PartOfSpeech):
            return NotImplemented

        if self.base_pos != other.base_pos:
            return False

        assert isinstance(other, Noun)

        if not _compare_characteristic(self.gender, other.gender):
            return False
        if not _compare_characteristic(self.number, other.number):
            return False
        return True

    def __str__(self) -> str:
        base = "n"
        if self.gender != Gender.UNSPECIFIED:
            base += self.gender.value

        components = [base]
        if self.number == Number.PLURAL:
            components.append("pl")

        if self.phrasal == Phrasality.PHRASAL:
            components.append("phr")

        return " ".join(components)

    def merge(self, other: PartOfSpeech) -> Self:
        if self != other:
            raise ValueError(f"Cannot merge {self} and {other}")

        assert isinstance(other, Noun)
        gender = self.gender if self.gender != Gender.UNSPECIFIED else other.gender
        number = self.number if self.number != Number.UNSPECIFIED else other.number
        phrasality = _merge_phrasality(self.phrasal, other.phrasal)

        return type(self)(gender=gender, number=number, phrasal=phrasality)


@attrs.frozen
class Verb(PartOfSpeech):
    """Represents types of verbs and verb phrases."""

    transitivity: Transitivity
    phrasal: Phrasality = Phrasality.UNSPECIFIED
    base_pos: BasePoS = attrs.field(
        init=False, eq=True, hash=True, repr=False, default=BasePoS.VERB
    )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PartOfSpeech):
            return NotImplemented

        if self.base_pos != other.base_pos:
            return False

        assert isinstance(other, Verb)

        if not _compare_characteristic(self.transitivity, other.transitivity):
            return False
        return True

    def __str__(self) -> str:
        base = "v"
        if self.transitivity != Transitivity.UNSPECIFIED:
            base += self.transitivity.value

        components = [base]
        if self.phrasal == Phrasality.PHRASAL:
            components.append("phr")

        return " ".join(components)

    def merge(self, other: PartOfSpeech) -> Self:
        if self != other:
            raise ValueError(f"Cannot merge {self} and {other}")

        assert isinstance(other, Verb)
        transitivity = (
            self.transitivity
            if self.transitivity != Transitivity.UNSPECIFIED
            else other.transitivity
        )
        phrasality = _merge_phrasality(self.phrasal, other.phrasal)
        return type(self)(transitivity=transitivity, phrasal=phrasality)


_BASE_POS_MAP: Mapping[str, BasePoS] = {
    x.value: x for x in BasePoS if x not in {BasePoS.VERB, BasePoS.NOUN}
}


@functools.cache
def base_part_of_speech(base_pos: BasePoS, /) -> PartOfSpeech:
    """Retrieve a PartOfSpeech stripped of any characteristics."""
    if base_pos == BasePoS.NOUN:
        return Noun(gender=Gender.UNSPECIFIED, number=Number.UNSPECIFIED)
    if base_pos == BasePoS.VERB:
        return Verb(transitivity=Transitivity.UNSPECIFIED)

    return PartOfSpeech(base_pos=base_pos)


@functools.cache
def parse_part_of_speech(s: str, /) -> PartOfSpeech:
    """Convert a string representation of a part of speech to a PartOfSpeech.

    The input strings tend to be somewhat ambiguous because some
    characteristics are specified only by their presence or absence. So for
    example, "nm" is consistent with both "Noun(Number.SINGULAR,
    Gender.MASCULINE, Phrasality.UNSPECIFIED)" and with "Noun(Number.SINGULAR,
    Gender.MASCULINE, Phrasality.NOT_PHRASAL)".  Technically it also doesn't
    specify that it is /not/ a plural noun, but the plurality of the noun tends
    to be represented correctly if the gender is also represented.

    The assumption will make is that if /any/ characteristics other than the
    base part of speech are specified, any characeristics specified by the
    absence of a marker will be considered to be affirmatively specified as
    the absent characteristic (i.e. "nm" would be "masculine singular noun,
    not a phrase", whereas "n" would be "noun, unspecified gender, number and
    phrasality").
    """
    if s in _BASE_POS_MAP:
        return PartOfSpeech(base_pos=_BASE_POS_MAP[s])

    components = s.split(" ")
    base = components[0]
    if not base:
        raise PartOfSpeechParseError(f"Empty part of speech: {s!r}")

    if base[0] == "n":
        if s == "n":
            # In this case we are assuming bare noun with no
            # specified characteristics
            return Noun(number=Number.UNSPECIFIED, gender=Gender.UNSPECIFIED)

        try:
            gender = Gender.from_str(base[1:])
        except EnumParseError as e:
            raise PartOfSpeechParseError(f"Unknown noun spec: {s}") from e

        if len(components) == 1:
            if gender != Gender.UNSPECIFIED:
                phrasality = Phrasality.NON_PHRASAL
                number = Number.SINGULAR
            else:
                phrasality = Phrasality.UNSPECIFIED
                number = Number.UNSPECIFIED
        elif len(components) == 2:
            if components[1] == "pl":
                number = Number.PLURAL
                phrasality = Phrasality.NON_PHRASAL
            elif components[1] == "phr":
                number = Number.SINGULAR
                phrasality = Phrasality.PHRASAL
            else:
                raise PartOfSpeechParseError(f"Unknown noun spec: {s}")
        elif len(components) == 3:
            plural_s, phrasal_s = components[1:]
            if plural_s != "pl" and phrasal_s != "phr":
                raise PartOfSpeechParseError(f"Unknown noun spec: {s}")
            number = Number.PLURAL
            phrasality = Phrasality.PHRASAL

        if (
            gender != Gender.UNSPECIFIED
            and phrasality == Phrasality.UNSPECIFIED
            and number == Number.UNSPECIFIED
        ):
            phrasality = Phrasality.NON_PHRASAL
            number = Number.SINGULAR
        return Noun(number=number, gender=gender, phrasal=phrasality)
    elif base[0] == "v":
        if s == "v":
            return Verb(
                transitivity=Transitivity.UNSPECIFIED, phrasal=Phrasality.UNSPECIFIED
            )

        try:
            transitivity = Transitivity.from_str(base[1:])
        except EnumParseError as e:
            raise PartOfSpeechParseError(f"Unknown verb spec: {s}") from e

        if len(components) == 2:
            if components[1] != "phr":
                raise PartOfSpeechParseError(f"Unknown verb spec: {s}")
            phrasality = Phrasality.PHRASAL
        elif len(components) > 2:
            raise PartOfSpeechParseError(f"Unknown verb spec: {s}")
        else:
            phrasality = Phrasality.NON_PHRASAL

        return Verb(transitivity=transitivity, phrasal=phrasality)
    else:
        # TODO: Allow manual adjustment of bad parts of speech
        if base == "adj" and components[1] == "mf":
            components.pop()

        if len(components) == 2:
            if components[1] != "phr":
                raise PartOfSpeechParseError(f"Unknown part of speech: {s}")
            phrasality = Phrasality.PHRASAL
        else:
            phrasality = Phrasality.UNSPECIFIED

        try:
            base_pos = _BASE_POS_MAP[base]
        except KeyError:
            raise PartOfSpeechParseError(f"Unknown part of speech: {s}") from None
        return PartOfSpeech(base_pos=base_pos, phrasal=phrasality)


class ParseError(ValueError):
    """Raised when part of speech parsing fails."""


class PartOfSpeechParseError(ParseError):
    """Raised when failing to parse the entire part of speech."""


class EnumParseError(ParseError):
    """Raised when failing to parse a specific enum."""
