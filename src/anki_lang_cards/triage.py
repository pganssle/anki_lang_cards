"""Deciding what needs merging before any merging starts.

Resolving conflicts in the merge UI is slow, the set of conflicts is fixed when
the session opens, and nothing records the pairs that were looked at and
deliberately left alone. Every new duplicate-detection heuristic therefore
costs another pass through the UI re-answering settled questions, which is a
good reason not to add heuristics.

Triage moves the decision earlier. Every group that any signal flags is written
to a file with a reason and a suggested disposition; that file is edited by
hand; and only the groups still marked for review reach the merge UI. Pairs
that are marked as separate are remembered, so a noisy signal costs one skim
rather than a permanent tax.

Nothing here decides to hide a group. The suggested disposition is only ever
"merge" or "review" — "keep-separate" comes from a person, or from the record
of what a person previously said.
"""

import csv
import enum
import io
import itertools
import json
import pathlib
from collections.abc import Collection, Iterable, Mapping, Sequence, Set
from typing import Any, Final, Self

import attrs

from . import _types as types
from . import conflict_resolver, definition_similarity, headword
from . import part_of_speech as pos


class Disposition(enum.Enum):
    """What should happen to a group of words that might be duplicates."""

    MERGE = "merge"
    """Combine into one note. Goes to the merge UI pre-approved."""

    REVIEW = "review"
    """Undecided. Goes to the merge UI to be resolved by hand."""

    KEEP_SEPARATE = "keep-separate"
    """Deliberately distinct. Filtered out, and remembered so it stops coming back."""


class Reason(enum.Enum):
    """Why a group was flagged, and how far the definitions can be trusted."""

    COMPATIBLE = "compatible"
    """Same headword and compatible parts of speech: the classic conflict."""

    CHARACTERISTICS = "characteristics"
    """Same headword and base part of speech, disagreeing on the details.

    "nm/f" against "nf", or "vi" against "vtr phr". Almost always the model
    labelling one word two ways between runs, and the definitions are a good
    guide to whether it is.
    """

    GENDER = "gender"
    """Same noun, two different genders, both stated.

    "el cura"/"la cura" is two cards, and the gender is what tells them apart
    on the front. Definitions cannot be trusted to settle this — "el orden"
    (sequence) and "la orden" (religious order) both gloss as "order" — so
    these are only ever listed, never proposed.
    """

    BASE_PART_OF_SPEECH = "base-part-of-speech"
    """Same headword, different base part of speech.

    English collapses many of these: "segundo" the adjective and "segundo" the
    noun are both "second", and "general" is "general" either way. The
    definitions therefore agree perfectly on pairs that must stay apart, so
    these are listed but never proposed.
    """


_ATOMIC_GENDERS: Final[frozenset[pos.Gender]] = frozenset(
    {pos.Gender.MASC, pos.Gender.FEM}
)

_PROPOSABLE: Final[frozenset[Reason]] = frozenset(
    {Reason.COMPATIBLE, Reason.CHARACTERISTICS}
)


def _classify(first: Sequence[types.Word], second: Sequence[types.Word]) -> Reason:
    """Work out how two groups of the same headword differ."""
    left = first[0].part_of_speech
    right = second[0].part_of_speech
    if left.base_pos != right.base_pos:
        return Reason.BASE_PART_OF_SPEECH
    if (
        isinstance(left, pos.Noun)
        and isinstance(right, pos.Noun)
        and left.gender in _ATOMIC_GENDERS
        and right.gender in _ATOMIC_GENDERS
        and left.gender != right.gender
    ):
        return Reason.GENDER
    return Reason.CHARACTERISTICS


@attrs.frozen
class Candidate:
    """A group of words that might belong on one note."""

    words: tuple[types.Word, ...]
    """The notes as they exist now. Never rewritten by editing."""

    reason: Reason
    comparison: definition_similarity.Comparison
    disposition: Disposition

    revisions: tuple[types.Word, ...] | None = None
    """Rewritten notes, when the answer was to tell the two apart instead.

    Kept beside the originals rather than replacing them, because the two are
    needed for different things: decisions and the rejection record are keyed on
    the notes as they exist in the deck, while the revisions are what has to be
    imported to change them.
    """

    @property
    def key(self) -> tuple[types.Word, ...]:
        return tuple(sorted(self.words))

    @property
    def current_words(self) -> tuple[types.Word, ...]:
        """The notes as the reviewer left them."""
        return self.words if self.revisions is None else self.revisions

    @property
    def is_revised(self) -> bool:
        return self.revisions is not None

    def summarize(self, *, revised: bool = True) -> Sequence[str]:
        """Render the group the way it appears in the triage file."""
        words = self.current_words if revised else self.words
        return tuple(
            f"{word.rank if word.rank is not None else '-'} "
            f"{word.word} ({word.part_of_speech}) {word.meaning}"
            for word in sorted(words)
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "disposition": self.disposition.value,
            "reason": self.reason.value,
            "score": round(self.comparison.score, 3),
            "shared_glosses": sorted(self.comparison.shared_glosses),
            "summary": list(self.summarize()),
            "words": _words_to_csv(self.words),
        }
        if self.revisions is not None:
            out["was"] = list(self.summarize(revised=False))
            out["revisions"] = _words_to_csv(self.revisions)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        revisions = data.get("revisions")
        return cls(
            words=tuple(_words_from_csv(data["words"])),
            reason=Reason(data["reason"]),
            comparison=definition_similarity.Comparison(
                score=float(data["score"]),
                shared_glosses=frozenset(data.get("shared_glosses", ())),
            ),
            disposition=Disposition(data["disposition"]),
            revisions=None if revisions is None else tuple(_words_from_csv(revisions)),
        )


class RejectionCache:
    """Remembers the groups that were looked at and deliberately left apart.

    This is the counterpart to `MergeCache`, which only records the merges that
    were made. Without it, every loosened heuristic re-asks about the same
    deliberate sense splits on every run, which is what makes loosening
    expensive.
    """

    def __init__(self, cache_path: pathlib.Path):
        self.cache_path: Final = cache_path
        self._rejected: set[tuple[types.Word, ...]] = set()
        self._load()

    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        with self.cache_path.open() as f:
            data = json.load(f)
        self._rejected = {tuple(_words_from_csv(entry)) for entry in data}

    def save(self) -> None:
        if not self.cache_path.parent.exists():
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps([_words_to_csv(words) for words in sorted(self._rejected)])
        )

    def reject(self, words: Collection[types.Word]) -> None:
        self._rejected.add(tuple(sorted(words)))

    def __contains__(self, words: Collection[types.Word]) -> bool:
        return tuple(sorted(words)) in self._rejected

    def __len__(self) -> int:
        return len(self._rejected)


def _loose_index(
    lookup: conflict_resolver.WordLookup,
) -> Mapping[str, Sequence[Sequence[types.Word]]]:
    """Index the groups in a lookup by headword alone, ignoring part of speech.

    Groups sharing a headword but not a part of speech never meet inside a
    `WordLookup`, and they are exactly the pairs worth a second look.
    """
    index: dict[str, list[Sequence[types.Word]]] = {}
    for group in lookup:
        key = headword.lookup_key(next(iter(group)).word)
        index.setdefault(key, []).append(tuple(group))
    return index


def find_candidates(
    words: Iterable[types.Word],
    comparer: definition_similarity.DefinitionComparer,
    *,
    rejections: RejectionCache | None = None,
    focus: Collection[types.Word] | None = None,
) -> Sequence[Candidate]:
    """Collect every group that might need merging, with a suggested outcome.

    Two signals feed this. Words that a `WordLookup` already groups are
    conflicts in the existing sense. Words that share a headword but land in
    different groups are flagged as well, because a part of speech that
    disagrees is as often the model being inconsistent as it is a real
    distinction — and the definitions say which.

    `focus` narrows the result to groups touching at least one of the given
    words. Passing the words being added keeps a run about the import at hand:
    two notes that have sat in the deck disagreeing for a year are a real
    finding, but not one that this import has anything to do with. Pass None to
    look at everything.
    """
    lookup = conflict_resolver.WordLookup(words)
    in_focus = None if focus is None else frozenset(focus)
    candidates: list[Candidate] = []

    for groups in _loose_index(lookup).values():
        for group in groups:
            if len(group) > 1:
                candidates.append(_make_candidate(group, Reason.COMPATIBLE, comparer))
        for first, second in itertools.combinations(groups, 2):
            candidates.append(
                _make_candidate(
                    tuple(itertools.chain(first, second)),
                    _classify(first, second),
                    comparer,
                )
            )

    if in_focus is not None:
        candidates = [
            candidate
            for candidate in candidates
            if any(word in in_focus for word in candidate.words)
        ]
    if rejections is not None:
        candidates = [c for c in candidates if c.words not in rejections]
    return tuple(sorted(candidates, key=_ordering))


def _make_candidate(
    words: Sequence[types.Word],
    reason: Reason,
    comparer: definition_similarity.DefinitionComparer,
) -> Candidate:
    comparison = comparer.compare_all([word.meaning for word in words])
    proposable = reason in _PROPOSABLE and comparison.is_strong
    disposition = Disposition.MERGE if proposable else Disposition.REVIEW
    return Candidate(
        words=tuple(words),
        reason=reason,
        comparison=comparison,
        disposition=disposition,
    )


_DISPOSITION_ORDER: Final[Mapping[Disposition, int]] = {
    Disposition.REVIEW: 0,
    Disposition.MERGE: 1,
    Disposition.KEEP_SEPARATE: 2,
}


def _ordering(candidate: Candidate) -> tuple[int, float, str]:
    """Order the file the way it will be read.

    Undecided groups come first, because they are the ones needing an answer;
    within each disposition the strongest agreement leads, so the most clear-cut
    cases are dealt with before the marginal ones.
    """
    return (
        _DISPOSITION_ORDER[candidate.disposition],
        -candidate.comparison.score,
        candidate.summarize()[0],
    )


def write_triage_file(path: pathlib.Path, candidates: Iterable[Candidate]) -> None:
    """Write candidates out for a human to edit."""
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    payload = [candidate.to_dict() for candidate in candidates]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def read_triage_file(path: pathlib.Path) -> Sequence[Candidate]:
    """Read back a triage file, honoring any dispositions that were edited."""
    with path.open() as f:
        data = json.load(f)
    return tuple(Candidate.from_dict(entry) for entry in data)


def partition(
    candidates: Iterable[Candidate],
) -> Mapping[Disposition, Sequence[Candidate]]:
    """Split candidates by what should happen to them."""
    out: dict[Disposition, list[Candidate]] = {
        disposition: [] for disposition in Disposition
    }
    for candidate in candidates:
        out[candidate.disposition].append(candidate)
    return {disposition: tuple(group) for disposition, group in out.items()}


def record_rejections(
    candidates: Iterable[Candidate], rejections: RejectionCache
) -> None:
    """Remember every group marked as deliberately separate."""
    for candidate in candidates:
        if candidate.disposition is Disposition.KEEP_SEPARATE:
            rejections.reject(candidate.words)


def collect_revisions(candidates: Iterable[Candidate]) -> Sequence[types.Word]:
    """Gather every note that was rewritten, ready to be imported.

    Only notes that actually changed are returned, so a group opened in the
    editor and left alone contributes nothing.
    """
    revised: list[types.Word] = []
    for candidate in candidates:
        if candidate.revisions is None:
            continue
        originals = {word.rank: word for word in candidate.words}
        for word in candidate.revisions:
            if originals.get(word.rank) != word:
                revised.append(word)
    return tuple(revised)


def to_merge_sets(candidates: Iterable[Candidate]) -> Sequence[Set[types.Word]]:
    """Convert the groups still needing a decision into merge-UI input.

    The original notes are used, not the revised ones: the merge UI works
    against what is in the deck, and revisions reach the deck by being imported.
    """
    return tuple(
        frozenset(candidate.words)
        for candidate in candidates
        if candidate.disposition is not Disposition.KEEP_SEPARATE
    )


def _words_to_csv(words: Iterable[types.Word]) -> str:
    buffer = io.StringIO()
    csv.writer(buffer).writerows(word.to_csv_line() for word in words)
    buffer.seek(0)
    return buffer.read()


def _words_from_csv(text: str) -> Sequence[types.Word]:
    buffer = io.StringIO()
    csv.writer(buffer).writerow(types.Word.make_csv_header())
    buffer.seek(0)
    return types.Word.words_from_csv("\n".join((buffer.read().strip(), text)))
