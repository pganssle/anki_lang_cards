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

    SPLIT = "split"
    """One group becomes several new cards, replacing every note in it."""


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

    sides: tuple[tuple[types.Word, ...], ...] = ()
    """The groups being asked about, when the question is between two of them.

    A cross-part-of-speech candidate asks whether one group joins another, not
    whether every note in it is the same card. "deber" raises both questions at
    once — the two noun notes against each other, and the noun against the verb
    — and they are separate entries here. Keeping the sides lets the UI say so,
    rather than showing three notes flat and implying one verdict covers all of
    them. Empty means the whole group is one side.
    """

    replacements: tuple[types.Word, ...] | None = None
    """The notes this group becomes, when an answer said so.

    The shape depends on the disposition: a merge is the single combined note,
    wording included; keep-separate is the group rewritten note-for-note (in
    sorted order) to tell the members apart; a split is however many new cards
    the group was divided into. Kept beside the originals rather than replacing
    them, because the two are needed for different things: decisions and the
    rejection record are keyed on the notes as they exist in the deck, while
    the replacements are what has to be imported to change them.
    """

    @property
    def key(self) -> tuple[types.Word, ...]:
        return tuple(sorted(self.words))

    @property
    def current_sides(self) -> tuple[tuple[types.Word, ...], ...]:
        """The sides, with any rewriting applied, or the whole group as one."""
        if not self.sides:
            return (self.current_words,)
        if self.replacements is None:
            return self.sides
        revised = {word.rank: word for word in self.replacements}
        return tuple(
            tuple(revised.get(word.rank, word) for word in side) for side in self.sides
        )

    @property
    def current_words(self) -> tuple[types.Word, ...]:
        """The notes as the reviewer left them."""
        return self.words if self.replacements is None else self.replacements

    @property
    def has_replacements(self) -> bool:
        return self.replacements is not None

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
        if self.sides:
            out["sides"] = [_words_to_csv(side) for side in self.sides]
        if self.replacements is not None:
            out["was"] = list(self.summarize(revised=False))
            out["replacements"] = _words_to_csv(self.replacements)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        # "revisions" is the name replacements carried before splits existed.
        replacements = data.get("replacements", data.get("revisions"))
        return cls(
            words=tuple(_words_from_csv(data["words"])),
            reason=Reason(data["reason"]),
            comparison=definition_similarity.Comparison(
                score=float(data["score"]),
                shared_glosses=frozenset(data.get("shared_glosses", ())),
            ),
            disposition=Disposition(data["disposition"]),
            sides=tuple(tuple(_words_from_csv(side)) for side in data.get("sides", ())),
            replacements=(
                None if replacements is None else tuple(_words_from_csv(replacements))
            ),
        )


@attrs.frozen
class Decision:
    """What was decided about a group, and the notes it becomes.

    `replacements` follows the disposition the way `Candidate.replacements`
    does: one merged note for a merge, the group rewritten note-for-note for
    keep-separate, the new cards for a split. A merge with no stored note is
    legal — decisions recorded before merges carried their wording left the
    combined text in the separate merge cache — and callers fall back
    accordingly.
    """

    disposition: Disposition
    replacements: tuple[types.Word, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"disposition": self.disposition.value}
        if self.replacements is not None:
            out["replacements"] = _words_to_csv(self.replacements)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        # "revisions" is the name replacements carried before splits existed.
        replacements = data.get("replacements", data.get("revisions"))
        return cls(
            disposition=Disposition(data["disposition"]),
            replacements=(
                None if replacements is None else tuple(_words_from_csv(replacements))
            ),
        )


class DecisionCache:
    """Remembers what was decided about a group, so it is only asked once.

    `MergeCache` records the merges the merge UI performed, which is a
    different thing: this records the answer to "should these be one note at
    all", including the answer "no". Without it every run re-asks about
    everything not yet acted on, and a heuristic that flags a few extra groups
    costs a permanent tax rather than one skim.

    Groups are keyed on the notes as they exist in the deck, so a group that
    was rewritten is still recognized until the rewrite is imported — at which
    point the group no longer arises.
    """

    def __init__(self, cache_path: pathlib.Path):
        self.cache_path: Final = cache_path
        self._decisions: dict[tuple[types.Word, ...], Decision] = {}
        self._load()

    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        with self.cache_path.open() as f:
            data = json.load(f)
        for entry in data:
            # Earlier versions stored only the groups to keep separate, as bare
            # CSV strings.
            if isinstance(entry, str):
                self._decisions[tuple(_words_from_csv(entry))] = Decision(
                    disposition=Disposition.KEEP_SEPARATE
                )
                continue
            key = tuple(_words_from_csv(entry["words"]))
            self._decisions[key] = Decision.from_dict(entry)

    def save(self) -> None:
        if not self.cache_path.parent.exists():
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"words": _words_to_csv(words), **decision.to_dict()}
            for words, decision in sorted(self._decisions.items())
        ]
        self.cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def remember(self, candidate: Candidate) -> None:
        self._decisions[candidate.key] = Decision(
            disposition=candidate.disposition, replacements=candidate.replacements
        )

    def get(self, words: Collection[types.Word]) -> Decision | None:
        return self._decisions.get(tuple(sorted(words)))

    def apply(self, candidate: Candidate) -> Candidate | None:
        """Restore what was decided, or drop the group if it was settled apart."""
        decision = self.get(candidate.words)
        if decision is None:
            return candidate
        if decision.disposition is Disposition.KEEP_SEPARATE:
            return None
        return attrs.evolve(
            candidate,
            disposition=decision.disposition,
            replacements=(
                candidate.replacements
                if decision.replacements is None
                else decision.replacements
            ),
        )

    def values(self) -> Sequence[Decision]:
        """Every decision recorded, for callers that need to act on them."""
        return tuple(self._decisions.values())

    def entries(self) -> Sequence[tuple[tuple[types.Word, ...], Decision]]:
        """Every decision with the group it was about."""
        return tuple(self._decisions.items())

    def __contains__(self, words: Collection[types.Word]) -> bool:
        return self.get(words) is not None

    def __len__(self) -> int:
        return len(self._decisions)


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
    decisions: DecisionCache | None = None,
    focus: Collection[types.Word] | None = None,
) -> Sequence[Candidate]:
    """Collect every group that might need merging, with a suggested outcome.

    Two signals feed this. Words that a `WordLookup` already groups are
    conflicts in the existing sense. Words that share a headword but land in
    different groups are flagged as well, because a part of speech that
    disagrees is as often the model being inconsistent as it is a real
    distinction — and the definitions say which.

    `decisions` restores what was previously answered about a group and drops
    the ones settled as separate, so nothing is asked twice.

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
                    sides=(tuple(first), tuple(second)),
                )
            )

    if in_focus is not None:
        candidates = [
            candidate
            for candidate in candidates
            if any(word in in_focus for word in candidate.words)
        ]
    if decisions is not None:
        candidates = [
            restored
            for candidate in candidates
            if (restored := decisions.apply(candidate)) is not None
        ]
    return tuple(sorted(candidates, key=_ordering))


def _make_candidate(
    words: Sequence[types.Word],
    reason: Reason,
    comparer: definition_similarity.DefinitionComparer,
    sides: tuple[tuple[types.Word, ...], ...] = (),
) -> Candidate:
    comparison = comparer.compare_all([word.meaning for word in words])
    proposable = reason in _PROPOSABLE and comparison.is_strong
    disposition = Disposition.MERGE if proposable else Disposition.REVIEW
    return Candidate(
        words=tuple(words),
        reason=reason,
        comparison=comparison,
        disposition=disposition,
        sides=sides,
    )


_DISPOSITION_ORDER: Final[Mapping[Disposition, int]] = {
    Disposition.REVIEW: 0,
    Disposition.MERGE: 1,
    Disposition.SPLIT: 2,
    Disposition.KEEP_SEPARATE: 3,
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


def record_decisions(candidates: Iterable[Candidate], decisions: DecisionCache) -> int:
    """Remember every group that was answered, and return how many were new.

    A group still marked for review is not an answer, so it is not remembered
    unless it was rewritten — an edit is work worth keeping either way.
    """
    recorded = 0
    for candidate in candidates:
        answered = candidate.disposition is not Disposition.REVIEW
        if not (answered or candidate.has_replacements):
            continue
        if decisions.get(candidate.words) is None:
            recorded += 1
        decisions.remember(candidate)
    return recorded


def collect_revisions(candidates: Iterable[Candidate]) -> Sequence[types.Word]:
    """Gather every note that was rewritten, ready to be imported.

    Only note-for-note rewrites count here — a merge or split replaces the
    group rather than editing it, and the length guard keeps those shapes out.
    Only notes that actually changed are returned, so a group opened in the
    editor and left alone contributes nothing.
    """
    revised: list[types.Word] = []
    for candidate in candidates:
        if candidate.replacements is None:
            continue
        if candidate.disposition is Disposition.SPLIT:
            # Split cards are not paired with the notes they came from, even
            # when the counts happen to line up.
            continue
        ordered = sorted(candidate.words)
        if len(ordered) != len(candidate.replacements):
            continue
        # Paired by position, not by rank: a group can hold two notes at one
        # rank, and matching on it writes one note's text over the other.
        for original, rewritten in zip(ordered, candidate.replacements, strict=True):
            if original != rewritten:
                revised.append(rewritten)
    return tuple(revised)


def to_merge_sets(candidates: Iterable[Candidate]) -> Sequence[Set[types.Word]]:
    """Convert the groups still needing a decision into merge-UI input.

    The original notes are used, not the revised ones: the merge UI works
    against what is in the deck, and revisions reach the deck by being imported.
    """
    return tuple(
        frozenset(candidate.words)
        for candidate in candidates
        if candidate.disposition in (Disposition.MERGE, Disposition.REVIEW)
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
