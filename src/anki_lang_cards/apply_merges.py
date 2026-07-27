"""Carrying out the decisions made during triage.

Triage answers "should these be one note". This does it: rewrites the notes that
were edited, combines the groups that were answered "merge", and reports what
the result implies for the deck.

Merging is iterative rather than a single sweep, because a decision can create
work for the next one. Rewriting "metomentodo" from an adjective and a noun into
two identical notes is a sensible answer to the question that was asked, and it
leaves two notes that are now plainly one card. Nothing asked about that, and
nothing needs to: notes that come out identical are combined without a decision,
since there is nothing left to decide. Anything else that newly collides is
reported rather than merged, so a real question goes back through triage instead
of being answered here by default.
"""

from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Final

import attrs

from . import _types as types
from . import conflict_resolver, definition_similarity, triage

DEFAULT_MAX_ROUNDS: Final[int] = 10
"""Enough for any real chain; a cap only so a cycle cannot spin forever."""


@attrs.frozen
class Merge:
    """One group combined into a single note."""

    result: types.Word
    sources: tuple[types.Word, ...]
    decided: bool
    """Whether a person answered "merge", as opposed to the notes being identical."""

    dropped_headwords: tuple[str, ...] = ()
    """Spellings the group disagreed on that the merge did not keep."""

    @property
    def superseded(self) -> tuple[types.Word, ...]:
        """The notes the merge replaces, other than the one whose rank survives."""
        return tuple(word for word in self.sources if word.rank != self.result.rank)


@attrs.frozen
class Outcome:
    """The result of carrying out a set of decisions."""

    words: tuple[types.Word, ...]
    merges: tuple[Merge, ...]
    revised: tuple[types.Word, ...]
    unresolved: tuple[triage.Candidate, ...]
    """Collisions that appeared along the way and were not covered by a decision."""

    rounds: int

    def touched_ranks(self) -> frozenset[int]:
        """Ranks whose note content changed, following merges through.

        A merge of a deck note with a new one keeps the deck note's rank, so the
        rank that has to be written is not necessarily one that was in the
        input. Following the chain finds it.
        """
        touched = {word.rank for word in self.revised if word.rank is not None}
        for merge in self.merges:
            ranks = {
                w.rank for w in (merge.result, *merge.sources) if w.rank is not None
            }
            touched |= ranks
        return frozenset(touched)


def _decisions_by_rank(
    decisions: triage.DecisionCache,
) -> Mapping[frozenset[int], triage.Decision]:
    """Index decisions by the notes they were about, rather than their contents.

    `DecisionCache` keys on the exact notes, which is right for triage: if the
    text changed, the question genuinely is a new one and should be asked again.
    It is wrong here. "Merge 8576 with 9615" is an instruction about two notes,
    and rewriting either of them first — which is a thing the same session
    encourages — must not quietly discard it.
    """
    out: dict[frozenset[int], triage.Decision] = {}
    for words, decision in decisions.entries():
        ranks = frozenset(w.rank for w in words if w.rank is not None)
        if ranks:
            out[ranks] = decision
    return out


def _apply_revisions(
    words: Sequence[types.Word], decisions: triage.DecisionCache
) -> tuple[list[types.Word], list[types.Word]]:
    """Replace notes with the versions they were rewritten to.

    Revisions are paired with the notes they came from inside their own
    decision, and only the ones that actually changed something are kept. Two
    notes in a group can share a rank — the deck's copy and the imported copy of
    one note — so a map keyed on rank alone would let one silently displace the
    other, which is how a typo the rewrite fixed came back.
    """
    replacements: dict[tuple[int, str, str], types.Word] = {}
    for group, decision in decisions.entries():
        if decision.revisions is None:
            continue
        originals = {w.rank: [] for w in group if w.rank is not None}
        for word in group:
            if word.rank is not None:
                originals[word.rank].append(word)
        for revised in decision.revisions:
            if revised.rank is None:
                continue
            for original in originals.get(revised.rank, ()):
                if original != revised:
                    key = (original.rank, original.word, original.meaning)
                    replacements[key] = revised

    out: list[types.Word] = []
    applied: list[types.Word] = []
    for word in words:
        key = (word.rank, word.word, word.meaning)
        replacement = replacements.get(key) if word.rank is not None else None
        if replacement is not None and replacement != word:
            # Keep the sentences: a rewrite is about the headword and the
            # definition, and the examples were never in the editor.
            merged = attrs.evolve(
                replacement, sentences=replacement.sentences or word.sentences
            )
            out.append(merged)
            applied.append(merged)
        else:
            out.append(word)
    return out, applied


def _align_headwords(
    group: Sequence[types.Word],
) -> tuple[Sequence[types.Word], tuple[str, ...]]:
    """Settle on one spelling for a group that is about to be merged.

    A group can hold two spellings of one headword — "seguir el rollo" beside
    "seguir el rollo (a alguien)" — because the lookup key ignores a valency
    marker while `Word.__eq__` does not. `auto_merge_words` reconciles meanings
    and parts of speech but not the headword, and merging without settling it
    raises.

    The lowest-ranked note wins, that being the one already in the deck and
    therefore the spelling on the card being reviewed. The discarded spellings
    are reported rather than dropped silently, since preferring one is a
    judgement and the reader may want the other.
    """
    ranked = sorted(group, key=lambda w: w.rank if w.rank is not None else 0)
    keeper = ranked[0].word
    dropped = tuple(dict.fromkeys(w.word for w in ranked if w.word != keeper))
    if not dropped:
        return group, ()
    return [attrs.evolve(word, word=keeper) for word in group], dropped


def _identical(group: Sequence[types.Word]) -> bool:
    """Whether a group has nothing left to decide."""
    first = group[0]
    return all(word == first for word in group[1:])


def apply_decisions(
    words: Iterable[types.Word],
    decisions: triage.DecisionCache,
    comparer: definition_similarity.DefinitionComparer,
    *,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> Outcome:
    """Rewrite and combine notes according to what triage decided.

    Runs until nothing more changes. Each round merges the groups that were
    answered "merge" and the groups whose members are now identical, then looks
    again, because a merge can bring a note into contact with one it did not
    previously touch.
    """
    by_rank = _decisions_by_rank(decisions)
    current, revised = _apply_revisions(tuple(words), decisions)
    merges: list[Merge] = []
    unresolved: tuple[triage.Candidate, ...] = ()
    rounds = 0

    while rounds < max_rounds:
        rounds += 1
        candidates = triage.find_candidates(current, comparer)
        consumed: set[int] = set()
        round_merges: list[Merge] = []
        leftover: list[triage.Candidate] = []

        for candidate in candidates:
            # A note can only be merged once per round. Groups overlap — the
            # same note appears in a within-group candidate and a between-group
            # one — and merging both in a round would use a stale copy.
            if any(id(word) in consumed for word in candidate.words):
                continue

            ranks = frozenset(w.rank for w in candidate.words if w.rank is not None)
            decision = by_rank.get(ranks) or decisions.get(candidate.words)
            decided = decision is not None and (
                decision.disposition is triage.Disposition.MERGE
            )
            if not decided and not _identical(candidate.words):
                if decision is None:
                    leftover.append(candidate)
                continue

            aligned, dropped = _align_headwords(candidate.words)
            round_merges.append(
                Merge(
                    result=conflict_resolver.auto_merge_words(aligned),
                    sources=tuple(candidate.words),
                    decided=decided,
                    dropped_headwords=dropped,
                )
            )
            consumed.update(id(word) for word in candidate.words)

        if not round_merges:
            unresolved = tuple(leftover)
            break

        current = [word for word in current if id(word) not in consumed]
        current.extend(merge.result for merge in round_merges)
        merges.extend(round_merges)

    return Outcome(
        words=tuple(current),
        merges=tuple(merges),
        revised=tuple(revised),
        unresolved=unresolved,
        rounds=rounds,
    )


def superseded_notes(
    outcome: Outcome, existing: Collection[types.Word]
) -> Sequence[types.Word]:
    """Deck notes a merge has replaced, which Anki will still be holding.

    Importing a merged note updates the rank it kept. The other ranks in the
    group are left behind as duplicates of what was just merged, and importing
    cannot remove them — that has to happen in Anki.
    """
    deck_ranks = {word.rank for word in existing if word.rank is not None}
    # One rank can be superseded by several source notes — the deck copy and the
    # imported copy of the same note both appear — but it is one note to delete.
    by_rank = {
        word.rank: word
        for merge in outcome.merges
        for word in merge.superseded
        if word.rank in deck_ranks
    }
    return tuple(by_rank[rank] for rank in sorted(by_rank))


def words_to_import(
    outcome: Outcome, inputs: Collection[types.Word]
) -> Sequence[types.Word]:
    """The notes to write out, being those the run actually changed.

    Everything else in the word set came from the deck and is unchanged, so
    writing it would be a no-op import of thousands of notes.
    """
    wanted = outcome.touched_ranks() | {
        word.rank for word in inputs if word.rank is not None
    }
    return tuple(
        sorted(
            (word for word in outcome.words if word.rank in wanted),
            key=lambda word: word.rank if word.rank is not None else 0,
        )
    )


def new_collisions(
    outcome: Outcome, inputs: Collection[types.Word]
) -> Sequence[triage.Candidate]:
    """Unresolved collisions that this import is responsible for.

    Merging is done over the whole word set, so the leftovers include groups
    that have sat in the deck disagreeing since long before this run. Those are
    real findings but not this import's business, and listing them here would
    bury the ones that are.
    """
    ranks = {word.rank for word in inputs if word.rank is not None}
    ranks |= outcome.touched_ranks()
    return tuple(
        candidate
        for candidate in outcome.unresolved
        if any(word.rank in ranks for word in candidate.words)
    )


def summarize(outcome: Outcome) -> Sequence[str]:
    """Lines describing what happened, for a person to read."""
    lines = [
        f"{len(outcome.revised)} notes rewritten",
        f"{len(outcome.merges)} groups merged over {outcome.rounds} rounds",
    ]
    automatic = [merge for merge in outcome.merges if not merge.decided]
    if automatic:
        lines.append(
            f"  {len(automatic)} of them became identical after rewriting and "
            "were combined without a decision"
        )
    renamed = [merge for merge in outcome.merges if merge.dropped_headwords]
    if renamed:
        lines.append(
            f"  {len(renamed)} kept one of two spellings; the other was dropped"
        )
    if outcome.unresolved:
        lines.append(
            f"{len(outcome.unresolved)} new collisions need another triage pass"
        )
    return tuple(lines)
