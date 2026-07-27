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

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Final

import attrs

from . import _types as types
from . import conflict_resolver, definition_similarity, triage

DEFAULT_MAX_ROUNDS: Final[int] = 10
"""Enough for any real chain; a cap only so a cycle cannot spin forever."""

Resolver = Callable[[Sequence[Sequence[types.Word]]], Sequence[types.Word | None]]
"""Decides what a merged note says, given the groups to combine.

Deciding that two notes are one card does not decide what the card reads. The
definitions still have to be reconciled, and that is a judgement — combining
them mechanically turns "piece of clothing" and "piece of glothing, garment"
into all three at once. `word_merger.merge_words` is the resolver in normal use;
returning None for a group leaves it unmerged.
"""


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

    declined: tuple[triage.Candidate, ...] = ()
    """Groups the resolver was asked about and left alone."""

    rounds: int = 0

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

    A decision's revisions are parallel to its group in sorted order — that is
    how the editor builds them — so they are paired by position. Matching them
    up by rank instead cross-wires a group whose notes share one: "prenda"
    exists twice at rank 3467, once as the deck holds it and once as the import
    revises it, and rank-matching wrote the deck's text over the import's,
    losing the corrected definition and all eight of its new sentences.

    Only pairs that actually differ count as a rewrite, so opening the editor
    and saving changes nothing.
    """
    replacements: list[tuple[types.Word, types.Word]] = []
    for group, decision in decisions.entries():
        if decision.revisions is None:
            continue
        ordered = sorted(group)
        if len(ordered) != len(decision.revisions):
            # Shapes disagree, so position means nothing; leave the group alone
            # rather than guess which note each revision belongs to.
            continue
        for original, revised in zip(ordered, decision.revisions, strict=True):
            if original != revised:
                replacements.append((original, revised))

    out: list[types.Word] = []
    applied: list[types.Word] = []
    for word in words:
        replacement = next(
            (
                rev
                for orig, rev in replacements
                if orig == word and orig.rank == word.rank
            ),
            None,
        )
        if replacement is not None:
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
    resolver: Resolver | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> Outcome:
    """Rewrite and combine notes according to what triage decided.

    Runs until nothing more changes. Each round merges the groups that were
    answered "merge" and the groups whose members are now identical, then looks
    again, because a merge can bring a note into contact with one it did not
    previously touch.

    Groups whose notes are identical are combined outright: there is only one
    thing they can say. Every other group goes to `resolver` to have its wording
    settled, and without one they are combined mechanically, which is only
    appropriate when nobody is going to read the result.
    """
    by_rank = _decisions_by_rank(decisions)
    current, revised = _apply_revisions(tuple(words), decisions)
    merges: list[Merge] = []
    declined: list[triage.Candidate] = []
    unresolved: tuple[triage.Candidate, ...] = ()
    rounds = 0

    while rounds < max_rounds:
        rounds += 1
        candidates = triage.find_candidates(current, comparer)
        consumed: set[int] = set()
        round_merges: list[Merge] = []
        leftover: list[triage.Candidate] = []

        pending: list[
            tuple[triage.Candidate, Sequence[types.Word], tuple[str, ...]]
        ] = []

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
            identical = _identical(candidate.words)
            if not decided and not identical:
                if decision is None:
                    leftover.append(candidate)
                continue

            aligned, dropped = _align_headwords(candidate.words)
            consumed.update(id(word) for word in candidate.words)
            if identical:
                round_merges.append(
                    Merge(
                        result=conflict_resolver.auto_merge_words(aligned),
                        sources=tuple(candidate.words),
                        decided=False,
                        dropped_headwords=dropped,
                    )
                )
            else:
                pending.append((candidate, aligned, dropped))

        if pending:
            groups = [aligned for _, aligned, _ in pending]
            if resolver is None:
                results: Sequence[types.Word | None] = [
                    conflict_resolver.auto_merge_words(group) for group in groups
                ]
            else:
                results = resolver(groups)
            for (candidate, _, dropped), result in zip(pending, results, strict=True):
                if result is None:
                    declined.append(candidate)
                    consumed.difference_update(id(w) for w in candidate.words)
                    continue
                round_merges.append(
                    Merge(
                        result=result,
                        sources=tuple(candidate.words),
                        decided=True,
                        dropped_headwords=dropped,
                    )
                )

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
        declined=tuple(declined),
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


def discarded_imports(
    outcome: Outcome,
    inputs: Collection[types.Word],
    existing: Collection[types.Word],
) -> Sequence[types.Word]:
    """Imported notes whose content did not survive to the output.

    A merge is supposed to combine two notes, not pick one and drop the other.
    When the result comes out byte-identical to what the deck already had, the
    imported version — its new sentences, its corrected definition — went
    nowhere, and the run has quietly undone the work it was carrying out.

    This is a check on the result rather than a fix for any particular cause:
    whatever produced it, the answer is to look before importing.
    """

    def substance(word: types.Word) -> tuple:
        # What the import exists to contribute: a definition and examples. The
        # article and part of speech are excluded, since a merge picking those
        # up from one side is normal and not a sign anything was lost.
        return (word.meaning, tuple(s.spanish for s in word.sentences))

    deck = {word.rank: word for word in existing if word.rank is not None}
    imported = {word.rank: word for word in inputs if word.rank is not None}
    out: list[types.Word] = []
    for word in outcome.words:
        original = deck.get(word.rank)
        contributed = imported.get(word.rank)
        if original is None or contributed is None:
            continue
        # Only a finding if the import had something to say in the first place.
        if substance(contributed) == substance(original):
            continue
        if substance(word) == substance(original):
            out.append(word)
    return tuple(out)


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
    if outcome.declined:
        lines.append(f"{len(outcome.declined)} groups were left unmerged")
    if outcome.unresolved:
        lines.append(
            f"{len(outcome.unresolved)} new collisions need another triage pass"
        )
    return tuple(lines)
