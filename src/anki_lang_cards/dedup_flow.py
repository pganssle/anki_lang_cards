"""The one flow that finds and settles duplicates.

Every road into deduplication — the main pipeline adding words, the cleanup
script preparing an import, the whole-deck duplicate hunt — runs through
`resolve_duplicates`: find the groups worth asking about, ask (on screen or
via the decision file), carry the answers out, and go around again when an
answer creates new work. What the caller does with the outcome differs; how
duplicates are found and settled does not.

Interruption is the normal case, not an error. Answers are persisted the
moment they are given, so a run that stops — quit, crash, or a headless run
hitting a question nobody has answered — reports what remains and resumes
where it left off next time.

This module does not require textual: the UI is imported only when a run is
interactive. Headless callers get the same flow with the pending questions
returned for the decision-file protocol instead of a screen.
"""

import json
import pathlib
import sys
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, Final

import attrs

from . import _types as types
from . import apply_merges, conflict_resolver, definition_similarity, triage

DEFAULT_MAX_PASSES: Final[int] = 3
"""Answer/apply passes per run. A merge rarely creates more than one round of
new questions; anything left after three is reported rather than chased."""


@attrs.frozen
class DedupResult:
    """What a run of the flow settled, and what it could not."""

    outcome: apply_merges.Outcome
    """The final word set with every answered decision carried out."""

    pending: tuple[triage.Candidate, ...] = ()
    """Groups still needing an answer, ready for the decision file."""

    rank_only_decisions: tuple[tuple[types.Word, ...], ...] = ()
    """Recorded groups matched by rank while their notes' text has changed.

    The stale-cache signature: a decision made about one version of the notes
    being applied to another. Never fatal — the decision still names the right
    ranks — but worth a look before trusting the output.
    """

    passes: int = 0

    @property
    def interrupted(self) -> bool:
        """Whether the run stopped short of answering everything it found."""
        return bool(self.pending)


def _needs_answer(candidate: triage.Candidate, decisions: triage.DecisionCache) -> bool:
    """Whether a group still lacks a recorded answer.

    Identical notes have nothing to decide, and a recorded non-review answer
    settles the group. A recorded review — an edit saved before the group was
    decided — keeps the question open.

    Matching mirrors what carrying the answer out will do: by the exact notes
    first, then by their ranks. A decision recorded when the deck was loaded
    holds three notes — both copies of one plus the other — and a later run
    seeing only two must still recognize it as answered, exactly as
    `apply_decisions` will when it merges the group.

    A merge decision with no stored wording is half an answer. The old flow
    recorded dispositions and worded the merge elsewhere; when that wording
    cannot be found (see `migrate_cached_wordings`), what the card should say
    is still an open question — asked once, and recorded with its wording
    this time.
    """
    if apply_merges._identical(candidate.words):
        return False
    decision = decisions.get(candidate.words)
    if decision is None:
        ranks = frozenset(w.rank for w in candidate.words if w.rank is not None)
        decision = apply_merges._decisions_by_rank(decisions).get(ranks)
    if decision is None or decision.disposition is triage.Disposition.REVIEW:
        return True
    return (
        decision.disposition is triage.Disposition.MERGE
        and decision.replacements is None
    )


def _recorder(decisions: triage.DecisionCache) -> Callable[[triage.Candidate], None]:
    """The per-answer persistence hook: remember and save immediately.

    An undecided group is only worth recording once it carries an edit — the
    same rule as `triage.record_decisions` — but every real answer hits disk
    before the next question is shown.
    """

    def record(candidate: triage.Candidate) -> None:
        undecided = candidate.disposition is triage.Disposition.REVIEW
        if undecided and not candidate.has_replacements:
            return
        decisions.remember(candidate)
        decisions.save()

    return record


def make_cache_resolver(
    merge_cache: conflict_resolver.MergeCache | None,
) -> apply_merges.Resolver:
    """Wording from the legacy merge cache, and nothing else.

    Decisions recorded before merges carried their wording left the combined
    text in a separate cache keyed on the group. Replaying it here keeps that
    work; a group with no cached wording is declined and becomes a wording
    question for a person rather than being worded mechanically behind the
    reviewer's back.
    """

    def resolve(
        groups: Sequence[Sequence[types.Word]],
    ) -> Sequence[types.Word | None]:
        results: list[types.Word | None] = []
        for group in groups:
            cached = None if merge_cache is None else merge_cache.get_merge(group)
            results.append(None if cached is None else cached.result)
        return results

    return resolve


def _rank_only_matches(
    candidates: Sequence[triage.Candidate], decisions: triage.DecisionCache
) -> tuple[tuple[types.Word, ...], ...]:
    """Recorded groups that only still apply because the ranks line up."""
    current_ranks = {
        frozenset(w.rank for w in candidate.words if w.rank is not None): candidate
        for candidate in candidates
    }
    stale: list[tuple[types.Word, ...]] = []
    for group, _decision in decisions.entries():
        ranks = frozenset(w.rank for w in group if w.rank is not None)
        candidate = current_ranks.get(ranks)
        if candidate is None:
            continue
        if decisions.get(candidate.words) is None:
            stale.append(group)
    return tuple(stale)


def migrate_cached_wordings(
    candidates: Sequence[triage.Candidate],
    decisions: triage.DecisionCache,
    merge_cache: conflict_resolver.MergeCache | None,
) -> int:
    """Turn legacy merge-cache wordings into decisions, returning how many.

    Before decisions carried their outcome, an answered merge lived only in
    the merge-wording cache — the main pipeline's answers all do, and the old
    triage flow recorded dispositions with no wording at all. A group with
    cached wording was answered once already, and asking again is exactly the
    re-asking the caches exist to prevent, so the old record is promoted into
    a worded decision the first time the group comes up. Groups whose wording
    is nowhere to be found stay as they are, to be asked once.
    """
    if merge_cache is None:
        return 0
    migrated = 0
    for candidate in candidates:
        decision = decisions.get(candidate.words)
        if decision is not None and decision.replacements is not None:
            continue
        if (
            decision is not None
            and decision.disposition is not triage.Disposition.MERGE
        ):
            continue
        cached = merge_cache.get_merge(candidate.words)
        if cached is None:
            aligned, _ = apply_merges.align_headwords(sorted(candidate.words))
            cached = merge_cache.get_merge(aligned)
        if cached is None:
            continue
        decisions.remember(
            attrs.evolve(
                candidate,
                disposition=triage.Disposition.MERGE,
                replacements=(cached.result,),
            )
        )
        migrated += 1
    if migrated:
        decisions.save()
    return migrated


def _record_proposals(
    pending: Sequence[triage.Candidate], decisions: triage.DecisionCache
) -> None:
    """Accept every proposed merge mechanically, for unattended runs."""
    recorded = False
    for candidate in pending:
        if candidate.disposition is not triage.Disposition.MERGE:
            continue
        proposal = apply_merges.propose_merge(candidate.words)
        if proposal is None:
            continue
        decisions.remember(attrs.evolve(candidate, replacements=(proposal,)))
        recorded = True
    if recorded:
        decisions.save()


async def resolve_duplicates(
    inputs: Sequence[types.Word],
    existing: Sequence[types.Word],
    *,
    decisions: triage.DecisionCache,
    merge_cache: conflict_resolver.MergeCache | None = None,
    interactive: bool = True,
    auto: bool = False,
    per_page: int | None = None,
    max_passes: int = DEFAULT_MAX_PASSES,
) -> DedupResult:
    """Find duplicates among the words, settle them, and carry the answers out.

    `inputs` are the words this run is about — candidates are narrowed to
    groups touching them. An empty `inputs` means the run is about the whole
    word set, the whole-deck hunt.

    Interactive runs open the unified screen for anything unanswered, with
    every answer persisted as it is given; a merge whose result collides with
    something new sends the new question back to the screen in the same
    sitting, up to `max_passes`. `auto` answers mechanically instead — every
    proposed merge is accepted with its mechanical wording — and headless,
    non-auto runs answer nothing: they carry out what is already recorded and
    return the rest as `pending`, which is the decision-file protocol's job to
    get answered.
    """
    words: tuple[types.Word, ...] = (*existing, *inputs)
    focus: Collection[types.Word] | None = inputs if inputs else None
    comparer = definition_similarity.DefinitionComparer(word.meaning for word in words)
    resolver = None if auto else make_cache_resolver(merge_cache)

    candidates = triage.find_candidates(
        words, comparer, decisions=decisions, focus=focus
    )
    migrated = migrate_cached_wordings(candidates, decisions, merge_cache)
    if migrated:
        print(
            f"  {migrated} groups already answered in the merge-wording "
            "cache; recorded as decisions"
        )
    to_ask = [c for c in candidates if _needs_answer(c, decisions)]
    rank_only = _rank_only_matches(candidates, decisions)

    # Questions shown and left unanswered — by u, or by quitting. They stay
    # pending, but re-showing them in the same run would be nagging.
    carryover: dict[tuple[types.Word, ...], triage.Candidate] = {}

    passes = 0
    while True:
        passes += 1
        if to_ask and auto:
            _record_proposals(to_ask, decisions)
        elif to_ask and interactive:
            from . import triage_ui

            await triage_ui.resolve_candidates(
                to_ask,
                per_page=per_page,
                on_decision=_recorder(decisions),
            )
        # Whatever was up for answering and still is not answered — shown and
        # left, no proposal to record, or nobody to ask — stays pending.
        walked_away = False
        for candidate in to_ask:
            if _needs_answer(candidate, decisions):
                carryover.setdefault(candidate.key, candidate)
                walked_away = True

        outcome = apply_merges.apply_decisions(
            words, decisions, comparer, resolver=resolver
        )

        # New questions this pass created: collisions the merges surfaced,
        # plus groups answered "merge" whose wording nothing could supply.
        # Declined groups are pending even though they carry an answer — only
        # a screen or the decision file can word them.
        fresh = [
            c
            for c in apply_merges.new_collisions(outcome, inputs)
            if _needs_answer(c, decisions) and c.key not in carryover
        ]
        fresh.extend(c for c in outcome.declined if c.key not in carryover)
        answering = auto or interactive
        if not fresh or not answering or passes >= max_passes:
            break
        # A session that ended with questions unanswered was quit, or the
        # questions were deliberately deferred. Either way the reviewer is
        # done for this run: reopening the screen with the next batch reads
        # as a UI that refuses to close, so everything left goes to pending
        # instead.
        if interactive and walked_away:
            break
        # In auto mode a second pass only helps when new proposals appeared;
        # anything else would just spin.
        if auto and not any(c.disposition is triage.Disposition.MERGE for c in fresh):
            break
        to_ask = fresh

    pending = {c.key: c for c in fresh}
    pending.update(carryover)
    return DedupResult(
        outcome=outcome,
        pending=tuple(pending.values()),
        rank_only_decisions=rank_only,
        passes=passes,
    )


def write_pending_file(path: pathlib.Path, pending: Sequence[triage.Candidate]) -> None:
    """Write the unanswered groups for a person or a harness to answer.

    The same file `triage.write_triage_file` produces, with two additions per
    entry: `index`, so validation errors and conversations can name a
    candidate, and `proposed`, the merged note accepting would record — the
    same proposal the screen shows. Answering means editing `disposition`
    (and, when the wording or the split matters, `replacements`) and
    re-running; see `read_decision_file` for what is accepted.
    """
    payload = []
    for index, candidate in enumerate(pending):
        entry = dict(candidate.to_dict())
        entry["index"] = index
        proposal = apply_merges.propose_merge(candidate.current_words)
        if proposal is not None:
            entry["proposed"] = triage.words_to_csv([proposal])
        payload.append(entry)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


class DecisionFileError(ValueError):
    """A decision file that cannot be applied, with every problem named."""

    def __init__(self, problems: Sequence[str]):
        self.problems = tuple(problems)
        super().__init__(
            "The decision file has problems:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )


def _entry_label(entry: Mapping[str, Any], position: int) -> str:
    index = entry.get("index", position)
    summary = entry.get("summary") or ["?"]
    return f"candidate {index} ({summary[0]})"


def _apply_sentence_allocation(
    candidate: triage.Candidate, allocation: Sequence[str], label: str
) -> triage.Candidate:
    """Distribute the group's sentences over split cards by number.

    The file-side mirror of the split editor's allocation column: one entry
    per sentence of the group's union, each naming the card numbers that
    sentence belongs to, comma-separated to share it.
    """
    if candidate.replacements is None:
        raise DecisionFileError(
            [f"{label}: sentence_allocation needs replacements to allocate to"]
        )
    union: list[types.Sentence] = []
    seen: set[types.Sentence] = set()
    for word in sorted(candidate.words):
        for sentence in word.sentences:
            if sentence not in seen:
                seen.add(sentence)
                union.append(sentence)
    if len(allocation) != len(union):
        raise DecisionFileError(
            [
                f"{label}: sentence_allocation names {len(allocation)} "
                f"sentences but the group has {len(union)}"
            ]
        )
    count = len(candidate.replacements)
    per_card: list[list[types.Sentence]] = [[] for _ in range(count)]
    for position, (sentence, raw) in enumerate(zip(union, allocation, strict=True)):
        tokens = [token.strip() for token in str(raw).split(",") if token.strip()]
        if not tokens or not all(token.isdigit() for token in tokens):
            raise DecisionFileError(
                [
                    f"{label}: sentence_allocation[{position}] must be card "
                    f"numbers like '1' or '1,2', not {raw!r}"
                ]
            )
        numbers = {int(token) for token in tokens}
        if not all(1 <= number <= count for number in numbers):
            raise DecisionFileError(
                [
                    f"{label}: sentence_allocation[{position}] names a card "
                    f"outside 1–{count}"
                ]
            )
        for number in sorted(numbers):
            per_card[number - 1].append(sentence)
    return attrs.evolve(
        candidate,
        replacements=tuple(
            attrs.evolve(card, sentences=tuple(sentences))
            for card, sentences in zip(candidate.replacements, per_card, strict=True)
        ),
    )


def read_decision_file(path: pathlib.Path) -> Sequence[triage.Candidate]:
    """Read back an answered decision file, naming every problem it has.

    Problems are collected across the whole file rather than failing on the
    first, so one pass over the error list fixes everything. A merge answered
    without replacements is legal — it means "accept the proposal", and the
    proposal is recomputed exactly as it was written.
    """
    try:
        with path.open() as f:
            data = json.load(f)
    except OSError as error:
        raise DecisionFileError([f"{path}: {error.strerror or error}"]) from error
    except json.JSONDecodeError as error:
        raise DecisionFileError([f"{path} is not JSON: {error}"]) from error
    if not isinstance(data, list):
        raise DecisionFileError(
            [f"{path} must hold a list of candidates, not {type(data).__name__}"]
        )
    candidates: list[triage.Candidate] = []
    problems: list[str] = []
    for position, entry in enumerate(data):
        if not isinstance(entry, Mapping):
            problems.append(f"candidate {position} is not an object")
            continue
        label = _entry_label(entry, position)
        try:
            candidate = triage.Candidate.from_dict(entry)
        except (KeyError, ValueError, TypeError) as error:
            problems.append(f"{label}: {error}")
            continue
        try:
            candidate = _finish_entry(candidate, entry, label)
        except DecisionFileError as error:
            problems.extend(error.problems)
            continue
        candidates.append(candidate)
    if problems:
        raise DecisionFileError(problems)
    return tuple(candidates)


def _finish_entry(
    candidate: triage.Candidate, entry: Mapping[str, Any], label: str
) -> triage.Candidate:
    if candidate.disposition is triage.Disposition.MERGE:
        if candidate.replacements is None:
            proposed = entry.get("proposed")
            if proposed is not None:
                candidate = attrs.evolve(
                    candidate,
                    replacements=tuple(triage.words_from_csv(proposed)),
                )
            else:
                proposal = apply_merges.propose_merge(candidate.words)
                if proposal is None:
                    raise DecisionFileError(
                        [
                            f"{label}: merge has no mechanical proposal; "
                            "write the merged note in 'replacements'"
                        ]
                    )
                candidate = attrs.evolve(candidate, replacements=(proposal,))
        elif len(candidate.replacements) != 1:
            raise DecisionFileError(
                [
                    f"{label}: a merge's replacements is the one combined "
                    f"note, not {len(candidate.replacements)} rows"
                ]
            )
    if candidate.disposition is triage.Disposition.SPLIT:
        if candidate.replacements is None or len(candidate.replacements) < 2:
            raise DecisionFileError(
                [f"{label}: a split needs at least two replacement rows"]
            )
        allocation = entry.get("sentence_allocation")
        if allocation is not None:
            candidate = _apply_sentence_allocation(candidate, allocation, label)
    return candidate


def record_decision_file(path: pathlib.Path, decisions: triage.DecisionCache) -> int:
    """Fold an answered decision file into the cache, returning new answers."""
    candidates = read_decision_file(path)
    recorded = triage.record_decisions(candidates, decisions)
    decisions.save()
    return recorded


def load_decisions(
    cache_path: pathlib.Path, decision_file: pathlib.Path | None
) -> triage.DecisionCache:
    """Open the decision cache and fold an answered decision file into it.

    The opening every entry point that resolves duplicates shares, so a
    harness sees the same messages from each and the same exit from a bad
    decision file: its problems on stderr, exit status 2, nothing recorded.
    The warning describes the cache as read from disk, before any fold.
    """
    decisions = triage.DecisionCache(cache_path)
    print(f"Read {len(decisions)} decisions from {cache_path}")
    if not len(decisions):
        print(
            "WARNING: no recorded decisions at "
            f"{cache_path.resolve()} - if you have answered "
            "questions before, check that you are running from the "
            "repository root: the default cache paths are relative."
        )
    if decision_file is not None:
        try:
            recorded = record_decision_file(decision_file, decisions)
        except DecisionFileError as error:
            print(f"error: {error}", file=sys.stderr)
            raise SystemExit(2) from error
        print(f"Folded in {recorded} new answers from {decision_file}")
    return decisions
