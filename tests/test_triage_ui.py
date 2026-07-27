"""Tests for the triage resolution UI.

The UI is driven through Textual's test pilot, so these exercise the real key
bindings rather than calling the action methods directly. There is no async
test plugin configured, so each scenario runs its own event loop.
"""

import asyncio
from collections.abc import Awaitable, Sequence

import pytest

from anki_lang_cards import _types as types
from anki_lang_cards import definition_similarity, triage

pytest.importorskip("textual", reason="the triage UI needs the merge_ui extra")

from anki_lang_cards import triage_ui  # noqa: E402


def run[T](coroutine: Awaitable[T]) -> T:
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def make_word(word: str, part_of_speech: str, meaning: str, rank: int) -> types.Word:
    return types.Word(
        rank=rank,
        word=word,
        meaning=meaning,
        part_of_speech=part_of_speech,
        article=None,
        sentences=(),
    )


def make_candidate(
    word: str = "trepa",
    reason: triage.Reason = triage.Reason.CHARACTERISTICS,
    disposition: triage.Disposition = triage.Disposition.REVIEW,
    score: float = 0.5,
) -> triage.Candidate:
    return triage.Candidate(
        words=(
            make_word(word, "nm/f", "social climber", rank=1),
            make_word(word, "nf", "climber", rank=2),
        ),
        reason=reason,
        comparison=definition_similarity.Comparison(score, frozenset()),
        disposition=disposition,
    )


async def _press(
    candidates: Sequence[triage.Candidate], *keys: str
) -> Sequence[triage.Candidate]:
    """Run the app, send some keys, and return whatever it exits with."""
    app = triage_ui.TriageApp(candidates)
    async with app.run_test() as pilot:
        for key in keys:
            await pilot.press(key)
        await pilot.pause()
    return app.return_value


def press(
    candidates: Sequence[triage.Candidate], *keys: str
) -> Sequence[triage.Candidate]:
    return run(_press(candidates, *keys))


def test_no_candidates_does_not_open_the_ui() -> None:
    assert run(triage_ui.resolve_candidates(())) == ()


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("m", triage.Disposition.MERGE),
        ("k", triage.Disposition.KEEP_SEPARATE),
        ("u", triage.Disposition.REVIEW),
    ],
)
def test_each_key_records_its_disposition(
    key: str, expected: triage.Disposition
) -> None:
    resolved = press([make_candidate()], key)
    assert [c.disposition for c in resolved] == [expected]


def test_deciding_advances_to_the_next_candidate() -> None:
    candidates = [make_candidate("trepa"), make_candidate("cura")]
    resolved = press(candidates, "m", "k")

    assert [c.disposition for c in resolved] == [
        triage.Disposition.MERGE,
        triage.Disposition.KEEP_SEPARATE,
    ]


def test_deciding_the_last_candidate_exits() -> None:
    resolved = press([make_candidate()], "m")
    assert resolved is not None
    assert len(resolved) == 1


def test_skipping_leaves_the_suggestion_alone() -> None:
    candidates = [
        make_candidate("trepa", disposition=triage.Disposition.MERGE),
        make_candidate("cura", disposition=triage.Disposition.REVIEW),
    ]
    resolved = press(candidates, "s", "k")

    assert [c.disposition for c in resolved] == [
        triage.Disposition.MERGE,
        triage.Disposition.KEEP_SEPARATE,
    ]


def test_quitting_early_leaves_the_rest_untouched() -> None:
    candidates = [
        make_candidate("trepa", disposition=triage.Disposition.REVIEW),
        make_candidate("cura", disposition=triage.Disposition.MERGE),
        make_candidate("orden", disposition=triage.Disposition.REVIEW),
    ]
    resolved = press(candidates, "k", "q")

    assert [c.disposition for c in resolved] == [
        triage.Disposition.KEEP_SEPARATE,
        triage.Disposition.MERGE,
        triage.Disposition.REVIEW,
    ]


def test_going_back_allows_a_decision_to_be_changed() -> None:
    candidates = [make_candidate("trepa"), make_candidate("cura")]
    resolved = press(candidates, "m", "b", "k", "q")

    assert resolved[0].disposition is triage.Disposition.KEEP_SEPARATE


def test_going_back_from_the_first_candidate_does_nothing() -> None:
    candidates = [make_candidate("trepa"), make_candidate("cura")]
    resolved = press(candidates, "b", "m", "q")

    assert resolved[0].disposition is triage.Disposition.MERGE


def test_decisions_do_not_disturb_the_grouped_words() -> None:
    candidate = make_candidate()
    (resolved,) = press([candidate], "m")

    assert resolved.words == candidate.words
    assert resolved.reason is candidate.reason
    assert resolved.comparison == candidate.comparison


def test_resolve_candidates_returns_the_decisions() -> None:
    async def scenario() -> Sequence[triage.Candidate]:
        app = triage_ui.TriageApp([make_candidate()])
        async with app.run_test() as pilot:
            await pilot.press("k")
            await pilot.pause()
        return app.return_value

    assert [c.disposition for c in run(scenario())] == [
        triage.Disposition.KEEP_SEPARATE
    ]


@pytest.mark.parametrize("reason", list(triage.Reason))
def test_every_reason_has_an_explanation(reason: triage.Reason) -> None:
    """A group with no explanation would render blank rather than fail loudly."""
    assert triage_ui._REASON_TEXT[reason]


@pytest.mark.parametrize("reason", list(triage.Reason))
def test_every_reason_renders(reason: triage.Reason) -> None:
    resolved = press([make_candidate(reason=reason)], "m")
    assert [c.disposition for c in resolved] == [triage.Disposition.MERGE]


def test_shared_glosses_are_shown_when_present() -> None:
    candidate = triage.Candidate(
        words=(
            make_word("zanja", "nf", "trench, ditch", rank=1),
            make_word("zanja", "nf", "ditch", rank=2),
        ),
        reason=triage.Reason.COMPATIBLE,
        comparison=definition_similarity.Comparison(0.8, frozenset({"ditch"})),
        disposition=triage.Disposition.MERGE,
    )
    view = triage_ui.CandidateView(candidate)
    assert "ditch" in view.render()


def test_groups_of_more_than_two_are_handled() -> None:
    candidate = triage.Candidate(
        words=(
            make_word("trepa", "nm/f", "social climber", rank=1),
            make_word("trepa", "nf", "climber", rank=2),
            make_word("trepa", "n", "social climber", rank=3),
        ),
        reason=triage.Reason.CHARACTERISTICS,
        comparison=definition_similarity.Comparison(0.9, frozenset()),
        disposition=triage.Disposition.REVIEW,
    )
    resolved = press([candidate], "m")
    assert len(resolved[0].words) == 3
