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

from textual.widgets import Input  # noqa: E402

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


async def _edit(
    candidates: Sequence[triage.Candidate],
    fields: dict[str, str],
    *,
    finish: str = "ctrl+s",
    then: Sequence[str] = (),
) -> Sequence[triage.Candidate]:
    """Open the editor, set some fields, and finish."""
    app = triage_ui.TriageApp(candidates)
    async with app.run_test() as pilot:
        await pilot.press("e")
        await pilot.pause()
        for widget_id, value in fields.items():
            app.screen.query_one(f"#{widget_id}", Input).value = value
        await pilot.press(finish)
        await pilot.pause()
        for key in then:
            await pilot.press(key)
        await pilot.pause()
    return app.return_value


def edit(
    candidates: Sequence[triage.Candidate],
    fields: dict[str, str],
    *,
    finish: str = "ctrl+s",
    then: Sequence[str] = (),
) -> Sequence[triage.Candidate]:
    return run(_edit(candidates, fields, finish=finish, then=then))


def segundo_candidate() -> triage.Candidate:
    """The case that motivated editing: two cards written identically."""
    return triage.Candidate(
        words=(
            make_word("segundo", "adj", "second", rank=223),
            make_word("segundo", "nm", "second", rank=1693),
        ),
        reason=triage.Reason.BASE_PART_OF_SPEECH,
        comparison=definition_similarity.Comparison(1.0, frozenset({"second"})),
        disposition=triage.Disposition.REVIEW,
    )


class TestEditing:
    def test_a_headword_can_be_rewritten(self) -> None:
        (resolved,) = edit(
            [segundo_candidate()], {"word-0": "segundo/a"}, then=["k"]
        )

        assert resolved.is_revised
        revised = {w.rank: w for w in resolved.revisions}
        assert revised[223].word == "segundo/a"
        assert revised[1693].word == "segundo"

    def test_an_article_can_be_added(self) -> None:
        (resolved,) = edit(
            [segundo_candidate()],
            {"word-0": "segundo/a", "article-1": "el"},
            then=["k"],
        )

        revised = {w.rank: w for w in resolved.revisions}
        assert revised[1693].article == "el"
        assert revised[223].article is None

    def test_the_originals_are_left_alone(self) -> None:
        """Decisions and the rejection record key on the notes in the deck."""
        candidate = segundo_candidate()
        (resolved,) = edit([candidate], {"word-0": "segundo/a"}, then=["k"])

        assert resolved.words == candidate.words
        assert resolved.current_words != candidate.words

    def test_cancelling_discards_the_edit(self) -> None:
        (resolved,) = edit(
            [segundo_candidate()],
            {"word-0": "segundo/a"},
            finish="ctrl+x",
            then=["k"],
        )

        assert not resolved.is_revised
        assert resolved.revisions is None

    def test_an_unparseable_part_of_speech_is_rejected(self) -> None:
        """A bad edit must not take down the session."""
        (resolved,) = edit(
            [segundo_candidate()],
            {"pos-0": "not-a-part-of-speech"},
            then=["ctrl+x", "k"],
        )

        assert not resolved.is_revised
        assert resolved.disposition is triage.Disposition.KEEP_SEPARATE

    def test_editing_does_not_decide(self) -> None:
        """Rewriting is a separate act from choosing what happens to the group."""
        (resolved,) = edit([segundo_candidate()], {"word-0": "segundo/a"}, then=["q"])

        assert resolved.is_revised
        assert resolved.disposition is triage.Disposition.REVIEW

    def test_editing_stays_on_the_same_group(self) -> None:
        candidates = [segundo_candidate(), make_candidate("cura")]
        resolved = edit(candidates, {"word-0": "segundo/a"}, then=["k", "m"])

        assert resolved[0].disposition is triage.Disposition.KEEP_SEPARATE
        assert resolved[1].disposition is triage.Disposition.MERGE

    def test_an_untouched_editor_records_no_revision(self) -> None:
        """Opening and saving without changing anything is not an edit."""
        (resolved,) = edit([segundo_candidate()], {}, then=["k"])

        assert triage.collect_revisions([resolved]) == ()

    def test_revisions_survive_the_triage_file(self, tmp_path) -> None:
        (resolved,) = edit([segundo_candidate()], {"word-0": "segundo/a"}, then=["k"])
        path = tmp_path / "triage.json"
        triage.write_triage_file(path, [resolved])

        (restored,) = triage.read_triage_file(path)
        assert restored.is_revised
        assert {w.word for w in restored.revisions} == {"segundo/a", "segundo"}
        assert {w.word for w in restored.words} == {"segundo"}

    def test_collect_revisions_returns_only_what_changed(self) -> None:
        (resolved,) = edit([segundo_candidate()], {"word-0": "segundo/a"}, then=["k"])

        (changed,) = triage.collect_revisions([resolved])
        assert changed.rank == 223
        assert changed.word == "segundo/a"
