"""Tests for the triage resolution UI.

The UI is driven through Textual's test pilot, so these exercise the real key
bindings rather than calling the action methods directly. There is no async
test plugin configured, so each scenario runs its own event loop.
"""

import asyncio
import re
from collections.abc import Awaitable, Sequence

import pytest

from anki_lang_cards import _types as types
from anki_lang_cards import definition_similarity, triage

pytest.importorskip("textual", reason="the triage UI needs the merge_ui extra")

from textual.widgets import Input  # noqa: E402

from anki_lang_cards import triage_ui  # noqa: E402


def run[T](coroutine: Awaitable[T]) -> T:
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def make_word(
    word: str, part_of_speech: str, meaning: str, rank: int | None
) -> types.Word:
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
        (resolved,) = edit([segundo_candidate()], {"word-0": "segundo/a"}, then=["k"])

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


def sentence(spanish: str, english: str = "", source=None) -> types.Sentence:
    return types.Sentence(spanish=spanish, english=english, source=source)


def word_with_sentences(*sentences: types.Sentence) -> types.Word:
    return types.Word(
        rank=1,
        word="nodriza",
        meaning="mother ship",
        part_of_speech="nf",
        article=None,
        sentences=sentences,
    )


class TestSentences:
    def test_sentences_are_shown(self) -> None:
        word = word_with_sentences(
            sentence("El turbojet regresa a la nave nodriza.", "It returns.")
        )
        rendered = triage_ui.SentencesView(word).render()

        assert "El turbojet regresa" in rendered
        assert "It returns." in rendered

    def test_a_note_with_no_sentences_says_so(self) -> None:
        rendered = triage_ui.SentencesView(word_with_sentences()).render()
        assert "No example sentences" in rendered

    def test_a_page_starts_where_it_is_told_to(self) -> None:
        word = word_with_sentences(*(sentence(f"Frase {i}.") for i in range(5)))
        rendered = triage_ui.SentencesView(word, start=3).render()

        assert "Frase 3." in rendered
        assert "Frase 0." not in rendered

    @pytest.mark.parametrize(
        ("citation", "expected"),
        [
            (
                types.BookCitation(
                    title="La vieja familia", authors=("X",), pages=(156,)
                ),
                "La vieja familia p156",
            ),
            (
                types.BookCitation(title="Dune", authors=("Herbert",)),
                "Dune",
            ),
            (
                types.ModelCitation(model="claude-sonnet-4-5"),
                "model: claude-sonnet-4-5",
            ),
            (None, "unattributed"),
        ],
    )
    def test_sources_are_named(self, citation, expected: str) -> None:
        """Whether a sentence came from a book or the model changes what it is
        evidence of."""
        assert triage_ui.describe_source(citation) == expected

    def test_toggling_does_not_disturb_decisions(self) -> None:
        resolved = press([make_candidate(), make_candidate("cura")], "v", "m", "v", "k")

        assert [c.disposition for c in resolved] == [
            triage.Disposition.MERGE,
            triage.Disposition.KEEP_SEPARATE,
        ]

    def test_toggling_survives_moving_between_groups(self) -> None:
        """Re-rendering used to crash on a duplicate container id."""
        candidates = [make_candidate("a"), make_candidate("b"), make_candidate("c")]
        resolved = press(candidates, "v", "m", "m", "v", "b", "k", "q")

        assert resolved[1].disposition is triage.Disposition.KEEP_SEPARATE


class TestSharedRank:
    def test_one_note_read_twice_is_called_out(self) -> None:
        """The "prenda" case: same rank, same word, differing definitions."""
        candidate = triage.Candidate(
            words=(
                make_word("prenda", "nf", "piece of clothing", rank=3467),
                make_word("prenda", "nf", "piece of glothing, garment", rank=3467),
            ),
            reason=triage.Reason.COMPATIBLE,
            comparison=definition_similarity.Comparison(0.35, frozenset()),
            disposition=triage.Disposition.REVIEW,
        )
        rendered = triage_ui.CandidateView(candidate).render()

        assert "one note (rank 3467)" in rendered

    def test_two_notes_are_not_called_out(self) -> None:
        candidate = triage.Candidate(
            words=(
                make_word("trepa", "nm/f", "social climber", rank=8576),
                make_word("trepa", "nf", "social climber", rank=9615),
            ),
            reason=triage.Reason.CHARACTERISTICS,
            comparison=definition_similarity.Comparison(1.0, frozenset()),
            disposition=triage.Disposition.MERGE,
        )
        assert "one note" not in triage_ui.CandidateView(candidate).render()

    def test_notes_without_ranks_are_not_called_out(self) -> None:
        candidate = triage.Candidate(
            words=(
                make_word("trepa", "nm/f", "social climber", rank=None),
                make_word("trepa", "nf", "climber", rank=None),
            ),
            reason=triage.Reason.CHARACTERISTICS,
            comparison=definition_similarity.Comparison(1.0, frozenset()),
            disposition=triage.Disposition.MERGE,
        )
        assert "one note" not in triage_ui.CandidateView(candidate).render()


class TestSentenceCarousel:
    """ "v" turns the page rather than revealing what was clipped."""

    @staticmethod
    def wordy(count: int, tag: str = "S") -> types.Word:
        return types.Word(
            rank=1,
            word="metomentodo",
            meaning="busybody",
            part_of_speech="nm/f",
            article=None,
            sentences=tuple(
                types.Sentence(
                    spanish=f"{tag}{i} es un metomentodo.",
                    english=f"EN{i}",
                    source=None,
                )
                for i in range(count)
            ),
        )

    @staticmethod
    def shown(view: triage_ui.SentencesView) -> list[str]:
        return re.findall(r"\bS(\d+)\b", view.render())

    def test_a_pane_shows_what_fits_and_says_where_it_is(self) -> None:
        view = triage_ui.SentencesView(self.wordy(9))
        view.start = 0
        rendered = view.render()
        assert "of 9" in rendered

    def test_a_short_note_says_how_many_it_has(self) -> None:
        view = triage_ui.SentencesView(self.wordy(2))
        assert "2 sentences" in view.render()
        assert "v for more" not in view.render()

    @pytest.mark.parametrize(
        ("height", "expected"),
        [(5, 1), (9, 2), (17, 4), (100, 9)],
    )
    def test_fitting_counts_whole_sentences(self, height: int, expected: int) -> None:
        view = triage_ui.SentencesView(self.wordy(9))
        assert view.fitting(60, height) == expected

    def test_at_least_one_sentence_is_always_reported(self) -> None:
        """A pane with no room should clip, not leave "v" nothing to move by."""
        view = triage_ui.SentencesView(self.wordy(9))
        assert view.fitting(20, 1) == 1

    def test_a_note_with_no_sentences_fits_nothing(self) -> None:
        assert triage_ui.SentencesView(self.wordy(0)).fitting(60, 20) == 0

    def test_fitting_accounts_for_wrapping(self) -> None:
        """A book sentence in a narrow column runs to more lines than a short one."""
        view = triage_ui.SentencesView(self.wordy(9))
        assert view.fitting(20, 20) < view.fitting(200, 20)

    def test_pressing_v_turns_the_page(self) -> None:
        async def scenario() -> list[list[str]]:
            candidate = triage.Candidate(
                words=(self.wordy(9),),
                reason=triage.Reason.COMPATIBLE,
                comparison=definition_similarity.Comparison(1.0, frozenset()),
                disposition=triage.Disposition.REVIEW,
            )
            app = triage_ui.TriageApp([candidate], per_page=2)
            pages = []
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                for _ in range(6):
                    view = app.screen.query(triage_ui.SentencesView).first()
                    pages.append(self.shown(view))
                    await pilot.press("v")
                    await pilot.pause()
            return pages

        pages = run(scenario())
        assert pages[0] == ["0", "1"]
        assert pages[1] == ["2", "3"]
        assert pages[4] == ["8"]
        assert pages[5] == ["0", "1"], "the carousel wraps"

    def test_the_page_resets_when_moving_to_another_group(self) -> None:
        async def scenario() -> int:
            candidates = [
                triage.Candidate(
                    words=(self.wordy(9),),
                    reason=triage.Reason.COMPATIBLE,
                    comparison=definition_similarity.Comparison(1.0, frozenset()),
                    disposition=triage.Disposition.REVIEW,
                )
                for _ in range(2)
            ]
            app = triage_ui.TriageApp(candidates, per_page=2)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await pilot.press("v")
                await pilot.press("s")
                await pilot.pause()
                return app.screen.sentence_offset

        assert run(scenario()) == 0

    def test_a_fixed_page_size_overrides_the_fit(self) -> None:
        async def scenario() -> int:
            candidate = triage.Candidate(
                words=(self.wordy(9),),
                reason=triage.Reason.COMPATIBLE,
                comparison=definition_similarity.Comparison(1.0, frozenset()),
                disposition=triage.Disposition.REVIEW,
            )
            app = triage_ui.TriageApp([candidate], per_page=3)
            async with app.run_test(size=(200, 80)) as pilot:
                await pilot.pause()
                await pilot.press("v")
                await pilot.pause()
                return app.screen.sentence_offset

        assert run(scenario()) == 3


class TestSides:
    """A cross-part-of-speech group asks about two sides, not N notes."""

    @staticmethod
    def deber() -> triage.Candidate:
        verb = make_word("deber", "v", "should, ought to", rank=71)
        nouns = (
            make_word("deber", "nm", "duty, obligation", rank=2187),
            make_word("deber", "nm", "duty, obligation, homework", rank=2187),
        )
        return triage.Candidate(
            words=(verb, *nouns),
            reason=triage.Reason.BASE_PART_OF_SPEECH,
            comparison=definition_similarity.Comparison(0.0, frozenset()),
            disposition=triage.Disposition.REVIEW,
            sides=((verb,), nouns),
        )

    def test_the_sides_are_kept(self) -> None:
        assert [len(side) for side in self.deber().current_sides] == [1, 2]

    def test_a_single_group_is_one_side(self) -> None:
        assert len(make_candidate().current_sides) == 1

    def test_the_screen_says_what_the_question_is(self) -> None:
        rendered = triage_ui.CandidateView(self.deber()).render()
        assert "whether the two sides belong together" in rendered
        assert "does not settle it" in rendered

    def test_a_single_group_does_not_mention_sides(self) -> None:
        assert "two sides" not in triage_ui.CandidateView(make_candidate()).render()

    def test_the_sides_are_laid_out_separately(self) -> None:
        async def scenario() -> tuple[int, int]:
            app = triage_ui.TriageApp([self.deber()])
            async with app.run_test(size=(170, 50)) as pilot:
                await pilot.pause()
                return len(app.screen.query(".side")), len(
                    app.screen.query(".word-col")
                )

        assert run(scenario()) == (2, 3)

    def test_keeping_sides_apart_still_lets_the_inner_pair_merge(self) -> None:
        """The worry this answers: separating the verb must not block the two
        noun notes from merging, which is a different candidate entirely."""
        answered = press([self.deber()], "k")
        assert answered[0].disposition is triage.Disposition.KEEP_SEPARATE
        # The inner pair is its own candidate, untouched by that answer.
        inner = triage.Candidate(
            words=self.deber().sides[1],
            reason=triage.Reason.COMPATIBLE,
            comparison=definition_similarity.Comparison(0.79, frozenset()),
            disposition=triage.Disposition.MERGE,
        )
        assert answered[0].key != inner.key
