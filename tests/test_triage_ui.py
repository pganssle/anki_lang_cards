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
        ("a", triage.Disposition.MERGE),
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
    resolved = press(candidates, "a", "k")

    assert [c.disposition for c in resolved] == [
        triage.Disposition.MERGE,
        triage.Disposition.KEEP_SEPARATE,
    ]


def test_deciding_the_last_candidate_exits() -> None:
    resolved = press([make_candidate()], "a")
    assert resolved is not None
    assert len(resolved) == 1


def test_skipping_leaves_the_suggestion_alone() -> None:
    candidates = [
        make_candidate("trepa", disposition=triage.Disposition.MERGE),
        make_candidate("cura", disposition=triage.Disposition.REVIEW),
    ]
    resolved = press(candidates, "n", "k")

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
    resolved = press(candidates, "a", "b", "k", "q")

    assert resolved[0].disposition is triage.Disposition.KEEP_SEPARATE


def test_going_back_from_the_first_candidate_does_nothing() -> None:
    candidates = [make_candidate("trepa"), make_candidate("cura")]
    resolved = press(candidates, "b", "a", "q")

    assert resolved[0].disposition is triage.Disposition.MERGE


def test_decisions_do_not_disturb_the_grouped_words() -> None:
    candidate = make_candidate()
    (resolved,) = press([candidate], "a")

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
    resolved = press([make_candidate(reason=reason)], "a")
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
    resolved = press([candidate], "a")
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

        assert resolved.has_replacements
        revised = {w.rank: w for w in resolved.replacements}
        assert revised[223].word == "segundo/a"
        assert revised[1693].word == "segundo"

    def test_an_article_can_be_added(self) -> None:
        (resolved,) = edit(
            [segundo_candidate()],
            {"word-0": "segundo/a", "article-1": "el"},
            then=["k"],
        )

        revised = {w.rank: w for w in resolved.replacements}
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

        assert not resolved.has_replacements
        assert resolved.replacements is None

    def test_an_unparseable_part_of_speech_is_rejected(self) -> None:
        """A bad edit must not take down the session."""
        (resolved,) = edit(
            [segundo_candidate()],
            {"pos-0": "not-a-part-of-speech"},
            then=["ctrl+x", "k"],
        )

        assert not resolved.has_replacements
        assert resolved.disposition is triage.Disposition.KEEP_SEPARATE

    def test_editing_does_not_decide(self) -> None:
        """Rewriting is a separate act from choosing what happens to the group."""
        (resolved,) = edit([segundo_candidate()], {"word-0": "segundo/a"}, then=["q"])

        assert resolved.has_replacements
        assert resolved.disposition is triage.Disposition.REVIEW

    def test_editing_stays_on_the_same_group(self) -> None:
        candidates = [segundo_candidate(), make_candidate("cura")]
        resolved = edit(candidates, {"word-0": "segundo/a"}, then=["k", "a"])

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
        assert restored.has_replacements
        assert {w.word for w in restored.replacements} == {"segundo/a", "segundo"}
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
        resolved = press([make_candidate(), make_candidate("cura")], "v", "a", "v", "k")

        assert [c.disposition for c in resolved] == [
            triage.Disposition.MERGE,
            triage.Disposition.KEEP_SEPARATE,
        ]

    def test_toggling_survives_moving_between_groups(self) -> None:
        """Re-rendering used to crash on a duplicate container id."""
        candidates = [make_candidate("a"), make_candidate("b"), make_candidate("c")]
        resolved = press(candidates, "v", "a", "a", "v", "b", "k", "q")

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
    def test_capacity_counts_whole_sentences(self, height: int, expected: int) -> None:
        view = triage_ui.SentencesView(self.wordy(9))
        assert view.capacity(60, height) == expected

    def test_at_least_one_sentence_is_always_reported(self) -> None:
        """A pane with no room should clip, not leave "v" nothing to move by."""
        view = triage_ui.SentencesView(self.wordy(9))
        assert view.capacity(20, 1) == 1

    def test_a_note_with_no_sentences_fits_nothing(self) -> None:
        assert triage_ui.SentencesView(self.wordy(0)).capacity(60, 20) == 0

    def test_capacity_accounts_for_wrapping(self) -> None:
        """A book sentence in a narrow column runs to more lines than a short one."""
        view = triage_ui.SentencesView(self.wordy(9))
        assert view.capacity(20, 20) < view.capacity(200, 20)

    def test_capacity_does_not_depend_on_the_page(self) -> None:
        """The pane holds as much on its last page as on its first."""
        first = triage_ui.SentencesView(self.wordy(9), start=0)
        last = triage_ui.SentencesView(self.wordy(9), start=8)
        assert first.capacity(60, 17) == last.capacity(60, 17)

    @pytest.mark.parametrize(
        ("sentences", "start", "capacity", "expected"),
        [
            (10, 0, 3, 0),
            (10, 3, 3, 3),
            (10, 9, 3, 7),  # past the end: hold at the last full page
            (1, 6, 3, 0),  # a one-sentence note never moves
            (2, 6, 3, 0),  # fewer sentences than a page: still from the top
        ],
    )
    def test_a_pane_holds_at_its_last_full_page(
        self, sentences: int, start: int, capacity: int, expected: int
    ) -> None:
        view = triage_ui.SentencesView(self.wordy(sentences), start=start)
        assert view.page_start(capacity) == expected

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
        # The last stop is a full page, not a page hanging off the end.
        assert pages[3] == ["6", "7"]
        assert pages[4] == ["7", "8"]
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
                await pilot.press("n")
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


class TestUnevenSentenceCounts:
    """A short note must not go blank while a long one pages."""

    @staticmethod
    def note(count: int, tag: str) -> types.Word:
        return types.Word(
            rank=1,
            word="w",
            meaning="m",
            part_of_speech="nm",
            article=None,
            sentences=tuple(
                types.Sentence(spanish=f"{tag}{i} frase.", english="", source=None)
                for i in range(count)
            ),
        )

    def pages(self, presses: int, per_page: int = 3) -> list[tuple[list, list]]:
        candidate = triage.Candidate(
            words=(self.note(10, "L"), self.note(1, "S")),
            reason=triage.Reason.COMPATIBLE,
            comparison=definition_similarity.Comparison(1.0, frozenset()),
            disposition=triage.Disposition.REVIEW,
        )

        async def scenario():
            app = triage_ui.TriageApp([candidate], per_page=per_page)
            seen = []
            async with app.run_test(size=(160, 50)) as pilot:
                await pilot.pause()
                for _ in range(presses):
                    views = list(app.screen.query(triage_ui.SentencesView))
                    seen.append(
                        (
                            re.findall(r"\bL(\d+)\b", views[0].render()),
                            re.findall(r"\bS(\d+)\b", views[1].render()),
                        )
                    )
                    await pilot.press("v")
                    await pilot.pause()
            return seen

        return run(scenario())

    def test_the_short_note_never_goes_blank(self) -> None:
        for _, short in self.pages(5):
            assert short == ["0"]

    def test_the_long_note_pages_through_all_of_them(self) -> None:
        seen = {index for long, _ in self.pages(5) for index in long}
        assert seen == {str(i) for i in range(10)}

    def test_the_last_page_is_full(self) -> None:
        """Rather than a single trailing sentence with empty space beside it."""
        long_pages = [long for long, _ in self.pages(5)]
        assert long_pages[3] == ["7", "8", "9"]

    def test_it_wraps_once_the_longest_is_at_its_end(self) -> None:
        long_pages = [long for long, _ in self.pages(5)]
        assert long_pages[4] == ["0", "1", "2"]


def colliding_candidate(with_sentences: bool = False) -> triage.Candidate:
    """Two notes the lookup puts on one card: keep-separate needs a warning."""
    sentences = (
        (sentence("Cavaron una zanja.", "They dug a trench."),),
        (sentence("La zanja es honda.", "The ditch is deep."),),
    )
    return triage.Candidate(
        words=(
            types.Word(
                rank=1,
                word="zanja",
                meaning="trench, ditch",
                part_of_speech="nf",
                article="la",
                sentences=sentences[0] if with_sentences else (),
            ),
            types.Word(
                rank=2,
                word="zanja",
                meaning="ditch",
                part_of_speech="nf",
                article=None,
                sentences=sentences[1] if with_sentences else (),
            ),
        ),
        reason=triage.Reason.COMPATIBLE,
        comparison=definition_similarity.Comparison(0.8, frozenset({"ditch"})),
        disposition=triage.Disposition.REVIEW,
    )


class TestAccept:
    """Accepting records the answer and the wording as one act."""

    def test_accept_records_the_proposed_note(self) -> None:
        (resolved,) = press([make_candidate()], "a")

        assert resolved.disposition is triage.Disposition.MERGE
        assert resolved.replacements is not None
        (merged,) = resolved.replacements
        assert merged.rank == 1
        assert merged.meaning == "social climber, climber"

    def test_the_proposal_is_shown_before_accepting(self) -> None:
        async def scenario() -> str:
            app = triage_ui.TriageApp([make_candidate()])
            async with app.run_test(size=(140, 50)) as pilot:
                await pilot.pause()
                view = app.screen.query(".proposal").first()
                return view.render()

        assert "social climber, climber" in str(run(scenario()))

    def test_notes_are_numbered_on_screen(self) -> None:
        async def scenario() -> list[str]:
            app = triage_ui.TriageApp([make_candidate()])
            async with app.run_test(size=(140, 50)) as pilot:
                await pilot.pause()
                return [
                    str(label.render()) for label in app.screen.query(".note-number")
                ]

        rendered = run(scenario())
        assert any("#1" in text for text in rendered)
        assert any("#2" in text for text in rendered)


class TestKeepSeparateGuard:
    """Keep-separate warns when the notes would just collide again."""

    def test_a_colliding_group_asks_for_confirmation(self) -> None:
        async def scenario() -> tuple[bool, Sequence[triage.Candidate]]:
            app = triage_ui.TriageApp([colliding_candidate()])
            async with app.run_test() as pilot:
                await pilot.press("k")
                await pilot.pause()
                modal_open = isinstance(app.screen, triage_ui.ConfirmScreen)
                await pilot.press("y")
                await pilot.pause()
            return modal_open, app.return_value

        modal_open, resolved = run(scenario())
        assert modal_open
        assert resolved[0].disposition is triage.Disposition.KEEP_SEPARATE

    def test_declining_the_warning_leaves_the_group_undecided(self) -> None:
        async def scenario() -> Sequence[triage.Candidate]:
            app = triage_ui.TriageApp([colliding_candidate()])
            async with app.run_test() as pilot:
                await pilot.press("k")
                await pilot.pause()
                await pilot.press("n")
                await pilot.pause()
                await pilot.press("q")
                await pilot.pause()
            return app.return_value

        resolved = run(scenario())
        assert resolved[0].disposition is triage.Disposition.REVIEW

    def test_notes_edited_apart_need_no_confirmation(self) -> None:
        """A sense marker changes the lookup key, so the collision is gone."""
        (resolved,) = edit(
            [colliding_candidate()], {"word-1": "zanja (obra)"}, then=["k"]
        )
        assert resolved.disposition is triage.Disposition.KEEP_SEPARATE

    def test_cross_group_candidates_need_no_confirmation(self) -> None:
        """Different parts of speech can already coexist as separate cards."""
        (resolved,) = press([make_candidate()], "k")
        assert resolved.disposition is triage.Disposition.KEEP_SEPARATE


class TestChooseNote:
    """c takes one note as the base without un-filling anything."""

    @staticmethod
    async def _choose(number: str, *then: str) -> Sequence[triage.Candidate]:
        app = triage_ui.TriageApp([colliding_candidate()])
        async with app.run_test() as pilot:
            await pilot.press("c")
            await pilot.pause()
            prompt = app.screen.query_one(triage_ui.ChooseNoteInput)
            prompt.value = number
            await pilot.press("enter")
            await pilot.pause()
            for key in then:
                await pilot.press(key)
                await pilot.pause()
        return app.return_value

    def test_choosing_takes_the_wording_but_not_the_gaps(self) -> None:
        (resolved,) = run(self._choose("2", "a"))

        (merged,) = resolved.replacements
        assert merged.meaning == "ditch"
        assert merged.rank == 1, "the rank is the group's lowest, not note 2's"
        assert merged.article == "la", "note 2 has no article; note 1's stays"

    def test_an_out_of_range_number_changes_nothing(self) -> None:
        (resolved,) = run(self._choose("9", "a"))

        (merged,) = resolved.replacements
        assert merged.meaning == "trench, ditch"


class TestMergedNoteEditor:
    """m writes the note a merge will produce."""

    @staticmethod
    async def _run_editor(
        candidates: Sequence[triage.Candidate],
        act,
        *then: str,
    ):
        app = triage_ui.TriageApp(candidates)
        async with app.run_test() as pilot:
            await pilot.press("m")
            await pilot.pause()
            await act(app, pilot)
            for key in then:
                await pilot.press(key)
                await pilot.pause()
        return app.return_value

    def test_an_edited_proposal_is_what_accept_records(self) -> None:
        async def act(app, pilot):
            app.screen.query_one("#meaning", Input).value = "climber (social)"
            await pilot.press("ctrl+s")
            await pilot.pause()

        resolved = run(self._run_editor([make_candidate()], act, "a"))
        (merged,) = resolved[0].replacements
        assert merged.meaning == "climber (social)"

    def test_cancelling_keeps_the_mechanical_proposal(self) -> None:
        async def act(app, pilot):
            app.screen.query_one("#meaning", Input).value = "climber (social)"
            await pilot.press("ctrl+x")
            await pilot.pause()

        resolved = run(self._run_editor([make_candidate()], act, "a"))
        (merged,) = resolved[0].replacements
        assert merged.meaning == "social climber, climber"

    def test_an_explicit_rank_edit_is_kept(self) -> None:
        async def act(app, pilot):
            app.screen.query_one("#rank", Input).value = "7"
            await pilot.press("ctrl+s")
            await pilot.pause()

        resolved = run(self._run_editor([make_candidate()], act, "a"))
        assert resolved[0].replacements[0].rank == 7

    def test_pull_copies_the_focused_field_from_a_numbered_note(self) -> None:
        async def act(app, pilot):
            app.screen.query_one("#meaning", Input).focus()
            await pilot.pause()
            await pilot.press("ctrl+p")
            await pilot.pause()
            app.screen.query_one("#pull-input", Input).value = "2"
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen.query_one("#meaning", Input).value == "climber"
            await pilot.press("ctrl+s")
            await pilot.pause()

        resolved = run(self._run_editor([make_candidate()], act, "a"))
        assert resolved[0].replacements[0].meaning == "climber"

    def test_pull_can_name_the_field_instead(self) -> None:
        async def act(app, pilot):
            await pilot.press("ctrl+p")
            await pilot.pause()
            app.screen.query_one("#pull-input", Input).value = "1 meaning"
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen.query_one("#meaning", Input).value == "social climber"
            await pilot.press("ctrl+x")
            await pilot.pause()

        run(self._run_editor([make_candidate()], act, "u"))


class TestSplitEditor:
    """s divides a group into new cards."""

    @staticmethod
    async def _split(candidate: triage.Candidate, act) -> Sequence[triage.Candidate]:
        app = triage_ui.TriageApp([candidate])
        async with app.run_test(size=(160, 60)) as pilot:
            await pilot.press("s")
            await pilot.pause()
            await pilot.press("enter")  # accept the default card count
            await pilot.pause()
            await act(app, pilot)
            await pilot.press("ctrl+s")
            await pilot.pause()
        return app.return_value

    def test_the_grid_prefills_from_the_group(self) -> None:
        async def act(app, pilot):
            assert app.screen.query_one("#word-0", Input).value == "zanja"
            assert app.screen.query_one("#meaning-1", Input).value == "ditch"

        (resolved,) = run(self._split(colliding_candidate(), act))
        assert resolved.disposition is triage.Disposition.SPLIT

    def test_a_split_records_its_cards(self) -> None:
        async def act(app, pilot):
            app.screen.query_one("#word-1", Input).value = "zanja (obra)"
            app.screen.query_one("#meaning-1", Input).value = "ditch (works)"

        (resolved,) = run(self._split(colliding_candidate(), act))

        assert resolved.disposition is triage.Disposition.SPLIT
        assert [w.word for w in resolved.replacements] == ["zanja", "zanja (obra)"]
        assert resolved.replacements[1].meaning == "ditch (works)"
        assert resolved.replacements[0].rank == 1
        assert resolved.replacements[1].rank == 2

    def test_sentences_follow_their_allocation(self) -> None:
        async def act(app, pilot):
            app.screen.query_one("#allocation-0", Input).value = "1,2"
            app.screen.query_one("#allocation-1", Input).value = "2"

        (resolved,) = run(self._split(colliding_candidate(with_sentences=True), act))

        first, second = resolved.replacements
        assert [s.spanish for s in first.sentences] == ["Cavaron una zanja."]
        assert [s.spanish for s in second.sentences] == [
            "Cavaron una zanja.",
            "La zanja es honda.",
        ]

    def test_allocations_start_empty_and_label_the_origin(self) -> None:
        """Where a sentence came from is shown, not answered on your behalf."""

        async def act(app, pilot):
            assert app.screen.query_one("#allocation-0", Input).value == ""
            assert app.screen.query_one("#allocation-1", Input).value == ""
            origins = [
                str(label.render()) for label in app.screen.query(".allocation-origin")
            ]
            assert "from #1" in origins[0]
            assert "from #2" in origins[1]
            # Saving with nothing chosen must refuse: an empty box is not an
            # answer.
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert isinstance(app.screen, triage_ui.SplitEditor)
            app.screen.query_one("#allocation-0", Input).value = "1"
            app.screen.query_one("#allocation-1", Input).value = "2"

        (resolved,) = run(self._split(colliding_candidate(with_sentences=True), act))
        assert resolved.disposition is triage.Disposition.SPLIT

    def test_a_bad_allocation_keeps_the_editor_open(self) -> None:
        async def scenario() -> Sequence[triage.Candidate]:
            app = triage_ui.TriageApp([colliding_candidate(with_sentences=True)])
            async with app.run_test(size=(160, 60)) as pilot:
                await pilot.press("s")
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                app.screen.query_one("#allocation-0", Input).value = "9"
                await pilot.press("ctrl+s")
                await pilot.pause()
                assert isinstance(app.screen, triage_ui.SplitEditor)
                await pilot.press("ctrl+x")
                await pilot.pause()
                await pilot.press("q")
                await pilot.pause()
            return app.return_value

        resolved = run(scenario())
        assert resolved[0].disposition is triage.Disposition.REVIEW

    def test_shift_right_copies_a_field_to_the_next_card(self) -> None:
        async def act(app, pilot):
            word_input = app.screen.query_one("#word-0", Input)
            word_input.value = "zanja (nueva)"
            word_input.focus()
            await pilot.pause()
            await pilot.press("shift+right")
            await pilot.pause()
            assert app.screen.query_one("#word-1", Input).value == "zanja (nueva)"
            app.screen.query_one("#meaning-1", Input).value = "ditch"

        (resolved,) = run(self._split(colliding_candidate(), act))
        assert resolved.replacements[1].word == "zanja (nueva)"

    def test_an_extra_card_starts_blank_and_rankless(self) -> None:
        async def scenario() -> Sequence[triage.Candidate]:
            app = triage_ui.TriageApp([colliding_candidate()])
            async with app.run_test(size=(160, 60)) as pilot:
                await pilot.press("s")
                await pilot.pause()
                count = app.screen.query_one("#count-input", Input)
                count.value = "3"
                await pilot.press("enter")
                await pilot.pause()
                assert app.screen.query_one("#word-2", Input).value == ""
                assert app.screen.query_one("#rank-2", Input).value == ""
                app.screen.query_one("#word-2", Input).value = "zanja (tercera)"
                app.screen.query_one("#pos-2", Input).value = "nf"
                app.screen.query_one("#meaning-2", Input).value = "third sense"
                await pilot.press("ctrl+s")
                await pilot.pause()
            return app.return_value

        (resolved,) = run(scenario())
        assert len(resolved.replacements) == 3
        assert resolved.replacements[2].rank is None
        assert resolved.replacements[2].word == "zanja (tercera)"


class TestOnDecision:
    """Every answer is reported the moment it is given."""

    @staticmethod
    async def _observe(
        candidates: Sequence[triage.Candidate], *keys: str
    ) -> list[triage.Candidate]:
        heard: list[triage.Candidate] = []
        app = triage_ui.TriageApp(candidates, on_decision=heard.append)
        async with app.run_test() as pilot:
            for key in keys:
                await pilot.press(key)
                await pilot.pause()
        return heard

    def test_answers_are_reported_as_they_happen(self) -> None:
        heard = run(
            self._observe([make_candidate("trepa"), make_candidate("cura")], "a", "k")
        )
        assert [c.disposition for c in heard] == [
            triage.Disposition.MERGE,
            triage.Disposition.KEEP_SEPARATE,
        ]
        assert heard[0].replacements is not None

    def test_quitting_early_reports_only_what_was_answered(self) -> None:
        heard = run(
            self._observe([make_candidate("trepa"), make_candidate("cura")], "a", "q")
        )
        assert len(heard) == 1

    def test_skipping_reports_nothing(self) -> None:
        heard = run(
            self._observe([make_candidate("trepa"), make_candidate("cura")], "n", "q")
        )
        assert heard == []

    def test_undecided_is_reported_for_the_caller_to_filter(self) -> None:
        heard = run(self._observe([make_candidate("trepa")], "u"))
        assert [c.disposition for c in heard] == [triage.Disposition.REVIEW]

    def test_an_edit_is_reported_before_any_answer(self) -> None:
        async def scenario() -> list[triage.Candidate]:
            heard: list[triage.Candidate] = []
            app = triage_ui.TriageApp([segundo_candidate()], on_decision=heard.append)
            async with app.run_test() as pilot:
                await pilot.press("e")
                await pilot.pause()
                app.screen.query_one("#word-0", Input).value = "segundo/a"
                await pilot.press("ctrl+s")
                await pilot.pause()
                await pilot.press("q")
                await pilot.pause()
            return heard

        heard = run(scenario())
        assert len(heard) == 1
        assert heard[0].disposition is triage.Disposition.REVIEW
        assert heard[0].has_replacements

    def test_changing_an_answer_reports_the_change(self) -> None:
        heard = run(
            self._observe(
                [make_candidate("trepa"), make_candidate("cura")], "a", "b", "k", "q"
            )
        )
        assert [(c.words[0].word, c.disposition) for c in heard] == [
            ("trepa", triage.Disposition.MERGE),
            ("trepa", triage.Disposition.KEEP_SEPARATE),
        ]
        assert heard[1].replacements is None, "the merged note went with the merge"


class TestSplitEditorLayout:
    """A long sentence list must not crush the card grid off the screen."""

    @staticmethod
    def wordy_candidate(count: int) -> triage.Candidate:
        def note(rank: int, meaning: str, tag: str) -> types.Word:
            return types.Word(
                rank=rank,
                word="llave",
                meaning=meaning,
                part_of_speech="nf",
                article=None,
                sentences=tuple(
                    types.Sentence(
                        spanish=f"{tag}{i} una llave.", english=f"EN{i}", source=None
                    )
                    for i in range(count)
                ),
            )

        return triage.Candidate(
            words=(note(810, "key, faucet", "A"), note(9241, "wrestling hold", "B")),
            reason=triage.Reason.COMPATIBLE,
            comparison=definition_similarity.Comparison(0.1, frozenset()),
            disposition=triage.Disposition.REVIEW,
        )

    def test_the_grid_stays_visible_above_many_sentences(self) -> None:
        async def scenario() -> tuple[int, int, int, int]:
            app = triage_ui.TriageApp([self.wordy_candidate(15)])
            async with app.run_test(size=(140, 40)) as pilot:
                await pilot.press("s")
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                grid = app.screen.query_one("#card-grid")
                panel = app.screen.query_one("#allocation-panel")
                return (
                    grid.region.height,
                    panel.region.height,
                    panel.virtual_size.height,
                    len(app.screen.query(".allocation-row")),
                )

        grid_height, panel_height, panel_content, rows = run(scenario())
        assert rows == 30
        assert grid_height >= 7, "every card field stays on screen"
        assert panel_height <= 25, "the sentence list is a bounded panel"
        assert panel_content > panel_height, "and it scrolls rather than clips"

    def test_a_bad_allocation_scrolls_to_the_sentence(self) -> None:
        async def scenario() -> tuple[bool, str | None]:
            app = triage_ui.TriageApp([self.wordy_candidate(15)])
            async with app.run_test(size=(140, 40)) as pilot:
                await pilot.press("s")
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                for row in app.screen.query(".allocation"):
                    row.value = "1"
                bad = app.screen.query_one("#allocation-25", Input)
                bad.value = ""
                await pilot.press("ctrl+s")
                await pilot.pause()
                focused = app.screen.focused
                visible = bad.region.height > 0
                return visible, focused.id if focused else None

        visible, focused_id = run(scenario())
        assert focused_id == "allocation-25"
        assert visible, "focusing scrolled the offending row into view"


class TestSplitEditorFieldLabels:
    def test_every_grid_row_is_named(self) -> None:
        async def scenario() -> tuple[list[str], int, int]:
            app = triage_ui.TriageApp([colliding_candidate()])
            async with app.run_test(size=(140, 40)) as pilot:
                await pilot.press("s")
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                labels = [
                    str(label.render()) for label in app.screen.query(".field-label")
                ]
                rank_label = app.screen.query(".field-label").first()
                rank_input = app.screen.query_one("#rank-0", Input)
                return labels, rank_label.region.y, rank_input.region.y

        labels, label_y, input_y = run(scenario())
        assert labels == ["Rank", "Word", "POS", "Article", "Meaning"]
        # The label sits beside its row, not somewhere above or below it.
        assert abs(label_y - input_y) <= 1
