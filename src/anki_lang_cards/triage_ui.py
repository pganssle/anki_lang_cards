"""Optional Textual UI for settling triage decisions.

The triage file can be edited by hand, and for a handful of groups that is the
quickest thing. For a long list it is not, so this presents the same decisions
one at a time: what was flagged, why, how far the definitions agree, and three
keys to answer with.

This decides only what the merge UI will later be asked about. Nothing here
writes to Anki, to the deck, or to any file — the resolved candidates are
returned to the caller, which owns persistence.

Importing this module requires `textual`, the same optional dependency as
`word_merger`.
"""

import functools
import textwrap
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from typing import Final

import attrs
import textual
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.driver import Driver
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Button, Footer, Header, Input, Label, Static

from . import _types as types
from . import triage, word_merger

_REASON_TEXT: Final[Mapping[triage.Reason, str]] = {
    triage.Reason.COMPATIBLE: (
        "Same headword and compatible parts of speech, so these belong on one "
        "card. What differs is the content — usually the definition."
    ),
    triage.Reason.CHARACTERISTICS: (
        "Same headword and base part of speech, disagreeing on the details. "
        "Usually one word labelled two ways."
    ),
    triage.Reason.GENDER: (
        "Same noun, two stated genders. The gender is on the front of the card, "
        "so this may well be two words; definitions cannot settle it."
    ),
    triage.Reason.BASE_PART_OF_SPEECH: (
        "Same headword, different part of speech. English often glosses both "
        "the same way even when they are two cards."
    ),
}


_FALLBACK_PER_PAGE: Final[int] = 2
"""Sentences assumed to fit before a layout pass has said how tall the pane is."""


def describe_source(citation: types.Citation | None) -> str:
    """Name where a sentence came from, briefly.

    Which it came from matters when judging usage: a sentence lifted from a
    book is evidence about how the word is used, and one the model wrote is
    evidence about how the model thinks it is used.
    """
    if citation is None:
        return "unattributed"
    if isinstance(citation, types.BookCitation):
        pages = citation.pages or ()
        location = f" p{pages[0]}" if pages else ""
        return f"{citation.title}{location}"
    if isinstance(citation, types.ModelCitation):
        return f"model: {citation.model}"
    return citation.source_type


def _shared_rank(words: Sequence[types.Word]) -> int | None:
    """The rank every note in a group has, when they all have the same one.

    A group sharing a rank is one note read from two places — the deck and the
    import — rather than two notes that might be duplicates. The answer is
    almost always to merge, and saying so avoids the question of why an
    apparently identical pair was flagged at all.
    """
    ranks = {word.rank for word in words}
    if len(ranks) == 1 and (rank := next(iter(ranks))) is not None:
        return rank
    return None


def _sentence_block(sentence: types.Sentence, width: int) -> Sequence[str]:
    """One sentence as the lines it will actually occupy at a given width."""
    wrap = functools.partial(textwrap.wrap, width=max(width, 20))
    lines = [f"[bold]{line}[/]" for line in wrap(sentence.spanish) or [""]]
    if sentence.english:
        lines.extend(f"  {line}" for line in wrap(sentence.english, initial_indent=""))
    lines.append(f"  [dim]{describe_source(sentence.source)}[/]")
    lines.append("")
    return lines


class SentencesView(Static):
    """Shows how a note actually uses its word.

    As many sentences as the pane has room for, starting from an offset. There
    is no scrolling and nothing is hidden behind a smaller view of itself: "v"
    turns the page, so every sentence is reachable without the reader having to
    notice that the pane was cut off at the bottom.

    The offset is shared across the panes on screen so that they stay roughly in
    step, but a pane that runs out of sentences stops rather than going blank:
    it settles on its last full page and stays there while the others catch up.
    A note with one sentence and a note with ten are a normal pairing, and the
    short one going empty on the second press is no use to anyone.
    """

    # Not "offset": `Widget.offset` is a reactive holding the widget's position,
    # and assigning an int to it fails at construction.
    def __init__(
        self,
        word: types.Word,
        start: int = 0,
        per_page: int | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.word = word
        self.start = start
        self.per_page = per_page
        """Sentences to show, or None to fit as many as the pane holds."""

    def capacity(self, width: int, height: int) -> int:
        """How many whole sentences fit in a pane of this size.

        Measured from the wrapped lines rather than assumed, since a sentence
        from a book runs to several lines in a narrow column and one the model
        wrote often fits on one. At least one is always reported: a pane too
        short for even that should clip a sentence, not show nothing and leave
        "v" with nothing to advance by.
        """
        sentences = self.word.sentences
        if not sentences:
            return 0
        if self.per_page is not None:
            return max(self.per_page, 1)
        available = max(height - 1, 1)  # one line for the counter
        used = 0
        for count, sentence in enumerate(sentences):
            used += len(_sentence_block(sentence, width))
            if used > available:
                return max(count, 1)
        return len(sentences)

    def page_start(self, capacity: int) -> int:
        """Where this pane's page actually begins, once it has run out.

        Past the end it holds at the last full page rather than sliding into
        empty space, so the shared offset can keep moving for the panes that
        still have sentences left.
        """
        return max(0, min(self.start, len(self.word.sentences) - capacity))

    def render(self) -> str:
        sentences = self.word.sentences
        if not sentences:
            return "[dim]No example sentences.[/]"

        width, height = self.size.width, self.size.height
        if height <= 0 and self.per_page is None:
            # Before the first layout pass the pane has no size to measure.
            count = _FALLBACK_PER_PAGE
        else:
            count = self.capacity(width, height)

        start = self.page_start(count)
        shown = sentences[start : start + count]
        lines = [
            line for sentence in shown for line in _sentence_block(sentence, width)
        ]

        if len(sentences) > len(shown):
            first, last = start + 1, start + len(shown)
            lines.append(f"[dim]{first}–{last} of {len(sentences)} — v for more[/]")
        else:
            lines.append(f"[dim]{len(sentences)} sentences[/]")
        return "\n".join(lines)


class CandidateView(Static):
    """Explains why one group was flagged."""

    def __init__(self, candidate: triage.Candidate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate = candidate

    def render(self) -> str:
        candidate = self.candidate
        lines = [f"[bold]{_REASON_TEXT[candidate.reason]}[/]"]
        if (rank := _shared_rank(candidate.current_words)) is not None:
            lines.append(
                f"These are one note (rank {rank}) whose fields disagree, not "
                "two notes to choose between."
            )
        if len(candidate.current_sides) > 1:
            lines.append(
                "The question is only whether the two sides belong together. "
                "Any disagreement within a side is a separate entry in this "
                "list, and answering here does not settle it."
            )
        lines.append(f"Definitions agree: {candidate.comparison.score:.2f}")
        if candidate.comparison.shared_glosses:
            shared = ", ".join(sorted(candidate.comparison.shared_glosses))
            lines.append(f"Identical senses: {shared}")
        lines.append(f"Suggested: [bold]{candidate.disposition.value}[/]")
        return "\n".join(lines)


class CandidateEditor(Screen):
    """Rewrites the notes in a group so they tell each other apart.

    Choosing between two notes is not always the answer. "segundo" the
    adjective and "segundo" the noun are two cards that happen to be written
    identically; making them "segundo/a" and "el segundo" is a better outcome
    than merging or than leaving the collision in place.

    The rank is shown but not editable — it identifies the note being changed.
    """

    BINDINGS = [
        Binding("ctrl+s", "save", "Save", show=True, priority=True),
        Binding("ctrl+x", "cancel", "Cancel", show=True, priority=True),
    ]

    CSS = """
    .word-col {
        width: 1fr;
    }

    Input {
        margin-bottom: 1;
    }
    """

    def __init__(self, words: Sequence[types.Word]):
        super().__init__()
        self.words: Final = tuple(sorted(words))

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Rewrite the notes, then ctrl+s:", classes="heading")
            with Horizontal():
                for index, word in enumerate(self.words):
                    with Vertical(classes="word-col"):
                        rank = word.rank if word.rank is not None else "no rank"
                        yield Label(f"[bold]{rank}[/]")
                        yield Input(word.word, id=f"word-{index}", placeholder="Word")
                        yield Input(
                            str(word.part_of_speech),
                            id=f"pos-{index}",
                            placeholder="Part of speech",
                        )
                        yield Input(
                            word.article or "",
                            id=f"article-{index}",
                            placeholder="Article",
                        )
                        yield Input(
                            word.meaning, id=f"meaning-{index}", placeholder="Meaning"
                        )
            yield Horizontal(
                Button("Save (ctrl+s)", id="save"),
                Button("Cancel (ctrl+x)", id="cancel"),
            )
            yield Footer()

    def _value(self, field: str, index: int) -> str:
        return self.query_one(f"#{field}-{index}", Input).value.strip()

    def action_save(self) -> None:
        revised: list[types.Word] = []
        for index, word in enumerate(self.words):
            try:
                revised.append(
                    attrs.evolve(
                        word,
                        word=self._value("word", index),
                        part_of_speech=self._value("pos", index),
                        article=self._value("article", index) or None,
                        meaning=self._value("meaning", index),
                    )
                )
            except (ValueError, TypeError) as error:
                # Rejecting the edit has to keep the editor alive; an uncaught
                # error here would take down the whole session.
                self.app.notify(f"{word.word}: {error}", severity="error", timeout=8)
                return
        self.dismiss(tuple(revised))

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            self.action_save()
        else:
            self.action_cancel()


class TriageScreen(Screen):
    """Steps through candidates, one decision at a time."""

    BINDINGS = [
        Binding("m", "merge", "Merge", show=True),
        Binding("k", "keep_separate", "Keep separate", show=True),
        Binding("u", "undecided", "Undecided", show=True),
        Binding("v", "more_sentences", "More sentences", show=True),
        Binding("e", "edit", "Edit", show=True),
        Binding("s", "skip", "Skip", show=True),
        Binding("b", "back", "Back", show=True),
        Binding("q", "quit", "Save and quit", show=True),
    ]

    def __init__(
        self,
        candidates: Sequence[triage.Candidate],
        per_page: int | None = None,
    ):
        super().__init__()
        self._candidates: Final = tuple(candidates)
        self._per_page: Final = per_page
        """Sentences per page, or None to fit as many as the pane holds."""
        self._decisions: MutableMapping[int, triage.Disposition] = {}
        self._replacements: MutableMapping[int, tuple[types.Word, ...]] = {}
        self._index = 0
        self._offset = 0

    @property
    def current(self) -> triage.Candidate:
        return self._candidates[self._index]

    @property
    def results(self) -> Sequence[triage.Candidate]:
        """Every candidate, with any decision made here applied to it.

        Candidates never reached keep whatever disposition they arrived with,
        so quitting early is the same as leaving the rest of the file alone.
        """
        return tuple(
            self._resolved(index, candidate)
            for index, candidate in enumerate(self._candidates)
        )

    def _resolved(self, index: int, candidate: triage.Candidate) -> triage.Candidate:
        changes: dict[str, object] = {}
        if (decision := self._decisions.get(index)) is not None:
            changes["disposition"] = decision
        if (replacement := self._replacements.get(index)) is not None:
            changes["replacements"] = replacement
        return attrs.evolve(candidate, **changes) if changes else candidate

    def _decide(self, disposition: triage.Disposition) -> None:
        self._decisions[self._index] = disposition
        self._advance()

    def _advance(self) -> None:
        if self._index + 1 >= len(self._candidates):
            self.dismiss(self.results)
            return
        self._index += 1
        self._offset = 0
        self.refresh_content()

    async def action_merge(self) -> None:
        self._decide(triage.Disposition.MERGE)

    async def action_keep_separate(self) -> None:
        self._decide(triage.Disposition.KEEP_SEPARATE)

    async def action_undecided(self) -> None:
        self._decide(triage.Disposition.REVIEW)

    @property
    def sentence_offset(self) -> int:
        return self._offset

    def _page_size(self) -> int:
        """How many sentences the smallest pane on screen can show.

        The panes advance together so that the notes stay side by side and
        comparable, which means the page is as big as the smallest of them.
        """
        if self._per_page is not None:
            return max(self._per_page, 1)
        views = tuple(self.query(SentencesView))
        if not views:
            return _FALLBACK_PER_PAGE
        return max(min(view.capacity(*view.size) for view in views), 1)

    async def action_more_sentences(self) -> None:
        """Move every pane on a page, and start over once they are all at the end.

        The last stop is the last full page of the longest note rather than a
        page running off its end, so the final press shows a full pane instead
        of a single trailing sentence.
        """
        page = self._page_size()
        longest = max(
            (len(word.sentences) for word in self.current.current_words), default=0
        )
        final = max(0, longest - page)
        self._offset = 0 if self._offset >= final else min(self._offset + page, final)
        self.refresh_content()

    @textual.work
    async def action_edit(self) -> None:
        revised = await self.app.push_screen_wait(
            CandidateEditor(self.current.current_words)
        )
        if revised is None:
            return
        self._replacements[self._index] = tuple(revised)
        self.refresh_content()

    async def action_skip(self) -> None:
        self._advance()

    async def action_back(self) -> None:
        if self._index == 0:
            return
        self._index -= 1
        self._offset = 0
        self.refresh_content()

    async def action_quit(self) -> None:
        self.dismiss(self.results)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "merge":
                await self.action_merge()
            case "keep_separate":
                await self.action_keep_separate()
            case "undecided":
                await self.action_undecided()
            case "more_sentences":
                await self.action_more_sentences()
            case "edit":
                self.action_edit()
            case "skip":
                await self.action_skip()
            case "back":
                await self.action_back()
            case "quit":
                await self.action_quit()
            case _:
                raise ValueError(f"Unknown action: {event.button.id}")

    def _note_column(self, word: types.Word, reference: types.Word) -> Widget:
        return Vertical(
            word_merger.WordView(
                word, word_merger.WordDiff.from_words(word, reference)
            ),
            SentencesView(word, self._offset, self._per_page, classes="sentences"),
            classes="word-col",
        )

    def _generate_layout_contents(self) -> Iterable[Widget]:
        candidate = self._resolved(self._index, self.current)
        sides = tuple(tuple(sorted(side)) for side in candidate.current_sides)
        words = sorted(candidate.current_words)
        reference = words[0]

        self.title = reference.word
        decided = self._decisions.get(self._index)
        position = f"{self._index + 1} of {len(self._candidates)}"
        state = [decided.value] if decided else []
        if candidate.has_replacements:
            state.append("edited")
        self.sub_title = position + (f" — {', '.join(state)}" if state else "")

        yield Header()
        yield CandidateView(candidate, classes="reason")
        heading = "Sides:" if len(sides) > 1 else "Grouped notes:"
        yield Static(heading, classes="heading")
        if len(sides) > 1:
            panes: list[Widget] = []
            for index, side in enumerate(sides):
                if index:
                    panes.append(Static("vs", classes="divider"))
                panes.append(
                    Horizontal(
                        *(self._note_column(word, reference) for word in side),
                        classes="side",
                    )
                )
            yield Horizontal(*panes, classes="group")
        else:
            yield Horizontal(
                *(self._note_column(word, reference) for word in words),
                classes="group",
            )
        yield Horizontal(
            Button("Merge (m)", id="merge"),
            Button("Keep separate (k)", id="keep_separate"),
            Button("Undecided (u)", id="undecided"),
            Button("More sentences (v)", id="more_sentences"),
            Button("Edit (e)", id="edit"),
            Button("Skip (s)", id="skip"),
            Button("Back (b)", id="back"),
            Button("Save and quit (q)", id="quit"),
        )
        yield Footer()

    def compose(self) -> ComposeResult:
        self._root_container = Container(
            *self._generate_layout_contents(), id="root-container"
        )
        yield self._root_container

    def refresh_content(self) -> None:
        self._root_container.remove_children()
        for widget in self._generate_layout_contents():
            self._root_container.mount(widget)
        self.refresh(layout=True)


class TriageApp(App):
    """Main application for resolving triage decisions."""

    # The sentences are the reason this screen exists, so the chrome is kept
    # thin: at a 30-row terminal, generous padding on everything else left the
    # sentence pane with no rows at all and the reader seeing two sentences by
    # accident rather than by fit.
    CSS = """
    .heading {
        text-style: bold;
        padding: 0 1;
    }

    .reason {
        padding: 0 1;
    }

    WordView {
        border: solid green;
        padding: 0 1;
        margin: 0 1;
    }

    .word-col {
        width: 1fr;
        height: 1fr;
    }

    .group {
        height: 1fr;
    }

    .side {
        width: 1fr;
        height: 1fr;
        border: round $panel;
    }

    .divider {
        width: 4;
        height: 1fr;
        content-align: center middle;
        color: $text-muted;
    }

    .sentences {
        padding: 0 2;
        height: 1fr;
        min-height: 5;
        overflow: hidden;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        candidates: Sequence[triage.Candidate],
        driver_class: type[Driver] | None = None,
        per_page: int | None = None,
    ):
        super().__init__(driver_class=driver_class)
        self.candidates = tuple(candidates)
        self.per_page = per_page

    @textual.work
    async def on_mount(self) -> None:
        resolved = await self.push_screen_wait(
            TriageScreen(self.candidates, self.per_page)
        )
        self.exit(resolved)


async def resolve_candidates(
    candidates: Sequence[triage.Candidate],
    driver_class: type[Driver] | None = None,
    per_page: int | None = None,
) -> Sequence[triage.Candidate]:
    """Present candidates for decision and return them with the answers applied.

    An empty list is returned unchanged rather than opening an empty UI.
    """
    candidates = tuple(candidates)
    if not candidates:
        return candidates

    app = TriageApp(candidates, driver_class=driver_class, per_page=per_page)
    resolved = await app.run_async()
    return candidates if resolved is None else tuple(resolved)
