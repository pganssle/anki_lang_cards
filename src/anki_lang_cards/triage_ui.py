"""Optional Textual UI for finding and merging duplicates.

The decision file can be edited by hand, and for a handful of groups that is
the quickest thing. For a long list it is not, so this presents each flagged
group once, with everything needed to answer in one sitting: what was flagged,
why, how far the definitions agree, the sentences, and the proposed merged
note. Accepting records both the answer and the wording; there is no second
pass through a separate merge UI.

Nothing here writes to Anki, to the deck, or to any file — the resolved
candidates are returned to the caller (and reported through `on_decision` as
each answer is given), and the caller owns persistence.

Importing this module requires `textual`, the `merge_ui` optional extra.
"""

import functools
import textwrap
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from typing import Final

import attrs
import textual
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.driver import Driver
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import Button, Footer, Header, Input, Label, Static

from . import _types as types
from . import apply_merges, conflict_resolver, triage


@attrs.frozen
class WordDiff:
    """Which fields differ between two words, for highlighting."""

    rank_diff: bool
    word_diff: bool
    meaning_diff: bool
    pos_diff: bool
    article_diff: bool
    example_count: int

    @classmethod
    def from_words(cls, word1: types.Word, word2: types.Word) -> "WordDiff":
        return cls(
            rank_diff=word1.rank != word2.rank,
            word_diff=word1.word != word2.word,
            meaning_diff=word1.meaning != word2.meaning,
            pos_diff=word1.part_of_speech != word2.part_of_speech,
            article_diff=word1.article != word2.article,
            example_count=len(word1.sentences) + len(word2.sentences),
        )


class WordView(Static):
    """Displays a single word with differences highlighted."""

    def __init__(self, word: types.Word, diff: WordDiff | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.word = word
        self.diff = diff

    def render(self) -> str:
        def style_text(text: str, is_diff: bool) -> str:
            return f"[bold red]{text}[/]" if is_diff else text

        rank_text = style_text(
            f"Rank: {self.word.rank or 'None'}", bool(self.diff and self.diff.rank_diff)
        )
        word_text = style_text(
            f"Word: {self.word.word}", bool(self.diff and self.diff.word_diff)
        )
        meaning_text = style_text(
            f"Meaning: {self.word.meaning}", bool(self.diff and self.diff.meaning_diff)
        )
        pos_text = style_text(
            f"POS: {self.word.part_of_speech}", bool(self.diff and self.diff.pos_diff)
        )
        article_text = style_text(
            f"Article: {self.word.article or 'None'}",
            bool(self.diff and self.diff.article_diff),
        )
        examples_text = f"Examples: {len(self.word.sentences)}"

        return (
            f"{rank_text}\n"
            f"{word_text}\n"
            f"{meaning_text}\n"
            f"{pos_text}\n"
            f"{article_text}\n"
            f"{examples_text}"
        )


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


_shared_rank = apply_merges.shared_rank


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


propose_merge = apply_merges.propose_merge
"""Re-exported: the proposal shown on screen is the one the file protocol
writes, so it lives beside the merge machinery rather than behind textual."""


def still_collides(words: Sequence[types.Word]) -> bool:
    """Whether these notes would still land on one card as written.

    "Keep separate" is only a real option when the notes can exist apart —
    different genders, different parts of speech, or spellings that were edited
    apart. Notes that still share a lookup group would go straight back onto
    the duplicate list, and the reviewer deserves to be told before recording
    an answer that changes nothing.
    """
    lookup = conflict_resolver.WordLookup(words)
    return any(len(group) > 1 for group in lookup)


class ConfirmScreen(ModalScreen[bool]):
    """A yes/no question that must not be answered by accident."""

    BINDINGS = [
        Binding("y", "confirm", "Yes", show=True),
        Binding("n,escape", "deny", "No", show=True),
    ]

    CSS = """
    ConfirmScreen {
        align: center middle;
    }

    #confirm-box {
        width: 60;
        height: auto;
        border: thick $warning;
        padding: 1 2;
    }
    """

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Static(self.message)
            yield Horizontal(
                Button("Yes (y)", id="confirm"),
                Button("No (n)", id="deny"),
            )

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")


_PULLABLE_FIELDS: Final[Mapping[str, Callable[[types.Word], str]]] = {
    "rank": lambda w: str(w.rank) if w.rank is not None else "",
    "word": lambda w: w.word,
    "pos": lambda w: str(w.part_of_speech),
    "article": lambda w: w.article or "",
    "meaning": lambda w: w.meaning,
}


class MergedNoteEditor(Screen):
    """Writes the note a merge will produce.

    The numbered source notes stay on screen, because the point of editing is
    almost always to take the best parts of each: ctrl+p pulls one field from
    one of them by number — "2" copies the focused field from note 2, and
    "2 meaning" names the field instead. Unlike the group editor, the rank is
    editable here; this is the one place an explicit rank choice overrides
    "lowest wins".
    """

    BINDINGS = [
        Binding("ctrl+s", "save", "Save", show=True, priority=True),
        Binding("ctrl+x", "cancel", "Cancel", show=True, priority=True),
        Binding("ctrl+p", "pull", "Pull field from note #", show=True, priority=True),
    ]

    CSS = """
    .word-col {
        width: 1fr;
    }

    Input {
        margin-bottom: 1;
    }

    #pull-input {
        border: tall $accent;
    }
    """

    def __init__(self, sources: Sequence[types.Word], merged: types.Word):
        super().__init__()
        self.sources: Final = tuple(sorted(sources))
        self.merged: Final = merged
        self._pull_target: str | None = None

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static("Source notes:", classes="heading")
            with Horizontal():
                for number, word in enumerate(self.sources, start=1):
                    with Vertical(classes="word-col"):
                        yield Label(f"[bold]#{number}[/]")
                        yield WordView(word, WordDiff.from_words(word, self.merged))
            yield Static("Merged note, then ctrl+s:", classes="heading")
            rank = self.merged.rank
            yield Input(
                str(rank) if rank is not None else "", id="rank", placeholder="Rank"
            )
            yield Input(self.merged.word, id="word", placeholder="Word")
            yield Input(
                str(self.merged.part_of_speech), id="pos", placeholder="Part of speech"
            )
            yield Input(self.merged.article or "", id="article", placeholder="Article")
            yield Input(self.merged.meaning, id="meaning", placeholder="Meaning")
            yield Horizontal(
                Button("Save (ctrl+s)", id="save"),
                Button("Cancel (ctrl+x)", id="cancel"),
            )
            yield Footer()

    def action_save(self) -> None:
        rank_text = self.query_one("#rank", Input).value.strip()
        try:
            revised = attrs.evolve(
                self.merged,
                rank=int(rank_text) if rank_text else None,
                word=self.query_one("#word", Input).value.strip(),
                part_of_speech=self.query_one("#pos", Input).value.strip(),
                article=self.query_one("#article", Input).value.strip() or None,
                meaning=self.query_one("#meaning", Input).value.strip(),
            )
        except (ValueError, TypeError) as error:
            # Rejecting the edit has to keep the editor alive; an uncaught
            # error here would take down the whole session.
            self.app.notify(f"{error}", severity="error", timeout=8)
            return
        self.dismiss(revised)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_pull(self) -> None:
        if self.query("#pull-input"):
            return
        focused = self.focused
        self._pull_target = (
            focused.id if isinstance(focused, Input) and focused.id else None
        )
        prompt = Input(
            id="pull-input",
            placeholder="Pull from note # (e.g. '2', or '2 meaning' to name the field)",
        )
        self.mount(prompt)
        prompt.focus()

    @textual.on(Input.Submitted, "#pull-input")
    def handle_pull(self, event: Input.Submitted) -> None:
        event.input.remove()
        parts = event.value.split()
        field = self._pull_target
        if len(parts) == 2:
            field = parts[1].casefold()
        if len(parts) not in (1, 2) or not parts[0].isdigit():
            self.app.notify("Say which note: '2', or '2 meaning'.", severity="warning")
            return
        number = int(parts[0])
        if not 1 <= number <= len(self.sources):
            self.app.notify(
                f"Note numbers run 1–{len(self.sources)}.", severity="warning"
            )
            return
        if field not in _PULLABLE_FIELDS:
            self.app.notify(
                "Name a field to pull: rank, word, pos, article or meaning.",
                severity="warning",
            )
            return
        target = self.query_one(f"#{field}", Input)
        target.value = _PULLABLE_FIELDS[field](self.sources[number - 1])
        target.focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            self.action_save()
        elif event.button.id == "cancel":
            self.action_cancel()


class SplitEditor(Screen):
    """Divides one group into several new cards.

    "llave" defined as "key, faucet" beside "wrestling hold" is not two
    wordings of one card, and merging is not the only alternative to choosing:
    it can become two or three cards, each with its own fields and its own
    sentences. The card grid prefills from the group (extra cards start empty
    and rank-less), shift+right copies the focused field onto the next card,
    and every sentence is allocated to a card by number — comma-separated to
    share one between several.
    """

    BINDINGS = [
        Binding("ctrl+s", "save", "Save", show=True, priority=True),
        Binding("ctrl+x", "cancel", "Cancel", show=True, priority=True),
        # Priority: the Input would otherwise consume shift+right for text
        # selection, which nobody needs in a five-word field.
        Binding(
            "shift+right",
            "copy_right",
            "Copy field to next card",
            show=True,
            priority=True,
        ),
    ]

    # The grid must always be fully visible — it is what the reviewer is
    # writing — so everything gets its natural height and only the sentence
    # list scrolls, inside its own bounded panel.
    CSS = """
    #card-grid {
        height: auto;
    }

    .card-col {
        width: 1fr;
        height: auto;
    }

    .field-labels {
        width: 9;
        height: auto;
    }

    .field-label {
        height: 3;
        padding: 1 1 0 0;
        width: 100%;
        text-align: right;
        color: $text-muted;
    }

    .field-label-spacer {
        height: 1;
    }

    Input {
        margin-bottom: 0;
    }

    #allocation-panel {
        height: auto;
        max-height: 25;
        border: round $panel;
    }

    .allocation-row {
        height: auto;
    }

    .allocation {
        /* Wide enough for a shared sentence's whole answer — "1,2,3,4" —
           after the Input's own border and padding take four cells. */
        width: 14;
    }

    .sentence-number {
        width: 5;
        padding: 1 1 0 0;
        text-align: right;
        color: $text-muted;
    }

    .allocation-origin {
        width: 10;
        padding: 1 1 0 1;
    }

    .sentence-text {
        width: 1fr;
        padding: 1 0 0 0;
    }

    #split-buttons {
        height: auto;
    }

    #count-input {
        width: 20;
    }
    """

    def __init__(
        self,
        words: Sequence[types.Word],
        proposal: types.Word | None,
        count: int | None = None,
    ):
        super().__init__()
        self.words: Final = tuple(sorted(words))
        self.proposal: Final = proposal
        self.count = count
        self.sentences: Final[tuple[types.Sentence, ...]] = tuple(
            self._sentence_union()
        )

    def _sentence_union(self) -> Iterable[types.Sentence]:
        seen: set[types.Sentence] = set()
        for word in self.words:
            for sentence in word.sentences:
                if sentence not in seen:
                    seen.add(sentence)
                    yield sentence

    def _origin_note(self, sentence: types.Sentence) -> int | None:
        """Which numbered note a sentence came from, for the row's label.

        Shown rather than prefilled: where the sentence lived is evidence for
        the allocation, not the answer to it. The numbers match the ones the
        candidate screen put on the notes.
        """
        for position, word in enumerate(self.words):
            if sentence in word.sentences:
                return position + 1
        return None

    def _prefill(self, index: int) -> types.Word | None:
        if index == 0 and self.proposal is not None and self.count != len(self.words):
            return self.proposal
        if index < len(self.words):
            return self.words[index]
        return None

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="split-root"):
            if self.count is None:
                yield Static("Split into how many cards?", classes="heading")
                yield Input(
                    str(len(self.words)),
                    id="count-input",
                    placeholder="Number of cards",
                )
            else:
                yield from self._grid()
            yield Footer()

    def on_mount(self) -> None:
        if self.count is None:
            self.query_one("#count-input", Input).focus()

    @textual.on(Input.Submitted, "#count-input")
    async def handle_count(self, event: Input.Submitted) -> None:
        try:
            count = int(event.value)
        except ValueError:
            self.app.notify("A number of cards, e.g. 2.", severity="warning")
            return
        if count < 2:
            self.app.notify("A split makes at least two cards.", severity="warning")
            return
        self.count = count
        event.input.remove()
        root = self.query_one("#split-root", VerticalScroll)
        for heading in self.query(".heading"):
            heading.remove()
        await root.mount(*self._grid())
        self.query_one("#word-0", Input).focus()

    def _grid(self) -> Iterable[Widget]:
        assert self.count is not None
        yield Static(
            f"{self.count} cards, then ctrl+s. shift+right copies a field "
            "to the next card.",
            classes="heading",
        )
        # Placeholders vanish as soon as a field is filled, so the field
        # names live in their own label column, row by row.
        field_labels = Vertical(
            Label("", classes="field-label-spacer"),
            *(
                Label(name, classes="field-label")
                for name in ("Rank", "Word", "POS", "Article", "Meaning")
            ),
            classes="field-labels",
        )
        columns: list[Widget] = [field_labels]
        for index in range(self.count):
            prefill = self._prefill(index)
            rank = prefill.rank if prefill is not None else None
            columns.append(
                Vertical(
                    Label(f"[bold]card {index + 1}[/]"),
                    Input(
                        str(rank) if rank is not None else "",
                        id=f"rank-{index}",
                        placeholder="Rank",
                    ),
                    Input(
                        prefill.word if prefill else "",
                        id=f"word-{index}",
                        placeholder="Word",
                    ),
                    Input(
                        str(prefill.part_of_speech) if prefill else "",
                        id=f"pos-{index}",
                        placeholder="Part of speech",
                    ),
                    Input(
                        (prefill.article or "") if prefill else "",
                        id=f"article-{index}",
                        placeholder="Article",
                    ),
                    Input(
                        prefill.meaning if prefill else "",
                        id=f"meaning-{index}",
                        placeholder="Meaning",
                    ),
                    classes="card-col",
                )
            )
        yield Horizontal(*columns, id="card-grid")
        if self.sentences:
            yield Static(
                f"Allocate each of the {len(self.sentences)} sentences to a "
                "card (1, 2, or 1,2 to share); every one needs an answer:",
                classes="heading",
            )
            yield VerticalScroll(
                *(
                    Horizontal(
                        Label(f"{position + 1}.", classes="sentence-number"),
                        Input(
                            "",
                            id=f"allocation-{position}",
                            classes="allocation",
                            placeholder="card #",
                        ),
                        Label(
                            (
                                f"[dim]from #{origin}[/]"
                                if (origin := self._origin_note(sentence)) is not None
                                else ""
                            ),
                            classes="allocation-origin",
                        ),
                        Static(sentence.spanish, classes="sentence-text"),
                        classes="allocation-row",
                    )
                    for position, sentence in enumerate(self.sentences)
                ),
                id="allocation-panel",
            )
        yield Horizontal(
            Button("Save (ctrl+s)", id="save"),
            Button("Cancel (ctrl+x)", id="cancel"),
            id="split-buttons",
        )

    def action_copy_right(self) -> None:
        focused = self.focused
        if not isinstance(focused, Input) or not focused.id:
            return
        field, _, index_text = focused.id.rpartition("-")
        if field not in _PULLABLE_FIELDS or not index_text.isdigit():
            return
        assert self.count is not None
        next_index = int(index_text) + 1
        if next_index >= self.count:
            return
        target = self.query_one(f"#{field}-{next_index}", Input)
        target.value = focused.value
        target.focus()

    def _card_allocations(self) -> Sequence[frozenset[int]] | None:
        """Card numbers for each sentence, or None after telling the user why."""
        assert self.count is not None
        allocations: list[frozenset[int]] = []
        for position in range(len(self.sentences)):
            raw = self.query_one(f"#allocation-{position}", Input).value
            tokens = [token.strip() for token in raw.split(",") if token.strip()]
            if not tokens or not all(token.isdigit() for token in tokens):
                self.app.notify(
                    f"Sentence {position + 1}: allocate it to a card number "
                    "(1, 2, or 1,2).",
                    severity="warning",
                )
                self._focus_allocation(position)
                return None
            numbers = frozenset(int(token) for token in tokens)
            if not all(1 <= number <= self.count for number in numbers):
                self.app.notify(
                    f"Sentence {position + 1}: card numbers run 1–{self.count}.",
                    severity="warning",
                )
                self._focus_allocation(position)
                return None
            allocations.append(numbers)
        return allocations

    def _focus_allocation(self, position: int) -> None:
        """Put the cursor on the offending sentence; focusing scrolls to it."""
        self.query_one(f"#allocation-{position}", Input).focus()

    def action_save(self) -> None:
        if self.count is None:
            self.app.notify("Choose how many cards first.", severity="warning")
            return
        allocations = self._card_allocations()
        if allocations is None:
            return
        template = self.words[0]
        cards: list[types.Word] = []
        for index in range(self.count):
            rank_text = self.query_one(f"#rank-{index}", Input).value.strip()
            word_text = self.query_one(f"#word-{index}", Input).value.strip()
            meaning = self.query_one(f"#meaning-{index}", Input).value.strip()
            if not word_text or not meaning:
                self.app.notify(
                    f"Card {index + 1} needs a word and a meaning.",
                    severity="warning",
                )
                return
            sentences = tuple(
                sentence
                for position, sentence in enumerate(self.sentences)
                if index + 1 in allocations[position]
            )
            try:
                cards.append(
                    attrs.evolve(
                        template,
                        rank=int(rank_text) if rank_text else None,
                        word=word_text,
                        part_of_speech=self.query_one(
                            f"#pos-{index}", Input
                        ).value.strip(),
                        article=self.query_one(f"#article-{index}", Input).value.strip()
                        or None,
                        meaning=meaning,
                        sentences=sentences,
                    )
                )
            except (ValueError, TypeError) as error:
                self.app.notify(
                    f"Card {index + 1}: {error}", severity="error", timeout=8
                )
                return
        self.dismiss(tuple(cards))

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            self.action_save()
        elif event.button.id == "cancel":
            self.action_cancel()


class ChooseNoteInput(Input):
    """Asks which numbered note to take as the base of the merged result."""

    def __init__(self, count: int, *args, **kwargs):
        super().__init__(
            *args, placeholder=f"Use note # (1–{count}) as the base", **kwargs
        )


class TriageScreen(Screen):
    """Steps through candidates, one answer at a time.

    An answer settles both whether the notes merge and what the result says:
    accepting records the proposed note shown on screen, editing (m), choosing
    a base note (c) and splitting (s) all shape what gets recorded. Every
    answer is reported through `on_decision` the moment it is given, so
    quitting — or crashing — loses nothing already decided.
    """

    BINDINGS = [
        Binding("a", "accept", "Accept merge", show=True),
        Binding("k", "keep_separate", "Keep separate", show=True),
        Binding("m", "edit_merged", "Edit merged", show=True),
        Binding("c", "choose_note", "Choose base note", show=True),
        Binding("s", "split", "Split", show=True),
        Binding("e", "edit", "Edit notes", show=True),
        Binding("u", "undecided", "Undecided", show=True),
        Binding("v", "more_sentences", "More sentences", show=True),
        Binding("n", "skip", "Next", show=True),
        Binding("b", "back", "Back", show=True),
        Binding("q", "quit", "Save and quit", show=True),
    ]

    def __init__(
        self,
        candidates: Sequence[triage.Candidate],
        per_page: int | None = None,
        on_decision: Callable[[triage.Candidate], None] | None = None,
    ):
        super().__init__()
        self._candidates: Final = tuple(candidates)
        self._per_page: Final = per_page
        """Sentences per page, or None to fit as many as the pane holds."""
        self._on_decision: Final = on_decision
        self._decisions: MutableMapping[int, triage.Disposition] = {}
        self._replacements: MutableMapping[int, tuple[types.Word, ...]] = {}
        self._proposals: MutableMapping[int, types.Word] = {}
        """Merged-note proposals as shaped by editing or choosing a base."""
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

    def _notify(self, index: int) -> None:
        if self._on_decision is not None:
            self._on_decision(self._resolved(index, self._candidates[index]))

    def _decide(self, disposition: triage.Disposition) -> None:
        index = self._index
        if disposition in (
            triage.Disposition.KEEP_SEPARATE,
            triage.Disposition.REVIEW,
        ):
            # These answers only carry note-for-note edits. A merged note or
            # split cards left over from an earlier answer to the same group
            # would be recorded in a shape the disposition cannot mean.
            replacements = self._replacements.get(index)
            if replacements is not None and len(replacements) != len(
                self._candidates[index].words
            ):
                del self._replacements[index]
        self._decisions[index] = disposition
        self._notify(index)
        self._advance()

    def _advance(self) -> None:
        if self._index + 1 >= len(self._candidates):
            self.dismiss(self.results)
            return
        self._index += 1
        self._offset = 0
        self.refresh_content()

    def _editable_words(self) -> tuple[types.Word, ...]:
        """The group as it currently stands, note for note.

        Replacements in another shape — a recorded merged note, split cards —
        do not describe the notes themselves, so the originals are used.
        """
        candidate = self._resolved(self._index, self.current)
        current = candidate.current_words
        if len(current) == len(candidate.words):
            return tuple(sorted(current))
        return tuple(sorted(candidate.words))

    def _display_candidate(self) -> triage.Candidate:
        """The candidate as the screen lays it out: always the group itself.

        A recorded merged note or split cards replace the group in the data
        model, but the screen keeps showing the notes being asked about — the
        recorded answer appears in the proposal pane and the subtitle, not as
        a rewriting of the question.
        """
        candidate = self._resolved(self._index, self.current)
        if candidate.replacements is not None and len(candidate.replacements) != len(
            candidate.words
        ):
            return attrs.evolve(candidate, replacements=None)
        return candidate

    def _current_proposal(self) -> types.Word | None:
        if (proposal := self._proposals.get(self._index)) is not None:
            return proposal
        candidate = self._resolved(self._index, self.current)
        if (
            candidate.replacements is not None
            and len(candidate.replacements) == 1
            and len(candidate.words) > 1
        ):
            # The answer already recorded for this group, shown when revisiting.
            return candidate.replacements[0]
        return propose_merge(self._editable_words())

    async def action_accept(self) -> None:
        proposal = self._current_proposal()
        if proposal is None:
            self.app.notify(
                "No mechanical merge for this group — write one with m first.",
                severity="warning",
            )
            return
        self._replacements[self._index] = (proposal,)
        self._decide(triage.Disposition.MERGE)

    async def action_keep_separate(self) -> None:
        # A sides candidate asks whether two lookup groups belong together;
        # the groups can coexist by construction, so keeping them apart needs
        # no warning — any collision inside a side is its own candidate.
        candidate = self._resolved(self._index, self.current)
        if candidate.sides or not still_collides(self._editable_words()):
            self._decide(triage.Disposition.KEEP_SEPARATE)
            return
        self._confirm_keep_separate()

    @textual.work
    async def _confirm_keep_separate(self) -> None:
        confirmed = await self.app.push_screen_wait(
            ConfirmScreen(
                "These notes will still collide as written — they land on "
                "one card, and this group will be flagged again. Edit (e) "
                "or split (s) to tell them apart.\n\nKeep separate anyway?"
            )
        )
        if confirmed:
            self._decide(triage.Disposition.KEEP_SEPARATE)

    async def action_undecided(self) -> None:
        self._decide(triage.Disposition.REVIEW)

    @textual.work
    async def action_edit_merged(self) -> None:
        words = self._editable_words()
        proposal = self._current_proposal()
        if proposal is None:
            # No mechanical proposal; start from the leading note so the
            # editor opens with something rather than a blank form.
            ranks = [w.rank for w in words if w.rank is not None]
            proposal = attrs.evolve(words[0], rank=min(ranks) if ranks else None)
        revised = await self.app.push_screen_wait(MergedNoteEditor(words, proposal))
        if revised is None:
            return
        self._proposals[self._index] = revised
        self.refresh_content()

    @textual.work
    async def action_split(self) -> None:
        words = self._editable_words()
        cards = await self.app.push_screen_wait(
            SplitEditor(words, self._current_proposal())
        )
        if cards is None:
            return
        self._replacements[self._index] = tuple(cards)
        self._decide(triage.Disposition.SPLIT)

    async def action_choose_note(self) -> None:
        if self.query(ChooseNoteInput):
            return
        prompt = ChooseNoteInput(len(self._editable_words()))
        await self.mount(prompt)
        prompt.focus()

    @textual.on(Input.Submitted)
    def handle_choose_note(self, event: Input.Submitted) -> None:
        if not isinstance(event.input, ChooseNoteInput):
            return
        event.input.remove()
        # A removed widget can keep keyboard focus, and then the next
        # keystroke goes to nothing instead of the screen's bindings.
        self.set_focus(None)
        words = self._editable_words()
        if not event.value.isdigit() or not 1 <= int(event.value) <= len(words):
            self.app.notify(f"Note numbers run 1–{len(words)}.", severity="warning")
            return
        chosen = words[int(event.value) - 1]
        fallback = self._current_proposal()
        ranks = [word.rank for word in words if word.rank is not None]
        # Choosing a base must not un-fill fields the chosen note happens to
        # lack: a missing article stays whatever the proposal had, and the
        # rank is always the group's lowest short of an explicit edit.
        self._proposals[self._index] = attrs.evolve(
            chosen,
            rank=min(ranks) if ranks else chosen.rank,
            article=(
                chosen.article
                if chosen.article is not None
                else (fallback.article if fallback is not None else None)
            ),
            sentences=(
                fallback.sentences if fallback is not None else chosen.sentences
            ),
        )
        self.refresh_content()

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
            CandidateEditor(self._editable_words())
        )
        if revised is None:
            return
        self._replacements[self._index] = tuple(revised)
        # An edit is work worth keeping even before the group is decided, so
        # it is reported now rather than when (or if) an answer follows.
        self._notify(self._index)
        # The mechanical proposal was built from the old text.
        self._proposals.pop(self._index, None)
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
            case "accept":
                await self.action_accept()
            case "keep_separate":
                await self.action_keep_separate()
            case "edit_merged":
                self.action_edit_merged()
            case "choose_note":
                await self.action_choose_note()
            case "split":
                self.action_split()
            case "edit":
                self.action_edit()
            case "undecided":
                await self.action_undecided()
            case "more_sentences":
                await self.action_more_sentences()
            case "skip":
                await self.action_skip()
            case "back":
                await self.action_back()
            case "quit":
                await self.action_quit()
            case _:
                raise ValueError(f"Unknown action: {event.button.id}")

    def _note_column(
        self, number: int, word: types.Word, reference: types.Word
    ) -> Widget:
        return Vertical(
            Label(f"[bold]#{number}[/]", classes="note-number"),
            WordView(word, WordDiff.from_words(word, reference)),
            SentencesView(word, self._offset, self._per_page, classes="sentences"),
            classes="word-col",
        )

    def _numbering(self, ordered: Sequence[types.Word]) -> Callable[[types.Word], int]:
        """Numbers notes the way c and ctrl+p refer to them: sorted order.

        Identity comes first because a group can hold two equal notes — the
        deck copy and the import copy — and equality would give them one
        number between them.
        """

        def number(word: types.Word) -> int:
            for position, candidate_word in enumerate(ordered):
                if candidate_word is word:
                    return position + 1
            return ordered.index(word) + 1

        return number

    def _generate_layout_contents(self) -> Iterable[Widget]:
        resolved = self._resolved(self._index, self.current)
        candidate = self._display_candidate()
        sides = tuple(tuple(sorted(side)) for side in candidate.current_sides)
        words = sorted(candidate.current_words)
        reference = words[0]
        number = self._numbering(self._editable_words())

        self.title = reference.word
        decided = self._decisions.get(self._index)
        position = f"{self._index + 1} of {len(self._candidates)}"
        state = [decided.value] if decided else []
        if resolved.has_replacements:
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
                        *(
                            self._note_column(number(word), word, reference)
                            for word in side
                        ),
                        classes="side",
                    )
                )
            yield Horizontal(*panes, classes="group")
        else:
            yield Horizontal(
                *(self._note_column(number(word), word, reference) for word in words),
                classes="group",
            )
        proposal = self._current_proposal()
        yield Static("Proposed result (a to accept):", classes="heading")
        if proposal is not None:
            yield WordView(
                proposal,
                WordDiff.from_words(proposal, reference),
                classes="proposal",
            )
        else:
            yield Static(
                "[dim]No mechanical merge for this group — m writes one, "
                "c starts from a note.[/]",
                classes="proposal",
            )
        yield Horizontal(
            Button("Accept (a)", id="accept"),
            Button("Keep separate (k)", id="keep_separate"),
            Button("Edit merged (m)", id="edit_merged"),
            Button("Choose note (c)", id="choose_note"),
            Button("Split (s)", id="split"),
            Button("Edit notes (e)", id="edit"),
            Button("Undecided (u)", id="undecided"),
            Button("Next (n)", id="skip"),
            Button("Back (b)", id="back"),
            Button("Quit (q)", id="quit"),
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

    # ctrl+p belongs to the merged-note editor's pull-field prompt; the
    # command palette would otherwise shadow it.
    ENABLE_COMMAND_PALETTE = False

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

    .note-number {
        padding: 0 1;
    }

    .proposal {
        border: solid $accent;
        padding: 0 1;
        margin: 0 1;
        height: auto;
    }

    ChooseNoteInput {
        border: tall $accent;
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
        on_decision: Callable[[triage.Candidate], None] | None = None,
    ):
        super().__init__(driver_class=driver_class)
        self.candidates = tuple(candidates)
        self.per_page = per_page
        self.on_decision = on_decision

    @textual.work
    async def on_mount(self) -> None:
        resolved = await self.push_screen_wait(
            TriageScreen(self.candidates, self.per_page, self.on_decision)
        )
        self.exit(resolved)


async def resolve_candidates(
    candidates: Sequence[triage.Candidate],
    driver_class: type[Driver] | None = None,
    per_page: int | None = None,
    on_decision: Callable[[triage.Candidate], None] | None = None,
) -> Sequence[triage.Candidate]:
    """Present candidates for decision and return them with the answers applied.

    `on_decision` hears each answer (and each edit) the moment it is given,
    with the resolved candidate. That is the interruption story: a caller that
    persists from the callback loses nothing when the session stops early,
    however it stops. The callback runs on the UI loop, so it should stay
    brief — writing a small file is fine.

    An empty list is returned unchanged rather than opening an empty UI.
    """
    candidates = tuple(candidates)
    if not candidates:
        return candidates

    app = TriageApp(
        candidates,
        driver_class=driver_class,
        per_page=per_page,
        on_decision=on_decision,
    )
    resolved = await app.run_async()
    return candidates if resolved is None else tuple(resolved)
