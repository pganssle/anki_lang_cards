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
    triage.Reason.COMPATIBLE: "Same headword, compatible parts of speech.",
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


class CandidateView(Static):
    """Explains why one group was flagged."""

    def __init__(self, candidate: triage.Candidate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate = candidate

    def render(self) -> str:
        candidate = self.candidate
        lines = [
            f"[bold]{_REASON_TEXT[candidate.reason]}[/]",
            f"Definitions agree: {candidate.comparison.score:.2f}",
        ]
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
        Binding("e", "edit", "Edit", show=True),
        Binding("s", "skip", "Skip", show=True),
        Binding("b", "back", "Back", show=True),
        Binding("q", "quit", "Save and quit", show=True),
    ]

    def __init__(self, candidates: Sequence[triage.Candidate]):
        super().__init__()
        self._candidates: Final = tuple(candidates)
        self._decisions: MutableMapping[int, triage.Disposition] = {}
        self._revisions: MutableMapping[int, tuple[types.Word, ...]] = {}
        self._index = 0

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
        if (revision := self._revisions.get(index)) is not None:
            changes["revisions"] = revision
        return attrs.evolve(candidate, **changes) if changes else candidate

    def _decide(self, disposition: triage.Disposition) -> None:
        self._decisions[self._index] = disposition
        self._advance()

    def _advance(self) -> None:
        if self._index + 1 >= len(self._candidates):
            self.dismiss(self.results)
            return
        self._index += 1
        self.refresh_content()

    async def action_merge(self) -> None:
        self._decide(triage.Disposition.MERGE)

    async def action_keep_separate(self) -> None:
        self._decide(triage.Disposition.KEEP_SEPARATE)

    async def action_undecided(self) -> None:
        self._decide(triage.Disposition.REVIEW)

    @textual.work
    async def action_edit(self) -> None:
        revised = await self.app.push_screen_wait(
            CandidateEditor(self.current.current_words)
        )
        if revised is None:
            return
        self._revisions[self._index] = tuple(revised)
        self.refresh_content()

    async def action_skip(self) -> None:
        self._advance()

    async def action_back(self) -> None:
        if self._index == 0:
            return
        self._index -= 1
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

    def _generate_layout_contents(self) -> Iterable[Widget]:
        candidate = self._resolved(self._index, self.current)
        words = sorted(candidate.current_words)
        reference = words[0]
        diffs = [word_merger.WordDiff.from_words(word, reference) for word in words]

        self.title = reference.word
        decided = self._decisions.get(self._index)
        position = f"{self._index + 1} of {len(self._candidates)}"
        state = [decided.value] if decided else []
        if candidate.is_revised:
            state.append("edited")
        self.sub_title = position + (f" — {', '.join(state)}" if state else "")

        yield Header()
        yield CandidateView(candidate, classes="reason")
        yield Static("\nGrouped notes:", classes="heading")
        yield Horizontal(
            *(
                word_merger.WordView(word, diff, classes="word-col")
                for word, diff in zip(words, diffs, strict=True)
            )
        )
        yield Horizontal(
            Button("Merge (m)", id="merge"),
            Button("Keep separate (k)", id="keep_separate"),
            Button("Undecided (u)", id="undecided"),
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

    CSS = """
    .heading {
        text-style: bold;
        padding: 1;
    }

    .reason {
        padding: 1;
    }

    WordView {
        border: solid green;
        padding: 1;
        margin: 1;
    }

    .word-col {
        width: 1fr;
    }

    Button {
        margin: 1;
    }
    """

    def __init__(
        self,
        candidates: Sequence[triage.Candidate],
        driver_class: type[Driver] | None = None,
    ):
        super().__init__(driver_class=driver_class)
        self.candidates = tuple(candidates)

    @textual.work
    async def on_mount(self) -> None:
        resolved = await self.push_screen_wait(TriageScreen(self.candidates))
        self.exit(resolved)


async def resolve_candidates(
    candidates: Sequence[triage.Candidate],
    driver_class: type[Driver] | None = None,
) -> Sequence[triage.Candidate]:
    """Present candidates for decision and return them with the answers applied.

    An empty list is returned unchanged rather than opening an empty UI.
    """
    candidates = tuple(candidates)
    if not candidates:
        return candidates

    app = TriageApp(candidates, driver_class=driver_class)
    resolved = await app.run_async()
    return candidates if resolved is None else tuple(resolved)
