import enum
from collections.abc import Collection, Iterable, Sequence, Set
from typing import Self

import attrs
import textual
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.driver import Driver
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import Button, Footer, Header, Input, Label, Static

from . import _types as types
from . import conflict_resolver
from . import part_of_speech as pos


class CacheHitBehavior(enum.Enum):
    SKIP = enum.auto()
    REVIEW = enum.auto()
    IGNORE = enum.auto()


@attrs.frozen
class WordDiff:
    """Represents differences between Word objects."""

    rank_diff: bool
    word_diff: bool
    meaning_diff: bool
    pos_diff: bool
    article_diff: bool
    example_count: int

    @classmethod
    def from_words(cls, word1: types.Word, word2: types.Word) -> Self:
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


class WordEditor(Screen):
    """Screen for editing a single word."""

    CSS = """
    Vertical {
        width: 1fr;
    }
    """

    BINDINGS = [
        Binding("ctrl+s", "save", "Save", show=True, priority=True),
        Binding("ctrl+x", "cancel", "Cancel", show=True, priority=True),
    ]

    def __init__(
        self, words: Sequence[types.Word], merged_word: types.Word, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.words = words
        self.merged_word = merged_word

    def action_save(self):
        rank_value = self.query_one("#rank").value
        word_value = self.query_one("#word").value
        meaning_value = self.query_one("#meaning").value
        pos_value = self.query_one("#pos").value
        article_value = self.query_one("#article").value

        try:
            self.result = attrs.evolve(
                self.merged_word,
                rank=int(rank_value) if rank_value else None,
                word=word_value,
                meaning=meaning_value,
                part_of_speech=pos_value,
                article=article_value,
            )
        except ValueError as e:
            # Reject the edit but keep the editor (and the app) alive; an
            # uncaught error here would take down the whole merge session.
            self.app.notify(f"Invalid value: {e}", severity="error", timeout=8)
            return
        self.dismiss(self.result)

    def action_cancel(self):
        self.result = None
        self.dismiss(None)

    def compose(self) -> ComposeResult:
        with Vertical():
            word = self.merged_word
            yield _get_comparative_view(self.words, word)
            yield Vertical(
                Input(str(word.rank or ""), id="rank", placeholder="Rank"),
                Input(word.word, id="word", placeholder="Word"),
                Input(word.meaning, id="meaning", placeholder="Meaning"),
                Input(
                    str(word.part_of_speech),
                    id="pos",
                    placeholder="Part of Speech",
                ),
                Input(word.article or "", id="article", placeholder="Article"),
            )
            yield Button("Save", id="save")
            yield Button("Cancel", id="cancel")
            yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            self.action_save()
        elif event.button.id == "cancel":
            self.action_cancel()


def _get_diffs(
    words: Sequence[types.Word], from_word: types.Word
) -> Sequence[WordDiff]:
    return tuple(WordDiff.from_words(w, from_word) for w in words)


def _get_comparative_view(
    words: Sequence[types.Word],
    from_word: types.Word,
) -> Container:
    diffs = _get_diffs(words, from_word)
    return Horizontal(
        *(WordView(w, d, classes="word-col") for w, d in zip(words, diffs, strict=True))
    )


class EarlyExitScreen(ModalScreen):
    """A modal exit screen."""

    DEFAULT_CSS = """
    ExitScreen {
        align: center middle;
    }

    ExitScreen > Container {
        width: auto;
        height: auto;
    }
    """

    def __init__(self, *args, msg: str = "Are you sure you want to quit?", **kwargs):
        super().__init__(*args, **kwargs)
        self.msg = msg

    def compose(self) -> ComposeResult:
        yield Label(self.msg)
        yield Button("no", id="no")
        yield Button("yes", id="yes")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            self.app.pop_screen()
        else:
            self.app.pop_screen()


class SelectWordInput(Input):
    """Input widget for word selection that shows available choices."""

    def __init__(self, word_count: int):
        super().__init__()
        self.word_count = word_count
        choices = ", ".join(f"{i+1}" for i in range(word_count))
        self.placeholder = f"Choose word ({choices}, or m for merged)"


class MergeScreen(Screen):
    """Main screen for merging words."""

    BINDINGS = [
        Binding("b", "back", "Back", show=True),
        Binding("q", "quit", "Quit", show=True),
        Binding("r", "refresh", "Refresh"),
        Binding("c", "choose_word", "Choose Word"),
        Binding("m", "edit_merge", "Edit Merge"),
        Binding("a", "accept", "Accept"),
        # These bindings are only active during word selection
        Binding("escape", "cancel_choose", "Cancel Choose", show=False),
    ]

    def __init__(
        self,
        word_sets: Sequence[Set[types.Word]],
        cache: conflict_resolver.MergeCache,
        cache_behavior: CacheHitBehavior,
    ):
        super().__init__()
        self.choosing_word = False
        self.current_idx = -1
        self.word_sets = word_sets
        self.cache = cache
        self.cache_behavior = cache_behavior
        # TODO: Allow for editing, moving back, then moving forward, etc
        self._updated_words: dict[int, types.Word] = {}
        self.next_word_or_dismiss()

    @property
    def words(self) -> Set[types.Word]:
        return self.word_sets[self.current_idx]

    def _load_word(self, idx: int) -> bool:
        self.current_idx = idx

        words = self.words
        self.title = f"Editing {next(iter(words)).word}"
        self.sub_title = f"{self.current_idx +1} of {len(self.word_sets)}"

        if (
            self.cache_behavior != CacheHitBehavior.IGNORE
            and (cached_result := self.cache.get_merge(words)) is not None
        ):
            self.merged_word = cached_result.result
            from_cache = True
        else:
            self.merged_word = auto_merge_words(words)
            from_cache = False

        return from_cache

    def next_word(self) -> None:
        while self.current_idx < len(self.word_sets) - 1:
            from_cache = self._load_word(self.current_idx + 1)
            if from_cache:
                self._updated_words[self.current_idx] = self.merged_word
                if self.cache_behavior == CacheHitBehavior.SKIP:
                    continue

            if not from_cache or self.cache_behavior == CacheHitBehavior.REVIEW:
                break
        else:
            raise StopIteration("Last word reached")

    def next_word_or_dismiss(self) -> None:
        try:
            self.next_word()
        except StopIteration:
            self.dismiss(self.results)

    def previous_word(self) -> None:
        if self.current_idx < 1:
            raise IndexError("Already at beginning of list.")
        self._load_word(self.current_idx - 1)

    @property
    def results(self) -> Sequence[types.Word | None]:
        return [
            self._updated_words.get(idx, None) for idx in range(len(self.word_sets))
        ]

    async def action_choose_word(self) -> None:
        """Allow user to choose which word's values to use."""
        if self.choosing_word:
            return

        self.choosing_word = True

        # Add selection input above footer
        input_widget = SelectWordInput(len(self.words))
        self.query_one("#root-container").mount(input_widget)
        input_widget.focus()

        # Update footer with instructions
        choices = ", ".join(f"{i+1}: {w.word}" for i, w in enumerate(self.words))
        footer = self.query_one(Footer)
        footer.highlight_key = "escape"
        self.sub_title = f"Choose word: {choices} (or 'm' for merged)"

    async def action_cancel_choose(self) -> None:
        """Cancel word selection mode."""
        if not self.choosing_word:
            return

        self.choosing_word = False

        # Remove input widget
        self.query_one(SelectWordInput).remove()

        # Restore footer
        self.sub_title = f"{self.current_idx + 1} of {len(self.word_sets)}"
        self.refresh_content()

    @textual.on(Input.Submitted)
    def handle_word_choice(self, event: Input.Submitted) -> None:
        """Handle word selection input."""
        if not self.choosing_word:
            return

        choice = event.value.strip().lower()
        if choice == "m":
            # Keep merged version - just exit selection mode
            self.merged_word = auto_merge_words(self.words)

            # Exit selection mode and refresh
            self.app.call_after_refresh(self.action_cancel_choose)
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(self.words):
                    # Update merged word with chosen word's values
                    chosen_word = list(self.words)[idx]
                    self.merged_word = attrs.evolve(
                        self.merged_word,
                        rank=chosen_word.rank,
                        word=chosen_word.word,
                        meaning=chosen_word.meaning,
                        part_of_speech=chosen_word.part_of_speech,
                        article=chosen_word.article,
                    )
                    # Exit selection mode and refresh
                    self.app.call_after_refresh(self.action_cancel_choose)
                else:
                    # Invalid number - clear input
                    event.input.value = ""
            except ValueError:
                # Invalid input - clear input
                event.input.value = ""

    async def action_back(self) -> None:
        self.previous_word()
        self.refresh_content()

    async def action_accept(self) -> None:
        self.cache.add_merge(
            conflict_resolver.MergeOperation(tuple(self.words), self.merged_word)
        )
        self.cache.save_cache()
        self._updated_words[self.current_idx] = self.merged_word
        self.next_word_or_dismiss()
        self.refresh_content()

    @textual.work
    async def action_edit_merge(self) -> None:
        word_editor = WordEditor(sorted(self.words), self.merged_word)
        await self.app.push_screen_wait(word_editor)
        if word_editor.result:
            self.merged_word = word_editor.result
            self.refresh_content()

    async def action_refresh(self) -> None:
        self.refresh_content()

    async def action_quit(self) -> None:
        self.dismiss(self.results)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "back":
                await self.action_back()
            case "choose_word":
                await self.action_choose_word()
            case "accept":
                await self.action_accept()
            case "edit_merge":
                await self.action_edit_merge()
            case "refresh":
                await self.action_refresh()
            case "quit":
                await self.action_quit()
            case _:
                raise ValueError(f"Unknown action: {event.button.id}")

    def _generate_layout_contents(self) -> Iterable[Widget]:
        if self.current_idx in self._updated_words:
            merged_classes = "accepted"
        else:
            merged_classes = ""

        yield Header()
        yield Static("Original Words:", classes="heading")
        yield _get_comparative_view(sorted(self.words), self.merged_word)
        yield Static("\nMerged Result:", classes="heading")
        yield WordView(self.merged_word, classes=merged_classes)
        yield Horizontal(
            Button("Back (b)", id="back"),
            Button("Accept Merge (a)", id="accept"),
            Button("Edit Merge (m)", id="edit_merge"),
            Button("Choose Word (c)", id="edit"),
            Button("Refresh (r)", id="refresh"),
            Button("Quit (q)", id="quit"),
        )
        yield Footer()

    def _create_layout(self) -> None:
        for w in self._generate_layout_contents():
            self._root_container.mount(w)

    def compose(self) -> ComposeResult:
        self._root_container = Container(
            *self._generate_layout_contents(), id="root-container"
        )
        yield self._root_container

    def refresh_content(self) -> None:
        self._root_container.remove_children()
        self._create_layout()
        self.refresh(layout=True)


def auto_merge_words(words: Collection[types.Word]) -> types.Word:
    """Automatically merge a sequence of words according to the specified rules."""
    # Split and clean meanings
    meanings: list[list[str]] = []
    for word in words:
        word_meanings = [m.strip() for m in word.meaning.split(",")]
        if any(w.part_of_speech.base_pos == pos.BasePoS.VERB for w in words):
            word_meanings = [m[3:] if m.startswith("to ") else m for m in word_meanings]
        meanings.append(word_meanings)

    # Calculate unique meanings maintaining order
    seen: set[str] = set()
    merged_meanings: list[str] = []

    # First, handle words with ranks in order
    ranked_words = [(w, m) for w, m in zip(words, meanings) if w.rank is not None]
    ranked_words.sort(key=lambda x: x[0].rank or float("inf"))

    for _, word_meanings in ranked_words:
        for meaning in word_meanings:
            if meaning not in seen:
                seen.add(meaning)
                merged_meanings.append(meaning)

    # Then handle unranked words in alternating fashion
    unranked_meanings = [m for w, m in zip(words, meanings) if w.rank is None]

    if unranked_meanings:
        max_len = max(len(m) for m in unranked_meanings)
        for i in range(max_len):
            for word_meanings in unranked_meanings:
                if i < len(word_meanings) and word_meanings[i] not in seen:
                    seen.add(word_meanings[i])
                    merged_meanings.append(word_meanings[i])

    # Add "to" back if it's a verb
    if any(w.part_of_speech.base_pos == pos.BasePoS.VERB for w in words):
        merged_meanings[0] = "to " + merged_meanings[0]

    # TODO: Move this to its own function, possibly on Word
    # Create merged word
    ranks = [w.rank for w in words if w.rank is not None]
    new_meaning = ", ".join(merged_meanings)
    new_rank = min(ranks) if ranks else None
    base_word = attrs.evolve(
        next(iter(words)), rank=min(ranks) if ranks else None, meaning=new_meaning
    )
    merged_words = (
        attrs.evolve(word, rank=new_rank, meaning=new_meaning) for word in words
    )
    base_word = next(merged_words)
    for word in merged_words:
        if base_word.part_of_speech != word.part_of_speech:
            base_word.part_of_speech = pos.base_part_of_speech(
                base_word.part_of_speech.base_pos
            )
            word.part_of_speech = pos.base_part_of_speech(word.part_of_speech.base_pos)
        base_word.update(word)
    return base_word


class WordMergerApp(App):
    """Main application for merging words."""

    CSS = """
    .heading {
        text-style: bold;
        padding: 1;
    }

    WordView {
        border: solid green;
        padding: 1;
        margin: 1;
    }

    WordView.accepted {
        border: thick green;
    }

    .word-col {
        width: 1fr;
    }

    Button {
        margin: 1;
    }
    """

    SCREENS = {"merge": "MergeScreen"}

    def __init__(
        self,
        word_sets: Sequence[Set[types.Word]],
        cache: conflict_resolver.MergeCache,
        cache_behavior: CacheHitBehavior,
        driver_class: type[Driver] | None = None,
    ):
        super().__init__(driver_class=driver_class)
        self.word_sets = word_sets
        self.cache_behavior = cache_behavior
        self.cache = cache

    @textual.work
    async def on_mount(self) -> None:
        resp = await self.push_screen_wait(
            MergeScreen(self.word_sets, self.cache, self.cache_behavior)
        )
        self.exit(resp)


async def merge_words(
    word_sets: Sequence[Set[types.Word]],
    cache: conflict_resolver.MergeCache,
    driver_class: type[Driver] | None = None,
    cache_behavior: CacheHitBehavior = CacheHitBehavior.REVIEW,
) -> Sequence[types.Word]:
    """
    Main entry point for merging words. This version is async-compatible.

    Args:
        words: Sequence of Word objects to merge
        cache: MergeCache representing the current cache
        driver_class: Optional Textual driver class for custom rendering

    Returns:
        The merged Word object
    """
    app = WordMergerApp(
        word_sets, cache, cache_behavior=cache_behavior, driver_class=driver_class
    )
    return await app.run_async()
