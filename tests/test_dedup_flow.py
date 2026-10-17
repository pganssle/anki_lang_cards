"""Tests for the unified deduplication flow.

The flow itself is UI-free; interactive scenarios substitute a scripted
answerer for the real screen, which also documents the contract the screen
has to meet — answer via `on_decision`, return the candidates.
"""

import asyncio
import json
import pathlib
from collections.abc import Awaitable, Sequence

import attrs
import pytest

from anki_lang_cards import _types as types
from anki_lang_cards import apply_merges, conflict_resolver, dedup_flow, triage


def run[T](coroutine: Awaitable[T]) -> T:
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def make_word(
    word: str,
    part_of_speech: str,
    meaning: str,
    rank: int | None,
    sentences: Sequence[types.Sentence] = (),
) -> types.Word:
    return types.Word(
        rank=rank,
        word=word,
        meaning=meaning,
        part_of_speech=part_of_speech,
        article=None,
        sentences=tuple(sentences),
    )


@pytest.fixture
def decisions(tmp_path: pathlib.Path) -> triage.DecisionCache:
    return triage.DecisionCache(tmp_path / "decisions.json")


def resolve(**kwargs) -> dedup_flow.DedupResult:
    return run(dedup_flow.resolve_duplicates(**kwargs))


class TestHeadless:
    def test_no_duplicates_changes_nothing(self, decisions) -> None:
        deck = [make_word("perro", "nm", "dog", 1)]
        added = [make_word("gato", "nm", "cat", 2)]

        result = resolve(
            inputs=added, existing=deck, decisions=decisions, interactive=False
        )

        assert not result.interrupted
        assert {w.word for w in result.outcome.words} == {"perro", "gato"}

    def test_recorded_decisions_replay_without_asking(self, decisions) -> None:
        deck = [make_word("prenda", "nf", "piece of clothing", 3467)]
        added = [make_word("prenda", "nf", "garment", 9200)]
        decisions.remember(
            triage.Candidate(
                words=(*deck, *added),
                reason=triage.Reason.COMPATIBLE,
                comparison=_comparison(),
                disposition=triage.Disposition.MERGE,
                replacements=(
                    make_word("prenda", "nf", "piece of clothing, garment", 3467),
                ),
            )
        )

        result = resolve(
            inputs=added, existing=deck, decisions=decisions, interactive=False
        )

        assert not result.interrupted
        (merged,) = result.outcome.words
        assert merged.meaning == "piece of clothing, garment"

    def test_an_unanswered_question_interrupts(self, decisions) -> None:
        deck = [make_word("zanja", "nf", "trench", 1)]
        added = [make_word("zanja", "nf", "ditch", 2)]

        result = resolve(
            inputs=added, existing=deck, decisions=decisions, interactive=False
        )

        assert result.interrupted
        assert len(result.pending) == 1
        assert {w.meaning for w in result.pending[0].words} == {"trench", "ditch"}

    def test_whole_deck_mode_finds_deck_only_groups(self, decisions) -> None:
        """With no inputs the run is about everything, focus or none."""
        deck = [
            make_word("zanja", "nf", "trench", 1),
            make_word("zanja", "nf", "ditch", 2),
        ]

        result = resolve(
            inputs=(), existing=deck, decisions=decisions, interactive=False
        )

        assert result.interrupted
        assert len(result.pending) == 1

    def test_a_decided_merge_without_wording_is_asked_exactly_once(
        self, decisions
    ) -> None:
        """Half an answer gets half re-asked: the old flow recorded "merge"
        with no wording, and nothing invents the card's text mechanically —
        the wording question is asked, once, and the answer sticks."""
        deck = [make_word("colonia", "nf", "settlement", 4181)]
        added = [make_word("colonia", "nf", "cologne", 9500)]
        decisions.remember(
            triage.Candidate(
                words=(*deck, *added),
                reason=triage.Reason.COMPATIBLE,
                comparison=_comparison(),
                disposition=triage.Disposition.MERGE,
            )
        )

        first = resolve(
            inputs=added, existing=deck, decisions=decisions, interactive=False
        )
        assert first.interrupted, "the wording still needs an answer"
        (candidate,) = first.pending
        assert candidate.disposition is triage.Disposition.MERGE
        assert len(first.outcome.words) == 2, "nothing was worded mechanically"

        # Answering with wording — as the screen or the decision file would —
        # settles it for good.
        decisions.remember(
            attrs.evolve(
                candidate,
                replacements=(
                    make_word("colonia", "nf", "colonial settlement, cologne", 4181),
                ),
            )
        )
        second = resolve(
            inputs=added, existing=deck, decisions=decisions, interactive=False
        )
        assert not second.interrupted
        (merged,) = second.outcome.words
        assert merged.meaning == "colonial settlement, cologne"

    def test_legacy_wording_comes_from_the_merge_cache(
        self, decisions, tmp_path
    ) -> None:
        deck = [make_word("colonia", "nf", "settlement", 4181)]
        added = [make_word("colonia", "nf", "cologne", 9500)]
        group = (*deck, *added)
        decisions.remember(
            triage.Candidate(
                words=group,
                reason=triage.Reason.COMPATIBLE,
                comparison=_comparison(),
                disposition=triage.Disposition.MERGE,
            )
        )
        merge_cache = conflict_resolver.MergeCache(tmp_path / "merges.json")
        merge_cache.add_merge(
            conflict_resolver.MergeOperation(
                original_words=tuple(sorted(group)),
                result=make_word("colonia", "nf", "colonial settlement, cologne", 4181),
            )
        )

        result = resolve(
            inputs=added,
            existing=deck,
            decisions=decisions,
            merge_cache=merge_cache,
            interactive=False,
        )

        assert not result.interrupted
        (merged,) = result.outcome.words
        assert merged.meaning == "colonial settlement, cologne"


class TestAuto:
    def test_proposed_merges_are_accepted_mechanically(self, decisions) -> None:
        deck = [make_word("zanja", "nf", "trench, ditch", 1)]
        added = [make_word("zanja", "nf", "ditch, trench", 2)]

        result = resolve(
            inputs=added,
            existing=deck,
            decisions=decisions,
            interactive=False,
            auto=True,
        )

        assert not result.interrupted
        (merged,) = result.outcome.words
        assert merged.rank == 1
        # The mechanical acceptance was recorded, so a rerun will not re-ask.
        assert len(decisions) == 1

    def test_review_reasons_stay_pending_even_in_auto(self, decisions) -> None:
        """Gender pairs are never proposed, so auto cannot answer them."""
        deck = [make_word("cura", "nm", "priest", 1790)]
        added = [make_word("cura", "nf", "cure, remedy", 9000)]

        result = resolve(
            inputs=added,
            existing=deck,
            decisions=decisions,
            interactive=False,
            auto=True,
        )

        assert result.interrupted
        assert len(result.outcome.words) == 2


def _comparison(score: float = 0.9) -> "object":
    from anki_lang_cards import definition_similarity

    return definition_similarity.Comparison(score, frozenset())


def scripted_ui(answers: dict[str, triage.Disposition], monkeypatch) -> list[str]:
    """Stand in for the screen: answer by headword, report what was asked."""
    pytest.importorskip("textual", reason="the fake patches the real UI module")
    from anki_lang_cards import triage_ui

    asked: list[str] = []

    async def fake_resolve_candidates(
        candidates, driver_class=None, per_page=None, on_decision=None
    ):
        resolved = []
        for candidate in candidates:
            headword = candidate.words[0].word
            asked.append(headword)
            answer = answers.get(headword)
            if answer is None:
                resolved.append(candidate)
                continue
            replacements = candidate.replacements
            if answer is triage.Disposition.MERGE:
                replacements = (apply_merges.propose_merge(candidate.words),)
            answered = attrs.evolve(
                candidate, disposition=answer, replacements=replacements
            )
            if on_decision is not None:
                on_decision(answered)
            resolved.append(answered)
        return tuple(resolved)

    monkeypatch.setattr(triage_ui, "resolve_candidates", fake_resolve_candidates)
    return asked


class TestInteractive:
    def test_answers_are_recorded_and_applied(self, decisions, monkeypatch) -> None:
        asked = scripted_ui({"zanja": triage.Disposition.MERGE}, monkeypatch)
        deck = [make_word("zanja", "nf", "trench", 1)]
        added = [make_word("zanja", "nf", "ditch", 2)]

        result = resolve(inputs=added, existing=deck, decisions=decisions)

        assert asked == ["zanja"]
        assert not result.interrupted
        (merged,) = result.outcome.words
        assert merged.meaning == "trench, ditch"
        assert decisions.get(tuple(sorted((*deck, *added)))) is not None

    def test_an_edit_that_creates_a_collision_is_asked_in_the_same_run(
        self, decisions, monkeypatch
    ) -> None:
        """The fixpoint: answering can create the next question."""
        deck = [
            make_word("viejo", "nm", "old man", 50),
            make_word("antiguo", "adj", "former", 60),
        ]
        added = [make_word("antiguo", "adj", "old, ancient", 100)]

        edited = make_word("viejo", "adj", "old, ancient", 100)

        pytest.importorskip("textual")
        from anki_lang_cards import triage_ui

        asked: list[frozenset[str]] = []

        async def fake_resolve_candidates(
            candidates, driver_class=None, per_page=None, on_decision=None
        ):
            resolved = []
            for candidate in candidates:
                meanings = frozenset(w.meaning for w in candidate.words)
                asked.append(meanings)
                if meanings == {"former", "old, ancient"}:
                    # First question: tell the two apart by renaming ours.
                    ordered = sorted(candidate.words)
                    replacements = tuple(
                        edited if word.rank == 100 else word for word in ordered
                    )
                    answered = attrs.evolve(
                        candidate,
                        disposition=triage.Disposition.KEEP_SEPARATE,
                        replacements=replacements,
                    )
                elif meanings == {"old man", "old, ancient"}:
                    # Second question, which only exists because of the first.
                    answered = attrs.evolve(
                        candidate,
                        disposition=triage.Disposition.KEEP_SEPARATE,
                    )
                else:
                    resolved.append(candidate)
                    continue
                if on_decision is not None:
                    on_decision(answered)
                resolved.append(answered)
            return tuple(resolved)

        monkeypatch.setattr(triage_ui, "resolve_candidates", fake_resolve_candidates)

        result = resolve(inputs=added, existing=deck, decisions=decisions)

        assert result.passes == 2
        assert {"old man", "old, ancient"} in [set(m) for m in asked]
        assert not result.interrupted

    def test_quitting_leaves_the_rest_pending_but_keeps_answers(
        self, decisions, monkeypatch
    ) -> None:
        asked = scripted_ui({"zanja": triage.Disposition.MERGE}, monkeypatch)
        deck = [
            make_word("zanja", "nf", "trench", 1),
            make_word("cura", "nm", "priest", 3),
        ]
        added = [
            make_word("zanja", "nf", "ditch", 2),
            make_word("cura", "nf", "cure", 4),
        ]

        result = resolve(inputs=added, existing=deck, decisions=decisions)

        assert sorted(asked) == ["cura", "zanja"]
        assert result.interrupted
        assert [c.words[0].word for c in result.pending] == ["cura"]
        # The answered group merged even though the run was left unfinished.
        assert len(result.outcome.merges) == 1


class TestStaleGuard:
    def test_rank_only_matches_are_reported(self, decisions) -> None:
        old = (
            make_word("botarate", "nm", "fool", 7608),
            make_word("botarate", "nm", "fool, idiot", 9100),
        )
        decisions.remember(
            triage.Candidate(
                words=old,
                reason=triage.Reason.COMPATIBLE,
                comparison=_comparison(),
                disposition=triage.Disposition.MERGE,
                replacements=(make_word("botarate", "nm", "fool, idiot", 7608),),
            )
        )
        # The same ranks, but the notes have been rewritten since.
        deck = [make_word("botarate", "nm", "featherbrain", 7608)]
        added = [make_word("botarate", "nm", "fool, idiot (squanderer)", 9100)]

        result = resolve(
            inputs=added, existing=deck, decisions=decisions, interactive=False
        )

        assert len(result.rank_only_decisions) == 1

    def test_matching_decisions_are_not_reported(self, decisions) -> None:
        deck = [make_word("prenda", "nf", "piece of clothing", 3467)]
        added = [make_word("prenda", "nf", "garment", 9200)]
        decisions.remember(
            triage.Candidate(
                words=(*deck, *added),
                reason=triage.Reason.COMPATIBLE,
                comparison=_comparison(),
                disposition=triage.Disposition.MERGE,
                replacements=(make_word("prenda", "nf", "clothing, garment", 3467),),
            )
        )

        result = resolve(
            inputs=added, existing=deck, decisions=decisions, interactive=False
        )

        assert result.rank_only_decisions == ()


class TestDecisionFile:
    def pending_candidate(self) -> triage.Candidate:
        return triage.Candidate(
            words=(
                make_word("zanja", "nf", "trench", 1),
                make_word("zanja", "nf", "ditch", 2),
            ),
            reason=triage.Reason.COMPATIBLE,
            comparison=_comparison(),
            disposition=triage.Disposition.REVIEW,
        )

    def test_the_file_carries_the_proposal_and_an_index(self, tmp_path) -> None:
        path = tmp_path / "pending.json"
        dedup_flow.write_pending_file(path, [self.pending_candidate()])

        (entry,) = json.loads(path.read_text())
        assert entry["index"] == 0
        (proposed,) = triage.words_from_csv(entry["proposed"])
        assert proposed.meaning == "trench, ditch"

    def test_a_merge_answer_defaults_to_the_proposal(self, tmp_path) -> None:
        path = tmp_path / "pending.json"
        dedup_flow.write_pending_file(path, [self.pending_candidate()])
        data = json.loads(path.read_text())
        data[0]["disposition"] = "merge"
        path.write_text(json.dumps(data))

        (candidate,) = dedup_flow.read_decision_file(path)

        assert candidate.disposition is triage.Disposition.MERGE
        (merged,) = candidate.replacements
        assert merged.meaning == "trench, ditch"
        assert merged.rank == 1

    def test_a_split_answer_can_allocate_sentences(self, tmp_path) -> None:
        first = make_word(
            "llave",
            "nf",
            "key, faucet",
            810,
            [
                types.Sentence(
                    spanish="La llave del grifo.", english="The tap key.", source=None
                )
            ],
        )
        second = make_word(
            "llave",
            "nf",
            "wrestling hold",
            9241,
            [
                types.Sentence(
                    spanish="Una llave de judo.", english="A judo hold.", source=None
                )
            ],
        )
        candidate = attrs.evolve(self.pending_candidate(), words=(first, second))
        path = tmp_path / "pending.json"
        dedup_flow.write_pending_file(path, [candidate])
        data = json.loads(path.read_text())
        data[0]["disposition"] = "split"
        data[0]["replacements"] = triage.words_to_csv(
            [
                make_word("llave", "nf", "key, faucet", 810),
                make_word("llave (lucha)", "nf", "wrestling hold", 9241),
            ]
        )
        data[0]["sentence_allocation"] = ["1", "2"]
        path.write_text(json.dumps(data))

        (candidate,) = dedup_flow.read_decision_file(path)

        first_card, second_card = candidate.replacements
        assert [s.spanish for s in first_card.sentences] == ["La llave del grifo."]
        assert [s.spanish for s in second_card.sentences] == ["Una llave de judo."]

    def test_problems_name_the_candidate_and_field(self, tmp_path) -> None:
        path = tmp_path / "pending.json"
        dedup_flow.write_pending_file(
            path, [self.pending_candidate(), self.pending_candidate()]
        )
        data = json.loads(path.read_text())
        data[0]["disposition"] = "split"  # no replacements: invalid
        data[1]["disposition"] = "definitely-not-a-disposition"
        path.write_text(json.dumps(data))

        with pytest.raises(dedup_flow.DecisionFileError) as excinfo:
            dedup_flow.read_decision_file(path)

        message = str(excinfo.value)
        assert "candidate 0" in message
        assert "at least two replacement rows" in message
        assert "candidate 1" in message

    def test_a_bad_allocation_names_the_sentence(self, tmp_path) -> None:
        first = make_word(
            "llave",
            "nf",
            "key",
            810,
            [types.Sentence(spanish="Frase.", english="Sentence.", source=None)],
        )
        candidate = attrs.evolve(
            self.pending_candidate(),
            words=(first, make_word("llave", "nf", "hold", 9241)),
        )
        path = tmp_path / "pending.json"
        dedup_flow.write_pending_file(path, [candidate])
        data = json.loads(path.read_text())
        data[0]["disposition"] = "split"
        data[0]["replacements"] = triage.words_to_csv(
            [
                make_word("llave", "nf", "key", 810),
                make_word("llave (lucha)", "nf", "hold", 9241),
            ]
        )
        data[0]["sentence_allocation"] = ["7"]
        path.write_text(json.dumps(data))

        with pytest.raises(dedup_flow.DecisionFileError) as excinfo:
            dedup_flow.read_decision_file(path)

        assert "sentence_allocation[0]" in str(excinfo.value)

    @pytest.mark.parametrize(
        ("contents", "expected"),
        [
            pytest.param(None, "No such file or directory", id="missing"),
            pytest.param("[1, 2", "is not JSON", id="not-json"),
            pytest.param('{"a": 1}', "list of candidates, not dict", id="object"),
            pytest.param('["x"]', "candidate 0 is not an object", id="entry"),
        ],
    )
    def test_an_unusable_file_names_the_problem(
        self, tmp_path: pathlib.Path, contents: str | None, expected: str
    ) -> None:
        """A file that cannot be read at all is a DecisionFileError too, so
        every caller reports it the same way as a bad answer."""
        path = tmp_path / "pending.json"
        if contents is not None:
            path.write_text(contents)

        with pytest.raises(dedup_flow.DecisionFileError) as excinfo:
            dedup_flow.read_decision_file(path)

        assert expected in str(excinfo.value)

    def test_load_decisions_folds_the_file_in(
        self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "pending.json"
        dedup_flow.write_pending_file(path, [self.pending_candidate()])
        data = json.loads(path.read_text())
        data[0]["disposition"] = "merge"
        path.write_text(json.dumps(data))
        cache_path = tmp_path / "decisions.json"

        decisions = dedup_flow.load_decisions(cache_path, path)

        assert len(decisions) == 1
        assert len(triage.DecisionCache(cache_path)) == 1
        out = capsys.readouterr().out
        assert "Read 0 decisions" in out
        assert "WARNING: no recorded decisions" in out
        assert "Folded in 1 new answers" in out

    def test_load_decisions_exits_2_on_a_bad_file(
        self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A bad file stops the run with its problems, and records nothing."""
        cache_path = tmp_path / "decisions.json"

        with pytest.raises(SystemExit) as excinfo:
            dedup_flow.load_decisions(cache_path, tmp_path / "missing.json")

        assert excinfo.value.code == 2
        assert "missing.json" in capsys.readouterr().err
        assert not cache_path.exists()

    def test_a_file_round_trip_matches_a_ui_session(self, decisions, tmp_path) -> None:
        """Answering by file and answering on screen record the same thing."""
        deck = [make_word("zanja", "nf", "trench", 1)]
        added = [make_word("zanja", "nf", "ditch", 2)]

        first = resolve(
            inputs=added, existing=deck, decisions=decisions, interactive=False
        )
        assert first.interrupted

        path = tmp_path / "pending.json"
        dedup_flow.write_pending_file(path, first.pending)
        data = json.loads(path.read_text())
        data[0]["disposition"] = "merge"
        path.write_text(json.dumps(data))
        assert dedup_flow.record_decision_file(path, decisions) == 1

        second = resolve(
            inputs=added, existing=deck, decisions=decisions, interactive=False
        )

        assert not second.interrupted
        (merged,) = second.outcome.words
        assert merged.meaning == "trench, ditch"
        assert merged.rank == 1


class TestQuitDoesNotReopen:
    def test_walking_away_ends_the_session_for_the_run(
        self, decisions, monkeypatch
    ) -> None:
        """Quitting with questions left must not reopen the screen with the
        next batch — one q means out, and the rest goes to pending."""
        deck = [
            make_word("viejo", "nm", "old man", 50),
            make_word("antiguo", "adj", "former", 60),
            make_word("cura", "nm", "priest", 70),
        ]
        added = [
            make_word("antiguo", "adj", "old, ancient", 100),
            make_word("cura", "nf", "cure, remedy", 110),
        ]
        edited = make_word("viejo", "adj", "old, ancient", 100)

        pytest.importorskip("textual")
        from anki_lang_cards import triage_ui

        sessions: list[int] = []

        async def fake_resolve_candidates(
            candidates, driver_class=None, per_page=None, on_decision=None
        ):
            sessions.append(len(candidates))
            resolved = []
            for candidate in candidates:
                meanings = frozenset(w.meaning for w in candidate.words)
                if meanings == {"former", "old, ancient"}:
                    # Answer one question in a way that creates a new
                    # collision, then "quit" with the cura pair unanswered.
                    ordered = sorted(candidate.words)
                    answered = attrs.evolve(
                        candidate,
                        disposition=triage.Disposition.KEEP_SEPARATE,
                        replacements=tuple(
                            edited if word.rank == 100 else word for word in ordered
                        ),
                    )
                    if on_decision is not None:
                        on_decision(answered)
                    resolved.append(answered)
                else:
                    resolved.append(candidate)
            return tuple(resolved)

        monkeypatch.setattr(triage_ui, "resolve_candidates", fake_resolve_candidates)

        result = resolve(inputs=added, existing=deck, decisions=decisions)

        assert len(sessions) == 1, "the screen must not reopen after a quit"
        assert result.passes == 1
        assert result.interrupted
        pending_words = {
            frozenset(w.meaning for w in candidate.words)
            for candidate in result.pending
        }
        assert {"priest", "cure, remedy"} in pending_words
        assert {
            "old man",
            "old, ancient",
        } in pending_words, "the collision the answer created is pending, not re-asked"


class TestCachedWordingMigration:
    def test_a_cached_wording_is_promoted_and_not_asked(
        self, decisions, tmp_path
    ) -> None:
        """The main pipeline's old answers live only in the merge-wording
        cache; a group answered there must not be asked again."""
        deck = [make_word("colonia", "nf", "settlement", 4181)]
        added = [make_word("colonia", "nf", "cologne", 9500)]
        merge_cache = conflict_resolver.MergeCache(tmp_path / "merged.json")
        merge_cache.add_merge(
            conflict_resolver.MergeOperation(
                original_words=tuple(sorted((*deck, *added))),
                result=make_word("colonia", "nf", "colonial settlement, cologne", 4181),
            )
        )

        result = resolve(
            inputs=added,
            existing=deck,
            decisions=decisions,
            merge_cache=merge_cache,
            interactive=False,
        )

        assert not result.interrupted
        (merged,) = result.outcome.words
        assert merged.meaning == "colonial settlement, cologne"
        # Promoted permanently: a rerun without the merge cache still knows.
        second = resolve(
            inputs=added, existing=deck, decisions=decisions, interactive=False
        )
        assert not second.interrupted
