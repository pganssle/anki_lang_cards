"""Tests for the annotations_to_csv command run in-process.

The model is replaced by a backend scripted from the prompt text, so these
cover what the command does with the runner's outcomes without the fake
``llm`` plugin the end-to-end tests need.
"""

import asyncio
import csv
import json
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self

import pytest

from anki_lang_cards import annotations_to_csv, caches, model_selection, runner
from anki_lang_cards.backends import base

FAILING_WORD = "fallador"
"""A highlight whose prompt the backend rejects."""


class ScriptedBackend:
    """Answers each prompt with one entry per highlight, or rejects it."""

    service: ClassVar[str] = "scripted"
    features: ClassVar[frozenset[base.Feature]] = frozenset()
    default_words_per_prompt: ClassVar[int] = 1

    def __init__(self) -> None:
        self.model_name = "scripted"
        self.max_concurrency: int | None = None
        self.queried: list[str] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def query_params(self) -> Mapping[str, object]:
        return {}

    async def query(self, request: base.Request) -> caches.ResponseData:
        self.queried.append(request.prompt)
        if FAILING_WORD in request.prompt:
            raise base.RequestError("HTTP 500: overloaded", status=500)
        words = [
            {
                "highlight": highlight,
                "headword": highlight,
                "part_of_speech": "nm",
                "meaning": f"meaning of {highlight}",
                "article": "el",
                "examples": [
                    {
                        "spanish": f"Una frase con {highlight}.",
                        "english": f"A sentence with {highlight}.",
                        "source_id": entry["source_id"],
                    }
                ],
            }
            for entry in _prompt_entries(request.prompt)
            for highlight in entry["highlights"]
        ]
        text = json.dumps({"words": words}, ensure_ascii=False)
        return {"text": text, "usage": {"input": 1, "output": 1, "details": None}}


def _prompt_entries(prompt: str) -> Sequence[Mapping[str, Any]]:
    start = prompt.index("INPUT:")
    start = prompt.index("[", start)
    end = prompt.index("\n\nOUTPUT:", start)
    return json.loads(prompt[start:end])


def _annotations(highlights: Sequence[str]) -> Sequence[Mapping[str, Any]]:
    """One book with one annotation per highlight, each with a context long
    enough that no two fit in one prompt."""
    return [
        {
            "book": {
                "title": "Libro",
                "author": "Autora",
                "path": "Autora/Libro (1)",
                "language": "spa",
            },
            "annotations": [
                {
                    "highlights": [highlight],
                    "context": f"Aquí aparece {highlight}. " + "Relleno. " * 1000,
                    "chapters": ["Capítulo 1"],
                    "page_numbers": [page],
                }
                for page, highlight in enumerate(highlights, start=1)
            ],
        }
    ]


@pytest.fixture
def workdir(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "config.toml").write_text(
        '[models]\ndefault_model = "scripted"\n', encoding="utf-8"
    )
    (tmp_path / "es.aff").write_text("SET UTF-8\n", encoding="utf-8")
    (tmp_path / "es.dic").write_text("1\npalabra\n", encoding="utf-8")
    return tmp_path


def _run_main(
    workdir: pathlib.Path,
    highlights: Sequence[str],
    monkeypatch: pytest.MonkeyPatch,
    *extra_args: str,
) -> tuple[ScriptedBackend, int | None]:
    (workdir / "combined.json").write_text(
        json.dumps(_annotations(highlights), ensure_ascii=False), encoding="utf-8"
    )
    backend = ScriptedBackend()

    def make_runner(*args: object, **kwargs: object):
        cacher = caches.QueryCacher(
            cache_type=caches.CacheTypes.ANNOTATIONS,
            model_name="scripted",
            model_service="scripted",
            cache_loc=workdir / "cache",
        )
        return (
            model_selection.ResolvedModel("scripted", "scripted", "scripted"),
            runner.QueryRunner(backend, cacher),
        )

    monkeypatch.setattr(annotations_to_csv.model_selection, "make_runner", make_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "annotations_to_csv",
            "--input",
            str(workdir / "combined.json"),
            "--output",
            str(workdir / "words.csv"),
            "--config",
            str(workdir / "config.toml"),
            "--hunspell-dict-path",
            str(workdir / "es.dic"),
            "--skip-list",
            str(workdir / "no-skip-list.json"),
            "--processed-pages-output",
            str(workdir / "processed_pages.json"),
            "--max-retries",
            "1",
            *extra_args,
        ],
    )
    try:
        asyncio.run(annotations_to_csv.main())
    except SystemExit as e:
        code = e.code
    else:
        code = None
    assert code is None or isinstance(code, int)
    return backend, code


def _headwords(csv_path: pathlib.Path) -> frozenset[str]:
    with csv_path.open(encoding="utf-8", newline="") as f:
        return frozenset(row[1] for row in csv.reader(f))


def test_complete_run(workdir: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend, code = _run_main(workdir, ["gato", "perro"], monkeypatch)

    assert code is None
    assert len(backend.queried) == 2
    assert _headwords(workdir / "words.csv") == {"gato", "perro"}
    assert (workdir / "processed_pages.json").exists()


def test_dropped_prompt(
    workdir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A prompt that fails every retry is reported, its pages are not marked
    processed, and the exit status says so; the rest is written as usual."""
    backend, code = _run_main(workdir, ["gato", FAILING_WORD, "perro"], monkeypatch)

    assert code == runner.EXIT_REQUESTS_DROPPED == 4
    assert sum(FAILING_WORD in prompt for prompt in backend.queried) == 2
    assert _headwords(workdir / "words.csv") == {"gato", "perro"}
    assert not (workdir / "processed_pages.json").exists()
    out = capsys.readouterr().out
    assert (
        "WARNING: 1 of 3 prompt(s) failed after every retry, so 1 annotation "
        "entries with 1 highlight(s) produced no cards:"
    ) in out
    assert "HTTP 500: overloaded" in out
    assert "Not writing processed pages: some prompts produced no cards." in out


def test_every_prompt_failing_aborts(
    workdir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run nothing gets through is stopped before any retries, and
    nothing is written."""
    with pytest.raises(runner.AllRequestsFailedError, match="HTTP 500"):
        _run_main(workdir, [FAILING_WORD, f"{FAILING_WORD}a"], monkeypatch)

    assert not (workdir / "words.csv").exists()
    assert not (workdir / "processed_pages.json").exists()


def test_limit_and_dropped_prompt_both_withhold_processed_pages(
    workdir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, code = _run_main(
        workdir, [FAILING_WORD, "gato", "perro"], monkeypatch, "--limit", "2"
    )

    assert code == runner.EXIT_REQUESTS_DROPPED
    assert _headwords(workdir / "words.csv") == {"gato"}
    assert not (workdir / "processed_pages.json").exists()


def test_pack_prompt_inputs_matches_generate_prompts() -> None:
    books = [
        annotations_to_csv.CombinedBookAnnotations.from_dict(book)
        for book in _annotations(["uno", "dos", "tres"])
    ]
    inputs = list(annotations_to_csv.prepare_prompt_inputs(books, {}))

    groups = list(annotations_to_csv.pack_prompt_inputs(inputs))
    prompts = list(annotations_to_csv.generate_prompts(inputs))

    assert [list(group) for group in groups] == [[entry] for entry in inputs]
    assert len(prompts) == len(groups)
    for prompt, (entry,) in zip(prompts, groups, strict=True):
        assert entry.annotation_entry["highlights"][0] in prompt.prompt
