"""End-to-end tests for the card-generation pipeline with a fake LLM.

These run the real CLIs (``combine_annotations``, ``annotations_to_csv`` and
``anki_lang_cards.__main__``) in subprocesses, with the LLM replaced by the
``fake-cards`` model from tests/fakes/llm-fake-cards. The fake fabricates
deterministic responses whose sentences deliberately contain the characters
that have historically been mangled somewhere between the LLM and Anki
(straight and Spanish quotation marks, escaped quotes, ampersands, inverted
punctuation), and marker words in the fixtures trigger the misbehaviors the
pipeline has to survive: fenced ```json responses, invalid JSON, unknown
citation ids, and terms the model fails to echo back.
"""

import csv
import json
import pathlib
import struct
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Final, NamedTuple

import attrs
import pytest

from anki_lang_cards import _types as types
from anki_lang_cards import anki_import, definition_similarity

try:
    import llm_fake_cards  # noqa: F401
except ImportError:
    _FAKE_MODEL_AVAILABLE = False
else:
    _FAKE_MODEL_AVAILABLE = True

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not _FAKE_MODEL_AVAILABLE,
        reason="The llm-fake-cards plugin is not installed",
    ),
]

DATA_DIR: Final = pathlib.Path(__file__).parent / "data"
CARDS_DIR: Final = DATA_DIR / "cards"
ANNOTATIONS_JSON: Final = DATA_DIR / "annotations.json"
SCRIPTS_DIR: Final = pathlib.Path(__file__).parent.parent / "scripts"

ANNOTATION_HEADWORDS: Final = frozenset(
    {
        "gatos",
        "fiambres (comida)",
        "fencear",
        "badjsonero",
        "badsourcero",
        "comillas",
    }
)


def _write_stardict(prefix: pathlib.Path, entries: Mapping[str, str]) -> None:
    """Write a minimal StarDict dictionary (a .ifo/.idx/.dict triple)."""
    idx_parts: list[bytes] = []
    dict_parts: list[bytes] = []
    offset = 0
    for word in sorted(entries):
        data = entries[word].encode("utf-8")
        idx_parts.append(
            word.encode("utf-8") + b"\0" + struct.pack(">II", offset, len(data))
        )
        dict_parts.append(data)
        offset += len(data)
    idx = b"".join(idx_parts)
    prefix.with_suffix(".idx").write_bytes(idx)
    prefix.with_suffix(".dict").write_bytes(b"".join(dict_parts))
    prefix.with_suffix(".ifo").write_text(
        "StarDict's dict ifo file\n"
        "version=2.4.2\n"
        f"wordcount={len(entries)}\n"
        f"idxfilesize={len(idx)}\n"
        "bookname=Fixture Dictionary\n"
        "sametypesequence=m\n",
        encoding="utf-8",
    )


class DictionaryPaths(NamedTuple):
    stardict_prefix: pathlib.Path
    hunspell_dic: pathlib.Path


@pytest.fixture(scope="session")
def dictionary_paths(tmp_path_factory: pytest.TempPathFactory) -> DictionaryPaths:
    directory = tmp_path_factory.mktemp("dictionaries")
    prefix = directory / "fixture-dict"
    _write_stardict(
        prefix,
        {
            "gato": "<k>gato</k>\ncat, male cat",
            "diamante": "<k>diamante</k><i>n</i><ol><li>diamond</li></ol>",
            "espada": "<k>espada</k>\nsword",
            "agua": "<k>agua</k>\nwater",
        },
    )
    # A hunspell dictionary with a single plural rule, so that e.g. "gatos"
    # stems to the dictionary headword "gato".
    (directory / "es_fixture.aff").write_text(
        "SET UTF-8\nSFX S Y 1\nSFX S 0 s .\n", encoding="utf-8"
    )
    (directory / "es_fixture.dic").write_text(
        "4\ngato/S\ndiamante/S\nespada/S\nagua/S\n", encoding="utf-8"
    )
    return DictionaryPaths(
        stardict_prefix=prefix, hunspell_dic=directory / "es_fixture.dic"
    )


class CliRun(NamedTuple):
    workdir: pathlib.Path
    output_csv: pathlib.Path
    stdout: str


def _write_config(
    workdir: pathlib.Path,
    dictionaries: DictionaryPaths,
    dialect: str | None = None,
) -> pathlib.Path:
    config_path = workdir / "config.toml"
    config_text = (
        "[models]\n"
        'default_model="fake-cards"\n'
        "[annotations]\n"
        f'dictionaries = ["{dictionaries.stardict_prefix.as_posix()}"]\n'
    )
    if dialect is not None:
        config_text += f'[sentences]\ndialect = "{dialect}"\n'
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def _run_annotations_to_csv_raw(
    workdir: pathlib.Path,
    combined_annotations: pathlib.Path,
    dictionaries: DictionaryPaths,
    output_csv: pathlib.Path,
    extra_args: Sequence[str] = (),
    dialect: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Like _run_annotations_to_csv, but tolerant of failure."""
    config_path = _write_config(workdir, dictionaries, dialect=dialect)
    return subprocess.run(
        (
            sys.executable,
            "-m",
            "anki_lang_cards.annotations_to_csv",
            "--input",
            str(combined_annotations),
            "--output",
            str(output_csv),
            "--model-name",
            "fake-cards",
            "--hunspell-dict-path",
            str(dictionaries.hunspell_dic),
            "--config",
            str(config_path),
            "--processed-pages-output",
            str(workdir / "processed_pages.json"),
            "--max-simultaneous-queries",
            "1",
            *extra_args,
        ),
        capture_output=True,
        text=True,
        cwd=workdir,
    )


def _run_annotations_to_csv(
    workdir: pathlib.Path,
    combined_annotations: pathlib.Path,
    dictionaries: DictionaryPaths,
    output_name: str = "words_from_annotations.csv",
    extra_args: Sequence[str] = (),
    dialect: str | None = None,
) -> CliRun:
    output_csv = workdir / output_name
    result = _run_annotations_to_csv_raw(
        workdir,
        combined_annotations,
        dictionaries,
        output_csv,
        extra_args=extra_args,
        dialect=dialect,
    )
    result.check_returncode()
    return CliRun(workdir=workdir, output_csv=output_csv, stdout=result.stdout)


@pytest.fixture(scope="session")
def annotations_run(
    dictionary_paths: DictionaryPaths, tmp_path_factory: pytest.TempPathFactory
) -> CliRun:
    """Run annotations_to_csv over the synthetic marker-bearing fixture."""
    workdir = tmp_path_factory.mktemp("annotations-run")
    return _run_annotations_to_csv(
        workdir, CARDS_DIR / "combined_annotations.json", dictionary_paths
    )


def _raw_rows(csv_path: pathlib.Path) -> Sequence[Mapping[str, str]]:
    """Read a headerless pipeline CSV with the canonical column names."""
    with csv_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, fieldnames=list(types.Word.make_csv_header())))


def _words_by_headword(csv_path: pathlib.Path) -> Mapping[str, types.Word]:
    words = types.Word.words_from_csv(csv_path)
    by_headword = {word.word: word for word in words}
    assert len(by_headword) == len(words), "duplicate headwords in output"
    return by_headword


def test_annotations_to_csv_words(annotations_run: CliRun) -> None:
    words = _words_by_headword(annotations_run.output_csv)
    assert set(words) == set(ANNOTATION_HEADWORDS)
    for word in words.values():
        assert len(word.sentences) == 3
        highlight = word.word.split(" (")[0]
        assert word.meaning == f'a fake "meaning" of {highlight}'


def test_note_becomes_headword_qualifier(annotations_run: CliRun) -> None:
    """A `K:`-style note rides along as a parenthetical on the headword."""
    words = _words_by_headword(annotations_run.output_csv)
    assert "fiambres (comida)" in words


def test_sentence_source_roundtrip(annotations_run: CliRun) -> None:
    """Citations survive the CSV as JSON usable by scripts/find_book_ranges."""
    words = _words_by_headword(annotations_run.output_csv)
    source = words["gatos"].sentence_source
    assert isinstance(source, types.BookCitation)
    assert source.title == "Libro de Pruebas"
    assert tuple(source.pages) == (12, 13)

    rows = {row["word"]: row for row in _raw_rows(annotations_run.output_csv)}
    raw_source = json.loads(rows["gatos"]["sentence_source"])
    assert raw_source["title"] == "Libro de Pruebas"
    assert raw_source["pages"] == [12, 13]


def test_example_sentence_sources_are_objects(annotations_run: CliRun) -> None:
    """Sources inside example_sentences must be JSON objects, not
    double-encoded JSON strings (the damage scripts/fix_example_sentences.py
    had to repair retrospectively)."""
    for row in _raw_rows(annotations_run.output_csv):
        if not row["example_sentences"]:
            continue
        for sentence in json.loads(row["example_sentences"]):
            if "source" in sentence:
                assert isinstance(sentence["source"], dict)


def test_unknown_source_id_dropped(annotations_run: CliRun) -> None:
    words = _words_by_headword(annotations_run.output_csv)
    assert all(sentence.source is None for sentence in words["badsourcero"].sentences)
    assert "Unknown citation id" in annotations_run.stdout


def test_invalid_json_response_retried(annotations_run: CliRun) -> None:
    """The first response for the badjsonero prompt is not JSON; the retry
    invalidates the response cache and succeeds."""
    assert "Attempt 1 failed" in annotations_run.stdout
    words = _words_by_headword(annotations_run.output_csv)
    assert "badjsonero" in words


def test_fenced_response_unwrapped(annotations_run: CliRun) -> None:
    assert "Removing ```json wrapper" in annotations_run.stdout
    words = _words_by_headword(annotations_run.output_csv)
    assert "fencear" in words


def test_quotes_survive_csv_roundtrip(annotations_run: CliRun) -> None:
    """Straight quotes, guillemets and literal backslash-quote sequences in
    the generated sentences survive the JSON-in-CSV round trip."""
    words = _words_by_headword(annotations_run.output_csv)
    spanish = [sentence.spanish for sentence in words["comillas"].sentences]
    english = [sentence.english for sentence in words["comillas"].sentences]
    assert any('"Las comillas dobles son peligrosas"' in s for s in spanish)
    assert any("«más comillas»" in s for s in spanish)
    assert any("«comillas»" in s for s in spanish)
    # The fake model escapes quotes with literal backslashes in English.
    assert any('\\"comillas\\"' in s for s in english)
    assert any("&" in sentence.spanish for sentence in words["gatos"].sentences)


def test_rag_entries_reach_the_prompt(annotations_run: CliRun) -> None:
    """The stemmed dictionary entry for "gatos" is offered to the model."""
    cache_dir = annotations_run.workdir / ".cache" / "annotations"
    cached_prompts = [
        json.loads(path.read_text(encoding="utf-8"))["input"]["prompt"]
        for path in cache_dir.rglob("*.json")
    ]
    assert any("gato" in prompt and "0. cat" in prompt for prompt in cached_prompts)


def test_processed_pages_output(annotations_run: CliRun) -> None:
    processed = json.loads(
        (annotations_run.workdir / "processed_pages.json").read_text(encoding="utf-8")
    )
    assert processed == [
        {
            "title": "Libro de Pruebas",
            "author": "Autora de Ejemplo",
            "page_numbers": [{"range": [12, 40]}],
        }
    ]


def test_second_run_is_cached_and_identical(
    annotations_run: CliRun, dictionary_paths: DictionaryPaths
) -> None:
    cache_files = sorted((annotations_run.workdir / ".cache").rglob("*.json"))
    mtimes = [path.stat().st_mtime_ns for path in cache_files]

    rerun = _run_annotations_to_csv(
        annotations_run.workdir,
        CARDS_DIR / "combined_annotations.json",
        dictionary_paths,
        output_name="second_run.csv",
    )

    assert sorted((annotations_run.workdir / ".cache").rglob("*.json")) == cache_files
    assert [path.stat().st_mtime_ns for path in cache_files] == mtimes

    def rows(path: pathlib.Path) -> Sequence[Sequence[str]]:
        with path.open(encoding="utf-8", newline="") as f:
            return sorted(map(tuple, csv.reader(f)))

    assert rows(rerun.output_csv) == rows(annotations_run.output_csv)


@pytest.mark.skipif(
    not (SCRIPTS_DIR / "find_book_ranges.py").exists(),
    reason="scripts/ is not present (e.g. inside the Docker test image)",
)
def test_find_book_ranges_script(
    dictionary_paths: DictionaryPaths, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The script reads the CSV with DictReader, so it needs --csv-header."""
    run = _run_annotations_to_csv(
        tmp_path_factory.mktemp("book-ranges-run"),
        CARDS_DIR / "combined_annotations.json",
        dictionary_paths,
        extra_args=("--csv-header",),
    )
    result = subprocess.run(
        (
            sys.executable,
            str(SCRIPTS_DIR / "find_book_ranges.py"),
            str(run.output_csv),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Libro de Pruebas" in result.stdout
    assert "12 - 40" in result.stdout


#####
# Full-chain test: extractor fixture -> combine_annotations -> annotations_to_csv
@pytest.fixture(scope="session")
def combined_from_extractor(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    workdir = tmp_path_factory.mktemp("combine-run")
    combined = workdir / "combined_annotations.json"
    subprocess.run(
        (
            sys.executable,
            "-m",
            "anki_lang_cards.combine_annotations",
            "--input",
            str(ANNOTATIONS_JSON),
            "--output",
            str(combined),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return combined


def test_combine_groups_overlapping_highlights(
    combined_from_extractor: pathlib.Path,
) -> None:
    combined = types.CombinedBookAnnotations.sequence_from_json(combined_from_extractor)
    (book,) = combined
    groups = {
        tuple(annotation.highlights): annotation for annotation in book.annotations
    }

    highlight_groups = [set(group) for group in groups]
    # Highlights sharing one sentence are combined into a single group.
    assert {"nombre", "acordarme", "astillero"} in highlight_groups
    assert {
        "Fuego soy apartado",
        "apartado",
        "espada",
        "espada puesta lejos",
    } in highlight_groups
    # The annotation with a non-vocabulary note ("Skip me") is dropped.
    all_highlights = {hl for group in highlight_groups for hl in group}
    assert "ingenioso hidalgo don Quijote de la Mancha" not in all_highlights
    assert "missing position" not in all_highlights


def test_chain_extractor_to_words(
    combined_from_extractor: pathlib.Path,
    dictionary_paths: DictionaryPaths,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The full chain from real extracted annotations to a words CSV."""
    workdir = tmp_path_factory.mktemp("chain-run")
    run = _run_annotations_to_csv(workdir, combined_from_extractor, dictionary_paths)

    words = _words_by_headword(run.output_csv)
    assert "diamante" in words
    snippet = words["diamante"].sentences[0].spanish
    assert "diamante" in snippet

    # "vuestra merced" was highlighted twice in different chapters: the two
    # results must merge into one word with sentences from both contexts.
    merced = words["vuestra merced"]
    spanish = {sentence.spanish for sentence in merced.sentences}
    assert any("disponerme a dar gusto" in s for s in spanish)
    assert any("déjeme dar otros mil azotes" in s for s in spanish)

    for word in words.values():
        source = word.sentence_source
        if source is not None:
            assert source.title == "Don Quijote (Abridged)"


#####
# The sentence-generation pipeline (anki_lang_cards.__main__)
def _run_main(
    workdir: pathlib.Path,
    inputs: pathlib.Path,
    dictionaries: DictionaryPaths,
    extra_args: Sequence[str] = (),
) -> CliRun:
    config_path = _write_config(workdir, dictionaries)
    output_dir = workdir / "output"
    output_dir.mkdir()
    output_csv = output_dir / "output.csv"
    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "anki_lang_cards",
            "--inputs",
            str(inputs),
            "--output",
            str(output_csv),
            "--model-name",
            "fake-cards",
            "--config",
            str(config_path),
            "--exported-deck",
            str(CARDS_DIR / "exported_deck.txt"),
            "--diff-output",
            str(output_dir / "diff.csv"),
            *extra_args,
        ),
        check=True,
        capture_output=True,
        text=True,
        cwd=workdir,
    )
    return CliRun(workdir=workdir, output_csv=output_csv, stdout=result.stdout)


@pytest.fixture(scope="session")
def main_run(
    dictionary_paths: DictionaryPaths, tmp_path_factory: pytest.TempPathFactory
) -> CliRun:
    return _run_main(
        tmp_path_factory.mktemp("main-run"), CARDS_DIR / "words.csv", dictionary_paths
    )


def test_main_generates_sentences(main_run: CliRun) -> None:
    words = _words_by_headword(main_run.output_csv)
    gato = words["gato"]
    assert gato.rank == 1401  # One above the highest rank in the deck export
    assert gato.model_used == "fake-cards"
    spanish = {sentence.spanish for sentence in gato.sentences}
    assert spanish == {
        'La palabra "gato" apareció en el libro.',
        "Los expertos dicen: «gato» es fundamental.",
        "¡gato! ¿Quién lo diría?",
        "Ayer aprendí gato — una palabra muy útil.",
    }
    for sentence in gato.sentences:
        assert isinstance(sentence.source, types.ModelCitation)
        assert sentence.source.model == "fake-cards"


def test_main_merges_existing_deck_word(main_run: CliRun) -> None:
    """A deck word sent to the LLM for more sentences keeps its rank and
    its existing sentences."""
    words = _words_by_headword(main_run.output_csv)
    existente = words["existente"]
    assert existente.rank == 1400  # The deck's rank, not a fresh one
    spanish = {sentence.spanish for sentence in existente.sentences}
    assert "La palabra existente ya estaba en el mazo." in spanish
    assert "Los expertos dicen: «existente» es fundamental." in spanish


def test_main_skips_mismatched_terms(main_run: CliRun) -> None:
    """Terms the model renames or reports under a "word" key are skipped."""
    assert "Term mismatch" in main_run.stdout
    words = _words_by_headword(main_run.output_csv)
    assert "renamedo/a" not in words
    assert "renamedo" not in words
    assert "wordkeyado" not in words


def test_main_output_roundtrips(main_run: CliRun) -> None:
    words = types.Word.words_from_csv(main_run.output_csv)
    for word in words:
        line = word.to_csv_line(require_rank=True)
        assert line[1] == word.word


def test_main_invalid_json_response_retried(
    dictionary_paths: DictionaryPaths, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A response that fails to parse is retried against the model (not the
    response cache), so the second, valid response is used."""
    run = _run_main(
        tmp_path_factory.mktemp("badjson-run"),
        CARDS_DIR / "words_badjson.csv",
        dictionary_paths,
    )
    assert "Retrying" in run.stdout
    words = _words_by_headword(run.output_csv)
    assert "badjsonito" in words
    assert "inocente" in words


def test_load_anki_export_repairs_mangled_json() -> None:
    """load_anki_words repairs the &quot; damage in real deck exports."""
    words = {
        word.word: word
        for word in anki_import.load_anki_words(CARDS_DIR / "exported_deck.txt")
    }
    existente = words["existente"]
    extra = existente.sentences[1]
    assert extra.spanish == 'Una frase "citada" de ejemplo.'
    assert extra.english == 'An "example" quoted sentence.'
    assert isinstance(extra.source, types.ModelCitation)
    assert extra.source.model == "old-model"


def test_load_anki_export_repairs_ampersands() -> None:
    """Escaped ampersands in mangled deck exports are repaired on load."""
    words = {
        word.word: word
        for word in anki_import.load_anki_words(CARDS_DIR / "exported_deck.txt")
    }
    ampersand = words["existente"].sentences[2]
    assert ampersand.spanish == "Pan & vino."
    assert ampersand.english == "Bread & wine."


@pytest.mark.parametrize(
    ("headword", "article"),
    [
        pytest.param("existente", "el", id="article"),
        pytest.param("solitario", None, id="no-article"),
    ],
)
def test_load_anki_export_keeps_article(headword: str, article: str | None) -> None:
    """The Article field of a deck export is loaded, and an empty one is None."""
    words = {
        word.word: word
        for word in anki_import.load_anki_words(CARDS_DIR / "exported_deck.txt")
    }
    assert words[headword].article == article


def test_annotations_csv_has_no_header_by_default(annotations_run: CliRun) -> None:
    with annotations_run.output_csv.open(encoding="utf-8", newline="") as f:
        first_row = next(csv.reader(f))
    assert first_row[:2] != ["rank", "word"]


def test_annotations_csv_header_flag_adds_header(
    dictionary_paths: DictionaryPaths, tmp_path_factory: pytest.TempPathFactory
) -> None:
    workdir = tmp_path_factory.mktemp("header-run")
    run = _run_annotations_to_csv(
        workdir,
        CARDS_DIR / "combined_annotations.json",
        dictionary_paths,
        extra_args=("--csv-header",),
    )
    with run.output_csv.open(encoding="utf-8", newline="") as f:
        first_row = next(csv.reader(f))
    assert first_row[:2] == ["rank", "word"]


def test_main_output_has_no_header_by_default(main_run: CliRun) -> None:
    with main_run.output_csv.open(encoding="utf-8", newline="") as f:
        first_row = next(csv.reader(f))
    assert first_row[:2] != ["rank", "word"]


def test_annotations_limit(
    dictionary_paths: DictionaryPaths, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """--limit 1 processes only the first annotation entry and suppresses
    the processed-pages output, which would otherwise mark every loaded
    page as processed."""
    workdir = tmp_path_factory.mktemp("annotations-limit-run")
    run = _run_annotations_to_csv(
        workdir,
        CARDS_DIR / "combined_annotations.json",
        dictionary_paths,
        extra_args=("--limit", "1"),
    )
    assert "Limiting to the first 1 of 5 entries." in run.stdout
    assert set(_words_by_headword(run.output_csv)) == {"gatos", "fiambres (comida)"}
    assert not (workdir / "processed_pages.json").exists()


def test_annotations_every_prompt_failing_aborts(
    dictionary_paths: DictionaryPaths, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """When nothing gets through, the run stops with an error rather than
    writing an empty CSV and marking every page processed."""
    workdir = tmp_path_factory.mktemp("annotations-apierror-run")
    combined = workdir / "combined.json"
    combined.write_text(
        json.dumps(
            [
                {
                    "book": {
                        "title": "Libro",
                        "author": "Autora",
                        "path": "Autora/Libro (1)",
                        "language": "spa",
                    },
                    "annotations": [
                        {
                            "highlights": ["apierrorado"],
                            "context": "El apierrorado nunca recibió respuesta.",
                            "chapters": ["Capítulo 1"],
                            "page_numbers": [1],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = _run_annotations_to_csv_raw(
        workdir, combined, dictionary_paths, workdir / "words.csv"
    )

    assert result.returncode == 1, result.stdout
    assert "AllRequestsFailedError" in result.stderr
    assert "the fake provider rejected this request" in result.stderr
    assert not (workdir / "words.csv").exists()
    assert not (workdir / "processed_pages.json").exists()


def test_main_dropped_request_exits_4(
    dictionary_paths: DictionaryPaths, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A request the model never answers leaves its words out of the output
    and is reported with a distinct exit status, as #11 asks."""
    workdir = tmp_path_factory.mktemp("main-apierror-run")
    inputs = workdir / "words.csv"
    # Fifteen words fill the first prompt; the failing one gets the second.
    rows = [f",palabra{n},nm,word number {n},,,,,el,,,,," for n in range(15)]
    rows.append(",apierrorado,nm,word the fake provider rejects,,,,,el,,,,,")
    inputs.write_text(
        "rank,word,part_of_speech,meaning,spanish_sentence,english_sentence,"
        "frequency,image,article,sentence_source,note,example_sentences,"
        "model_used,tags\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    result = _run_main_raw(workdir, inputs, dictionary_paths)

    assert result.returncode == 4, result.stdout + result.stderr
    assert (
        "WARNING: 1 request(s) failed after every retry, so 1 word(s) got no "
        "sentences and are not in the output:"
    ) in result.stdout
    assert "apierrorado: ModelError: the fake provider rejected" in result.stdout
    words = _words_by_headword(workdir / "output" / "output.csv")
    assert "apierrorado" not in words
    assert {f"palabra{n}" for n in range(15)} <= set(words)


def test_main_limit(
    dictionary_paths: DictionaryPaths, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """--limit 1 sends only the first input word to the model."""
    run = _run_main(
        tmp_path_factory.mktemp("main-limit-run"),
        CARDS_DIR / "words.csv",
        dictionary_paths,
        extra_args=("--limit", "1"),
    )
    assert "Limiting to the first 1 of 4 input words." in run.stdout
    assert set(_words_by_headword(run.output_csv)) == {"gato"}


def _cached_annotation_prompts(workdir: pathlib.Path) -> Sequence[str]:
    """Everything sent to the model for each cached request, system prompt
    included."""
    cache_dir = workdir / ".cache" / "annotations"
    cached_inputs = [
        json.loads(path.read_text(encoding="utf-8"))["input"]
        for path in cache_dir.rglob("*.json")
    ]
    return [
        (cached["system_prompt"] or "") + cached["prompt"] for cached in cached_inputs
    ]


def test_default_dialect_reaches_the_prompt(annotations_run: CliRun) -> None:
    """With no [sentences] config, the prompt asks for international Spanish
    and for native Spanish punctuation."""
    cached_prompts = _cached_annotation_prompts(annotations_run.workdir)
    assert cached_prompts
    for prompt in cached_prompts:
        assert "international Spanish" in prompt
        assert "—Dime algo que no sepa —dijo Ramón." in prompt


def test_configured_dialect_reaches_the_prompt(
    dictionary_paths: DictionaryPaths, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The [sentences] dialect from config.toml lands in the prompt."""
    workdir = tmp_path_factory.mktemp("dialect-run")
    _run_annotations_to_csv(
        workdir,
        CARDS_DIR / "combined_annotations.json",
        dictionary_paths,
        dialect="Martian Spanish",
    )
    cached_prompts = _cached_annotation_prompts(workdir)
    assert cached_prompts
    for prompt in cached_prompts:
        assert "Martian Spanish" in prompt


#####
# Deduplication in the main pipeline
def _run_main_raw(
    workdir: pathlib.Path,
    inputs: pathlib.Path,
    dictionaries: DictionaryPaths,
    extra_args: Sequence[str] = (),
) -> subprocess.CompletedProcess:
    """Like _run_main, but reusable in one workdir and tolerant of failure."""
    config_path = _write_config(workdir, dictionaries)
    output_dir = workdir / "output"
    output_dir.mkdir(exist_ok=True)
    return subprocess.run(
        (
            sys.executable,
            "-m",
            "anki_lang_cards",
            "--inputs",
            str(inputs),
            "--output",
            str(output_dir / "output.csv"),
            "--model-name",
            "fake-cards",
            "--config",
            str(config_path),
            "--exported-deck",
            str(CARDS_DIR / "exported_deck.txt"),
            "--diff-output",
            str(output_dir / "diff.csv"),
            *extra_args,
        ),
        capture_output=True,
        text=True,
        cwd=workdir,
    )


def test_main_run_twice_emits_nothing_new(
    dictionary_paths: DictionaryPaths, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A rerun is a no-op: the output CSV read back is the resume state."""
    workdir = tmp_path_factory.mktemp("rerun")
    first = _run_main_raw(workdir, CARDS_DIR / "words.csv", dictionary_paths)
    assert first.returncode == 0, first.stderr
    output_csv = workdir / "output" / "output.csv"
    before = output_csv.read_bytes()

    second = _run_main_raw(workdir, CARDS_DIR / "words.csv", dictionary_paths)

    assert second.returncode == 0, second.stderr
    assert output_csv.read_bytes() == before
    assert "Found 0 new or updated words." in second.stdout


def _conflicting_inputs(workdir: pathlib.Path) -> pathlib.Path:
    """An input that collides with the deck's "existente" without matching it."""
    path = workdir / "conflicting.csv"
    path.write_text(
        "rank,word,part_of_speech,meaning,spanish_sentence,english_sentence,"
        "frequency,image,article,sentence_source,note,example_sentences,"
        "model_used,tags\n"
        ",existente,nm,extant word,,,,,el,,,,,\n"
    )
    return path


def test_main_exits_3_on_an_unanswered_duplicate(
    dictionary_paths: DictionaryPaths, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Headless, an unanswered group stops the run before the LLM is used."""
    workdir = tmp_path_factory.mktemp("undecided")
    inputs = _conflicting_inputs(workdir)

    result = _run_main_raw(workdir, inputs, dictionary_paths, ("--no-ui",))

    assert result.returncode == 3, result.stderr
    assert "Generating sentences" not in result.stdout
    assert not (workdir / "output" / "output.csv").exists()
    pending = json.loads((workdir / "output" / "merge_candidates.json").read_text())
    assert len(pending) == 1
    assert "existente" in pending[0]["words"]


def _answer_pending(workdir: pathlib.Path, **answer: Any) -> pathlib.Path:
    """Answer every pending group the same way, as a harness would."""
    pending_path = workdir / "output" / "merge_candidates.json"
    pending = json.loads(pending_path.read_text())
    for entry in pending:
        entry.update(answer)
    pending_path.write_text(json.dumps(pending, ensure_ascii=False))
    return pending_path


def test_main_folds_in_an_answered_decision_file(
    dictionary_paths: DictionaryPaths, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The headless loop: exit 3, answer the pending file, re-run with it."""
    workdir = tmp_path_factory.mktemp("decision-file")
    inputs = _conflicting_inputs(workdir)
    first = _run_main_raw(workdir, inputs, dictionary_paths, ("--no-ui",))
    assert first.returncode == 3, first.stderr
    assert "--decision-file output/merge_candidates.json" in first.stdout

    pending_path = _answer_pending(workdir, disposition="merge")
    second = _run_main_raw(
        workdir,
        inputs,
        dictionary_paths,
        ("--no-ui", "--decision-file", str(pending_path)),
    )

    assert second.returncode == 0, second.stdout + second.stderr
    assert "Folded in 1 new answers" in second.stdout
    words = types.Word.words_from_csv(workdir / "output" / "output.csv")
    assert [word.word for word in words].count("existente") == 1

    # The answer is in the cache now, so later runs need no file at all.
    third = _run_main_raw(workdir, inputs, dictionary_paths, ("--no-ui",))
    assert third.returncode == 0, third.stdout + third.stderr


def test_main_leaves_review_groups_pending(
    dictionary_paths: DictionaryPaths, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A group left as "review" is not an answer: the run stops again."""
    workdir = tmp_path_factory.mktemp("still-undecided")
    inputs = _conflicting_inputs(workdir)
    _run_main_raw(workdir, inputs, dictionary_paths, ("--no-ui",))

    pending_path = _answer_pending(workdir, disposition="review")
    result = _run_main_raw(
        workdir,
        inputs,
        dictionary_paths,
        ("--no-ui", "--decision-file", str(pending_path)),
    )

    assert result.returncode == 3, result.stdout + result.stderr
    assert "Folded in 0 new answers" in result.stdout
    assert len(json.loads(pending_path.read_text())) == 1


def test_main_rejects_a_broken_decision_file(
    dictionary_paths: DictionaryPaths, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Every problem in the file is named, and nothing is recorded."""
    workdir = tmp_path_factory.mktemp("broken-decision-file")
    inputs = _conflicting_inputs(workdir)
    _run_main_raw(workdir, inputs, dictionary_paths, ("--no-ui",))

    pending_path = _answer_pending(workdir, disposition="split")
    result = _run_main_raw(
        workdir,
        inputs,
        dictionary_paths,
        ("--no-ui", "--decision-file", str(pending_path)),
    )

    assert result.returncode == 2
    assert "a split needs at least two replacement rows" in result.stderr
    assert not (workdir / ".cache" / "merges" / "decisions.json").exists()


def test_main_replays_a_recorded_answer_headlessly(
    dictionary_paths: DictionaryPaths, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A decision recorded anywhere (UI, file, another script) settles the
    pipeline's question too: the caches are shared."""
    from anki_lang_cards import triage

    workdir = tmp_path_factory.mktemp("recorded")
    inputs = _conflicting_inputs(workdir)

    deck_copy = types.Word(
        rank=1400,
        word="existente",
        meaning="existing word",
        part_of_speech="nm",
        article="el",
        sentences=(),
    )
    incoming = types.Word(
        rank=None,
        word="existente",
        meaning="extant word",
        part_of_speech="nm",
        article="el",
        sentences=(),
    )
    decisions = triage.DecisionCache(workdir / ".cache" / "merges" / "decisions.json")
    decisions.remember(
        triage.Candidate(
            words=(deck_copy, incoming),
            reason=triage.Reason.COMPATIBLE,
            comparison=definition_similarity.Comparison(1.0, frozenset()),
            disposition=triage.Disposition.MERGE,
            replacements=(
                attrs.evolve(deck_copy, meaning="existing word, extant word"),
            ),
        )
    )
    decisions.save()

    result = _run_main_raw(workdir, inputs, dictionary_paths, ("--no-ui",))

    assert result.returncode == 0, result.stderr
    words = _words_by_headword(workdir / "output" / "output.csv")
    assert words["existente"].meaning == "existing word, extant word"
    assert words["existente"].rank == 1400
