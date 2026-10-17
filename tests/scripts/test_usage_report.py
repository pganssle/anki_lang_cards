"""Tests for the usage report, against synthetic response caches."""

import datetime
import importlib.util
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any, Final

import pytest
from hypothesis import given
from hypothesis import strategies as st

from anki_lang_cards import runner

SCRIPT: Final = (
    pathlib.Path(__file__).parent.parent.parent / "scripts" / "usage_report.py"
)

START: Final = datetime.datetime(2026, 10, 10, 9, 0)
SONNET: Final = "anthropic/claude-sonnet-5"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("usage_report_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_entry(
    root: pathlib.Path,
    name: str,
    *,
    cache_type: str = "annotations",
    model: str = SONNET,
    timestamp: datetime.datetime = START,
    input_tokens: int = 1000,
    output_tokens: int = 100,
    details: Mapping[str, Any] | None = None,
    query_params: Mapping[str, Any] | None = None,
) -> pathlib.Path:
    path = root / cache_type / "anthropic" / model / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "input": {
                    "model_name": model,
                    "model_service": "anthropic",
                    "prompt": name,
                    "system_prompt": None,
                    "query_params": dict(query_params or {}),
                    "timestamp": timestamp.isoformat(),
                },
                "response": {
                    "text": "{}",
                    "usage": {
                        "input": input_tokens,
                        "output": output_tokens,
                        "details": None if details is None else dict(details),
                    },
                },
            }
        )
    )
    return path


def sdk_details(
    *, batch: bool, write: int = 0, read: int = 0, **extra: int
) -> Mapping[str, Any]:
    return {
        "cache_creation_input_tokens": write,
        "cache_read_input_tokens": read,
        "batch": batch,
        "message_id": "msg_1",
        **extra,
    }


#####
# Pricing one response


@pytest.mark.parametrize(
    ("model", "details", "query_params", "expected"),
    [
        # llm-era response: plain input and output at list price.
        (SONNET, None, None, (2 * 1000 + 10 * 100) / 1e6),
        # Batched, older entry without the write split: the write is assumed
        # to be at the hour-long lifetime.
        (
            SONNET,
            sdk_details(batch=True, write=2000, read=3000),
            {"batch": True},
            0.5 * (2 * (1000 + 2.0 * 2000 + 0.1 * 3000) + 10 * 100) / 1e6,
        ),
        # The warm-up request ahead of a batch: billed at full price, but
        # writing the hour-long entry the batch reads.
        (
            SONNET,
            sdk_details(batch=False, write=2000),
            {"batch": True},
            (2 * (1000 + 2.0 * 2000) + 10 * 100) / 1e6,
        ),
        # Not batched, no split recorded: the five-minute lifetime.
        (
            SONNET,
            sdk_details(batch=False, write=2000, read=500),
            {"batch": False},
            (2 * (1000 + 1.25 * 2000 + 0.1 * 500) + 10 * 100) / 1e6,
        ),
        # Cache reads are priced per model, not as a share of the input
        # price: Fable 5.1 reads at $0.25/MTok against $10 input.
        (
            "anthropic/claude-fable-5-1",
            sdk_details(batch=False, write=2000, read=40_000),
            {"batch": False},
            (10 * (1000 + 1.25 * 2000) + 0.25 * 40_000 + 50 * 100) / 1e6,
        ),
        # A recorded split wins over any assumption.
        (
            SONNET,
            sdk_details(
                batch=True,
                write=3000,
                cache_creation_5m_input_tokens=1000,
                cache_creation_1h_input_tokens=2000,
            ),
            {"batch": True},
            0.5 * (2 * (1000 + 1.25 * 1000 + 2.0 * 2000) + 10 * 100) / 1e6,
        ),
    ],
)
def test_entry_cost(
    script,
    tmp_path: pathlib.Path,
    model: str,
    details: Mapping[str, Any] | None,
    query_params: Mapping[str, Any] | None,
    expected: float,
) -> None:
    write_entry(tmp_path, "a", model=model, details=details, query_params=query_params)

    (entry,) = script.load_entries(tmp_path).entries

    assert entry.cost() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("model", "price"),
    [
        ("anthropic/claude-sonnet-5", (2.0, 10.0, 0.2)),
        ("claude-sonnet-5", (2.0, 10.0, 0.2)),
        ("anthropic/claude-fable-5-1", (10.0, 50.0, 0.25)),
        ("anthropic/claude-opus-5-5", (4.0, 20.0, 0.20)),
        ("claude-3-5-sonnet-20241022", (3.0, 15.0, 0.3)),
        # The llm library's spellings, as older runs recorded them.
        ("claude-sonnet-4.5", (3.0, 15.0, 0.3)),
        ("anthropic/claude-3.7-sonnet-latest", (3.0, 15.0, 0.3)),
    ],
)
def test_price_for(script, model: str, price: tuple[float, float, float]) -> None:
    assert script.price_for(model) == script.Price(*price)


def test_cache_read_defaults_to_a_tenth_of_input(script) -> None:
    assert script.Price(3.0, 15.0).cache_read == pytest.approx(0.3)


def test_unknown_model_is_not_priced(script, tmp_path: pathlib.Path) -> None:
    write_entry(tmp_path, "a", model="claude-sonnet-9")
    write_entry(tmp_path, "b")

    loaded = script.load_entries(tmp_path)
    runs = script.group_runs(loaded.entries, datetime.timedelta(minutes=30))
    totals = script.Totals.of(loaded.entries)

    unknown = next(run for run in runs if run.model == "claude-sonnet-9")
    assert unknown.totals().to_json()["cost"] is None
    assert totals.unpriced == 1
    assert totals.cost == pytest.approx((2 * 1000 + 10 * 100) / 1e6)
    assert totals.input_tokens == 2000
    table = script.format_table(runs, len(loaded.skipped))
    assert "unknown" in table
    assert "+?" in table


#####
# Reading the cache


def test_skips_journal_and_non_responses(script, tmp_path: pathlib.Path) -> None:
    write_entry(tmp_path, "real")
    journal = tmp_path / "annotations" / "anthropic" / SONNET / runner.BATCH_JOURNAL_DIR
    journal.mkdir()
    (journal / "msgbatch_1.json").write_text(
        json.dumps({"job_id": "msgbatch_1", "request_keys": ["k"]})
    )
    (tmp_path / "annotations" / "list.json").write_text("[1, 2]")
    (tmp_path / "annotations" / "broken.json").write_text("{not json")
    (tmp_path / "sentences").mkdir()
    (tmp_path / "sentences" / "partial.json").write_text(json.dumps({"input": {}}))
    # Other caches under the root are not LLM responses at all.
    (tmp_path / "merges").mkdir()
    (tmp_path / "merges" / "decisions.json").write_text("{}")

    loaded = script.load_entries(tmp_path)

    assert len(loaded.entries) == 1
    assert sorted(skipped.path for skipped in loaded.skipped) == [
        tmp_path / "annotations" / "broken.json",
        tmp_path / "annotations" / "list.json",
        tmp_path / "sentences" / "partial.json",
    ]


@pytest.mark.parametrize("verbose", [True, False])
def test_verbose_names_skipped_files(
    script, tmp_path: pathlib.Path, capsys, verbose: bool
) -> None:
    write_entry(tmp_path, "real")
    partial = tmp_path / "annotations" / "partial.json"
    partial.write_text(json.dumps({"input": {}}))
    broken = tmp_path / "annotations" / "broken.json"
    broken.write_text("{not json")

    args = ["--cache-root", str(tmp_path), *(["--verbose"] if verbose else [])]
    assert script.main(args) == 0

    out, err = capsys.readouterr()
    assert "Skipped 2 files" in out
    assert (str(partial) in err) == verbose
    assert (str(broken) in err) == verbose


def test_missing_cache_root(script, tmp_path: pathlib.Path, capsys) -> None:
    assert script.main(["--cache-root", str(tmp_path / "nowhere")]) == 0
    assert "No cached responses" in capsys.readouterr().out


def test_since(script, tmp_path: pathlib.Path) -> None:
    write_entry(tmp_path, "old", timestamp=START - datetime.timedelta(days=1))
    write_entry(tmp_path, "new", timestamp=START)

    loaded = script.load_entries(tmp_path, since=START)

    assert [entry.timestamp for entry in loaded.entries] == [START]


@pytest.mark.parametrize(
    "value", ["2026-10-10", "2026-10-10T09:00", "2026-10-10 09:00"]
)
def test_since_option_parses_dates(
    script, tmp_path: pathlib.Path, capsys, value: str
) -> None:
    write_entry(tmp_path, "old", timestamp=START - datetime.timedelta(days=1))
    write_entry(tmp_path, "new", timestamp=START)

    script.main(["--cache-root", str(tmp_path), "--since", value, "--json"])

    report = json.loads(capsys.readouterr().out)
    assert report["totals"]["requests"] == 1


def test_since_option_rejects_garbage(script, capsys) -> None:
    with pytest.raises(SystemExit):
        script.main(["--since", "last tuesday"])
    assert "expected a date or datetime" in capsys.readouterr().err


#####
# Grouping into runs


def test_group_runs(script, tmp_path: pathlib.Path) -> None:
    minutes = datetime.timedelta(minutes=1)
    for offset in (0, 10, 35, 100):
        write_entry(tmp_path, f"a{offset}", timestamp=START + offset * minutes)
    write_entry(tmp_path, "s", cache_type="sentences", timestamp=START + 5 * minutes)
    write_entry(tmp_path, "o", model="anthropic/claude-opus-5", timestamp=START)

    runs = script.group_runs(
        script.load_entries(tmp_path).entries, datetime.timedelta(minutes=30)
    )

    summary = [
        (run.cache_type, run.model, len(run.entries), run.start - START) for run in runs
    ]
    assert sorted(summary) == sorted(
        [
            ("annotations", SONNET, 3, datetime.timedelta(0)),
            ("annotations", SONNET, 1, 100 * minutes),
            ("annotations", "anthropic/claude-opus-5", 1, datetime.timedelta(0)),
            ("sentences", SONNET, 1, 5 * minutes),
        ]
    )
    assert [run.start for run in runs] == sorted(run.start for run in runs)


@given(
    offsets=st.lists(st.integers(min_value=0, max_value=10_000), max_size=40),
    gap=st.integers(min_value=0, max_value=500),
)
def test_grouping_invariants(script, offsets: Sequence[int], gap: int) -> None:
    entries = [
        script.Entry(
            cache_type="annotations",
            model=SONNET,
            timestamp=START + datetime.timedelta(minutes=offset),
            input_tokens=1,
            output_tokens=1,
        )
        for offset in offsets
    ]
    threshold = datetime.timedelta(minutes=gap)

    runs = script.group_runs(entries, threshold)

    # Every response lands in exactly one run.
    assert sorted(entry.timestamp for run in runs for entry in run.entries) == sorted(
        entry.timestamp for entry in entries
    )
    for run in runs:
        times = [entry.timestamp for entry in run.entries]
        assert times == sorted(times)
        assert all(b - a <= threshold for a, b in zip(times, times[1:]))
    # Consecutive runs are separated by more than the gap.
    for earlier, later in zip(runs, runs[1:]):
        assert later.start - earlier.end > threshold


#####
# Output


def test_hit_rate(script, tmp_path: pathlib.Path) -> None:
    write_entry(tmp_path, "a", details=sdk_details(batch=False, write=3000))
    write_entry(tmp_path, "b", details=sdk_details(batch=False, read=3000))
    write_entry(tmp_path, "c", details=sdk_details(batch=False, read=3000))

    totals = script.Totals.of(script.load_entries(tmp_path).entries)

    assert totals.hit_rate == pytest.approx(2 / 3)
    assert totals.cache_write == 3000
    assert totals.cache_read == 6000


def test_hit_rate_without_caching(script) -> None:
    assert script.Totals.of([]).hit_rate is None


def test_json_report(script, tmp_path: pathlib.Path, capsys) -> None:
    write_entry(
        tmp_path,
        "a",
        details=sdk_details(batch=True, write=2000),
        query_params={"batch": True},
    )
    write_entry(
        tmp_path,
        "b",
        timestamp=START + datetime.timedelta(minutes=5),
        details=sdk_details(batch=True, read=2000),
        query_params={"batch": True},
    )
    (tmp_path / "annotations" / "junk.json").write_text("[]")

    assert script.main(["--cache-root", str(tmp_path), "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    (run,) = report["runs"]
    assert run["pipeline"] == "annotations"
    assert run["model"] == SONNET
    assert run["start"] == START.isoformat()
    assert run["requests"] == 2
    assert run["batched"] == 2
    assert run["cache_write"] == 2000
    assert run["cache_read"] == 2000
    assert run["hit_rate"] == pytest.approx(0.5)
    expected = 0.5 * (2 * (2000 + 2.0 * 2000 + 0.1 * 2000) + 10 * 200) / 1e6
    assert run["cost"] == pytest.approx(expected)
    assert report["totals"]["cost"] == pytest.approx(expected)
    assert report["skipped"] == 1


def test_table_report(script, tmp_path: pathlib.Path, capsys) -> None:
    write_entry(tmp_path, "a", input_tokens=1_234_567)

    assert script.main(["--cache-root", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "1,234,567" in out
    assert "annotations" in out
    assert out.count("total") == 1


def test_since_option_accepts_an_offset(script, tmp_path: pathlib.Path, capsys) -> None:
    """`date -Iseconds` gives an offset; cache timestamps are local and naïve."""
    write_entry(tmp_path, "old", timestamp=START - datetime.timedelta(days=1))
    write_entry(tmp_path, "new", timestamp=START)
    since = START.astimezone().isoformat()

    script.main(["--cache-root", str(tmp_path), "--since", since, "--json"])

    report = json.loads(capsys.readouterr().out)
    assert report["totals"]["requests"] == 1
