"""Summarize what LLM runs cost, from the response cache.

Every response the pipelines get back is cached with its token counts, so
the cache doubles as a usage log: this reads it, groups the responses into
runs and prices them. It makes no API calls.
"""

import argparse
import datetime
import itertools
import json
import pathlib
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final, Self

import attrs

from anki_lang_cards import caches, runner

RESPONSE_CACHE_TYPES: Final = (
    caches.CacheTypes.ANNOTATIONS,
    caches.CacheTypes.SENTENCES,
)
"""The caches that hold model responses with their usage.

``CacheTypes.RESPONSES`` is left out: ``org_to_csv`` queries with
``skip_cache=True`` and keeps only the CSV it parses out of each answer there,
so that directory holds no token counts.
"""

CACHE_WRITE_5M_MULTIPLIER: Final = 1.25
CACHE_WRITE_1H_MULTIPLIER: Final = 2.0
BATCH_MULTIPLIER: Final = 0.5


@attrs.frozen
class Price:
    """Dollars per million tokens."""

    input: float
    output: float
    cache_read: float = attrs.field(
        default=attrs.Factory(lambda self: self.input / 10, takes_self=True)
    )
    """The price of input read back from the prompt cache.

    Most models read at a tenth of the input price, but it is set per model:
    the newest ones read cheaper.
    """


PRICES: Final[Mapping[str, Price]] = {
    "claude-fable-5-1": Price(10.0, 50.0, cache_read=0.25),
    "claude-opus-5-5": Price(4.0, 20.0, cache_read=0.20),
    "claude-opus-5": Price(5.0, 25.0),
    "claude-sonnet-5": Price(2.0, 10.0),
    "claude-haiku-4-5": Price(1.0, 5.0),
    "claude-sonnet-4-5": Price(3.0, 15.0),
    "claude-sonnet-4-0": Price(3.0, 15.0),
    "claude-opus-4-0": Price(15.0, 75.0),
    "claude-3-7-sonnet-20250219": Price(3.0, 15.0),
    "claude-3-5-sonnet-20241022": Price(3.0, 15.0),
}


def price_for(model: str) -> Price | None:
    return PRICES.get(model.removeprefix("anthropic/"))


@attrs.frozen
class Entry:
    """One cached response's usage."""

    cache_type: str
    model: str
    """The model as the request named it, e.g. ``anthropic/claude-sonnet-5``."""

    timestamp: datetime.datetime
    input_tokens: int
    """Input that was neither written to nor read from the prompt cache."""

    output_tokens: int
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    cache_read: int = 0
    batch: bool = False
    """Whether this request was billed at the batch rate."""

    @property
    def cache_write(self) -> int:
        return self.cache_write_5m + self.cache_write_1h

    def cost(self) -> float | None:
        price = price_for(self.model)
        if price is None:
            return None
        input_cost = price.input * (
            self.input_tokens
            + CACHE_WRITE_5M_MULTIPLIER * self.cache_write_5m
            + CACHE_WRITE_1H_MULTIPLIER * self.cache_write_1h
        )
        cache_read_cost = price.cache_read * self.cache_read
        output_cost = price.output * self.output_tokens
        total = (input_cost + cache_read_cost + output_cost) / 1_000_000
        return total * BATCH_MULTIPLIER if self.batch else total


@attrs.frozen
class Totals:
    requests: int
    batched: int
    input_tokens: int
    cache_write: int
    cache_read: int
    output_tokens: int
    cost: float
    """The cost of the requests whose model has a known price."""

    unpriced: int
    """Requests whose model has no known price, so are missing from ``cost``."""

    @classmethod
    def of(cls, entries: Iterable[Entry]) -> Self:
        entries = tuple(entries)
        costs = [entry.cost() for entry in entries]
        return cls(
            requests=len(entries),
            batched=sum(entry.batch for entry in entries),
            input_tokens=sum(entry.input_tokens for entry in entries),
            cache_write=sum(entry.cache_write for entry in entries),
            cache_read=sum(entry.cache_read for entry in entries),
            output_tokens=sum(entry.output_tokens for entry in entries),
            cost=sum(cost for cost in costs if cost is not None),
            unpriced=sum(cost is None for cost in costs),
        )

    @property
    def hit_rate(self) -> float | None:
        """The share of cacheable input that was read rather than written."""
        cacheable = self.cache_read + self.cache_write
        return self.cache_read / cacheable if cacheable else None

    def to_json(self) -> dict[str, Any]:
        return {
            **attrs.asdict(self),
            "hit_rate": self.hit_rate,
            "cost": None if self.unpriced == self.requests else self.cost,
        }


@attrs.frozen
class Run:
    """Responses of one kind from one model, close together in time."""

    cache_type: str
    model: str
    entries: tuple[Entry, ...]

    @property
    def start(self) -> datetime.datetime:
        return self.entries[0].timestamp

    @property
    def end(self) -> datetime.datetime:
        return self.entries[-1].timestamp

    def totals(self) -> Totals:
        return Totals.of(self.entries)


def _entry(cache_type: str, data: Any) -> Entry | None:
    """Read one cache file's contents, or None if it is not a cached response."""
    try:
        request = data["input"]
        usage = data["response"]["usage"]
        timestamp = datetime.datetime.fromisoformat(request["timestamp"])
        input_tokens = int(usage["input"] or 0)
        output_tokens = int(usage["output"] or 0)
        model = str(request["model_name"])
    except (KeyError, TypeError, ValueError):
        return None

    details = usage.get("details")
    query_params = request.get("query_params") or {}
    if not isinstance(details, Mapping):
        # Responses from the `llm` library era carry no cache or batch detail.
        return Entry(cache_type, model, timestamp, input_tokens, output_tokens)

    if "cache_creation_5m_input_tokens" in details:
        write_5m = int(details["cache_creation_5m_input_tokens"] or 0)
        write_1h = int(details.get("cache_creation_1h_input_tokens") or 0)
    else:
        # Before the split was recorded, batch runs (including the request
        # sent ahead to warm the cache) wrote at the hour-long lifetime.
        written = int(details.get("cache_creation_input_tokens") or 0)
        if query_params.get("batch"):
            write_5m, write_1h = 0, written
        else:
            write_5m, write_1h = written, 0
    return Entry(
        cache_type=cache_type,
        model=model,
        timestamp=timestamp,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_write_5m=write_5m,
        cache_write_1h=write_1h,
        cache_read=int(details.get("cache_read_input_tokens") or 0),
        batch=bool(details.get("batch", query_params.get("batch", False))),
    )


@attrs.frozen
class Skipped:
    """A file under the cache that is not a readable cached response."""

    path: pathlib.Path
    reason: str


@attrs.frozen
class Loaded:
    entries: tuple[Entry, ...]
    skipped: tuple[Skipped, ...]


def load_entries(
    cache_root: pathlib.Path, since: datetime.datetime | None = None
) -> Loaded:
    entries: list[Entry] = []
    skipped: list[Skipped] = []
    for cache_type in RESPONSE_CACHE_TYPES:
        directory = cache_root / cache_type.value
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.json")):
            if runner.BATCH_JOURNAL_DIR in path.relative_to(directory).parts:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
                skipped.append(Skipped(path, f"unreadable: {e}"))
                continue
            entry = _entry(cache_type.value, data)
            if entry is None:
                skipped.append(Skipped(path, "not a cached response"))
            elif since is None or entry.timestamp >= since:
                entries.append(entry)
    return Loaded(tuple(entries), tuple(skipped))


def group_runs(entries: Iterable[Entry], gap: datetime.timedelta) -> Sequence[Run]:
    """Split each pipeline and model's responses wherever they pause for longer
    than ``gap``."""

    def key(entry: Entry) -> tuple[str, str]:
        return (entry.cache_type, entry.model)

    runs: list[Run] = []
    for (cache_type, model), group in itertools.groupby(
        sorted(entries, key=lambda entry: (key(entry), entry.timestamp)), key=key
    ):
        current: list[Entry] = []
        for entry in group:
            if current and entry.timestamp - current[-1].timestamp > gap:
                runs.append(Run(cache_type, model, tuple(current)))
                current = []
            current.append(entry)
        runs.append(Run(cache_type, model, tuple(current)))
    return sorted(runs, key=lambda run: run.start)


def _format_cost(totals: Totals) -> str:
    if totals.unpriced == totals.requests:
        return "unknown"
    cost = f"${totals.cost:.2f}"
    return f"{cost}+?" if totals.unpriced else cost


def _format_rate(rate: float | None) -> str:
    return "-" if rate is None else f"{rate:.0%}"


def format_table(runs: Sequence[Run], skipped: int) -> str:
    header = (
        "start",
        "end",
        "pipeline",
        "model",
        "requests",
        "batched",
        "input",
        "cache write",
        "cache read",
        "output",
        "hit rate",
        "cost",
    )
    rows: list[Sequence[str]] = []
    for run in runs:
        totals = run.totals()
        rows.append(
            (
                run.start.strftime("%Y-%m-%d %H:%M"),
                run.end.strftime("%Y-%m-%d %H:%M"),
                run.cache_type,
                run.model,
                str(totals.requests),
                str(totals.batched),
                f"{totals.input_tokens:,}",
                f"{totals.cache_write:,}",
                f"{totals.cache_read:,}",
                f"{totals.output_tokens:,}",
                _format_rate(totals.hit_rate),
                _format_cost(totals),
            )
        )
    overall = Totals.of(entry for run in runs for entry in run.entries)
    rows.append(
        (
            "total",
            "",
            "",
            "",
            str(overall.requests),
            str(overall.batched),
            f"{overall.input_tokens:,}",
            f"{overall.cache_write:,}",
            f"{overall.cache_read:,}",
            f"{overall.output_tokens:,}",
            _format_rate(overall.hit_rate),
            _format_cost(overall),
        )
    )

    widths = [
        max(len(row[column]) for row in (header, *rows))
        for column in range(len(header))
    ]
    numeric = range(4, len(header))

    def line(row: Sequence[str]) -> str:
        return "  ".join(
            cell.rjust(width) if column in numeric else cell.ljust(width)
            for column, (cell, width) in enumerate(zip(row, widths, strict=True))
        ).rstrip()

    lines = [line(header), line(["-" * width for width in widths])]
    lines.extend(line(row) for row in rows[:-1])
    lines.append(line(["-" * width for width in widths]))
    lines.append(line(rows[-1]))
    if overall.unpriced:
        lines.append(
            f"{overall.unpriced} requests used models with no known price "
            "and are left out of the costs (marked +?)."
        )
    if skipped:
        lines.append(f"Skipped {skipped} files that are not cached responses.")
    return "\n".join(lines)


def to_json(runs: Sequence[Run], skipped: int) -> dict[str, Any]:
    return {
        "runs": [
            {
                "pipeline": run.cache_type,
                "model": run.model,
                "start": run.start.isoformat(),
                "end": run.end.isoformat(),
                **run.totals().to_json(),
            }
            for run in runs
        ],
        "totals": Totals.of(entry for run in runs for entry in run.entries).to_json(),
        "skipped": skipped,
    }


def _parse_since(value: str) -> datetime.datetime:
    try:
        since = datetime.datetime.fromisoformat(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"expected a date or datetime like 2026-10-10 or 2026-10-10T14:30, "
            f"not {value!r}"
        ) from e
    if since.tzinfo is not None:
        # Cache timestamps are naïve local times, so compare in those terms.
        since = since.astimezone().replace(tzinfo=None)
    return since


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Retrying a request overwrites its cached response, so the counts "
        "are of the responses that survive, not of every request sent: a run "
        "that needed retries cost somewhat more than reported.",
    )
    parser.add_argument(
        "--cache-root",
        type=pathlib.Path,
        default=caches.DEFAULT_CACHE_LOC,
        help="The response cache directory (default: %(default)s).",
    )
    parser.add_argument(
        "--gap-minutes",
        type=float,
        default=30.0,
        help="A pause longer than this between responses starts a new run "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--since",
        type=_parse_since,
        default=None,
        help="Only count responses from this date or datetime on.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON instead."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Name each skipped file on stderr. The report only counts them, "
        "so this is how to tell a stray file from a response that no longer "
        "parses and is going uncounted.",
    )
    args = parser.parse_args(argv)

    loaded = load_entries(args.cache_root, since=args.since)
    if args.verbose:
        for skipped in loaded.skipped:
            print(f"Skipped {skipped.path}: {skipped.reason}", file=sys.stderr)
    runs = group_runs(loaded.entries, datetime.timedelta(minutes=args.gap_minutes))
    if args.json:
        print(json.dumps(to_json(runs, len(loaded.skipped)), indent=2))
    elif not runs:
        print(f"No cached responses under {args.cache_root}.")
    else:
        print(format_table(runs, len(loaded.skipped)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
