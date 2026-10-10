"""The query runner, which answers requests through the response cache.

Cache misses go to a backend, either as one batch job or as concurrent
requests. Backends only appear here through the protocols in
:mod:`.backends.base`.
"""

import asyncio
import json
import pathlib
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Final, Self, cast

import attrs

from . import caches
from .backends import base


@attrs.frozen
class BatchJournal:
    """On-disk record of submitted batch jobs whose results are uncollected.

    One small JSON file per job lives in ``directory``. A file is written
    when a job is submitted and removed once its responses are in the
    response cache, so a run that dies in between can pick the job back up
    instead of paying for the same requests twice.
    """

    directory: pathlib.Path

    def pending(self) -> Mapping[str, frozenset[str]]:
        """Map each uncollected job's ID to its request keys."""
        if not self.directory.exists():
            return {}
        out: dict[str, frozenset[str]] = {}
        for path in sorted(self.directory.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            out[record["job_id"]] = frozenset(record["request_keys"])
        return out

    def record(self, job_id: str, request_keys: frozenset[str]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(job_id).write_text(
            json.dumps(
                {"job_id": job_id, "request_keys": sorted(request_keys)}, indent=2
            ),
            encoding="utf-8",
        )

    def discard(self, job_id: str) -> None:
        self._path(job_id).unlink(missing_ok=True)

    def _path(self, job_id: str) -> pathlib.Path:
        return self.directory / f"{job_id}.json"


_ABORTING_STATUSES: Final[frozenset[int]] = frozenset({401, 403})
"""HTTP statuses that mean no request will get through: bad or missing
credentials, or no access to the model."""

_PROGRAMMING_ERRORS: Final[tuple[type[Exception], ...]] = (
    AssertionError,
    AttributeError,
    NameError,
    TypeError,
)


def aborts_run(error: Exception) -> bool:
    """Whether a backend error should stop the run rather than fail one request.

    Most errors from a backend — a 400 for one prompt, a server error, a
    plugin choking on one input — say nothing about the other requests, so
    they become a :class:`Failure` and the rest of the run carries on and
    gets cached. Errors every request would hit the same way stop the run
    instead, because carrying on would only fail (and maybe bill) each one in
    turn: configuration errors, authentication or permission errors (an HTTP
    401 or 403, whether the client reports it as ``status_code`` like the
    Anthropic SDK or ``status`` like aiohttp), and programming errors, which
    are bugs to fix rather than responses to retry.

    Only :class:`Exception` reaches this: ``KeyboardInterrupt`` and
    :class:`asyncio.CancelledError` always propagate.
    """
    if isinstance(error, (base.ConfigurationError, *_PROGRAMMING_ERRORS)):
        return True
    for attribute in ("status_code", "status"):
        status = getattr(error, attribute, None)
        if isinstance(status, int) and status in _ABORTING_STATUSES:
            return True
    return False


@attrs.frozen
class Success:
    request: base.Request
    response: caches.ResponseData


@attrs.frozen
class Failure:
    request: base.Request
    reason: str


class QueryRunner:
    """Answers requests from the response cache, sending misses to a backend.

    Misses go out as one batch job when ``batch`` is set (the backend must
    support :attr:`base.Feature.BATCH`), or as concurrent requests otherwise.
    Entering the runner enters its backend.
    """

    def __init__(
        self,
        backend: base.Backend,
        cacher: caches.QueryCacher,
        *,
        batch: bool = False,
        max_concurrency: int = 5,
    ):
        if batch and base.Feature.BATCH not in backend.features:
            raise ValueError(f"The {backend.service} backend cannot batch requests.")
        self.backend = backend
        self.cacher = cacher
        self.batch = batch
        if backend.max_concurrency is not None:
            max_concurrency = min(max_concurrency, backend.max_concurrency)
        self.max_concurrency = max_concurrency

    @property
    def journal(self) -> BatchJournal:
        return BatchJournal(self.cacher.cache_dir / "batch-jobs")

    async def __aenter__(self) -> Self:
        await self.backend.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.backend.__aexit__(exc_type, exc_val, exc_tb)

    async def run(
        self,
        requests: Sequence[base.Request],
        *,
        invalidate_cache: bool = False,
        skip_cache: bool = False,
    ) -> AsyncIterator[Success | Failure]:
        """Yield an outcome for each distinct request, in no particular order.

        :param invalidate_cache: Ignore cached responses, but cache the new ones.
        :param skip_cache: Neither read nor write the response cache.
        """
        misses: list[base.Request] = []
        for request in dict.fromkeys(requests):
            if not (invalidate_cache or skip_cache):
                cached = self.cacher.load_from_cache(
                    request.prompt, system_prompt=request.system_prompt
                )
                if cached is not None:
                    yield Success(request, cached["response"])
                    continue
            misses.append(request)

        if not misses:
            return

        if self.batch:
            outcomes = self._run_batch(misses, write_cache=not skip_cache)
        else:
            outcomes = self._run_concurrently(misses, write_cache=not skip_cache)
        async for outcome in outcomes:
            yield outcome

    async def _run_concurrently(
        self, requests: Sequence[base.Request], *, write_cache: bool
    ) -> AsyncIterator[Success | Failure]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_one(request: base.Request) -> Success | Failure:
            async with semaphore:
                try:
                    response = await self.backend.query(request)
                except base.IncompleteResponseError as e:
                    return Failure(request, str(e))
                except Exception as e:
                    if aborts_run(e):
                        raise
                    return Failure(request, f"{type(e).__name__}: {e}")
            if write_cache:
                self._cache(request, response)
            return Success(request, response)

        for outcome in asyncio.as_completed(map(run_one, requests)):
            yield await outcome

    async def _run_batch(
        self, requests: Sequence[base.Request], *, write_cache: bool
    ) -> AsyncIterator[Success | Failure]:
        backend = cast(base.BatchBackend, self.backend)
        journal = self.journal
        results = await backend.query_batch(
            requests, resume=journal.pending(), on_submit=journal.record
        )

        # Everything is in the response cache before any job leaves the
        # journal, so a crash here never loses a paid-for response.
        outcomes: list[Success | Failure] = []
        for request in requests:
            if (response := results.responses.get(request.key)) is not None:
                if write_cache:
                    self._cache(request, response)
                outcomes.append(Success(request, response))
            else:
                reason = results.failures.get(request.key, "no response")
                outcomes.append(Failure(request, reason))
        for job_id in results.completed_jobs:
            journal.discard(job_id)

        for outcome in outcomes:
            yield outcome

    def _cache(self, request: base.Request, response: caches.ResponseData) -> None:
        self.cacher.cache_result(
            request.prompt,
            response,
            system_prompt=request.system_prompt,
            query_params={**self.backend.query_params(), "batch": self.batch},
        )


async def run_with_retries[T](
    runner: QueryRunner,
    requests: Sequence[base.Request],
    parse: Callable[[base.Request, caches.ResponseData], T],
    *,
    max_retries: int,
    describe: Callable[[base.Request], str],
    invalidate_cache: bool = False,
    skip_cache: bool = False,
) -> AsyncIterator[tuple[base.Request, T]]:
    """Run requests and parse the responses, retrying any that fail.

    A request fails if the backend's response was incomplete, the backend
    raised an error that does not abort the run (see :func:`aborts_run`), or
    ``parse`` raises :class:`ValueError` (which includes JSON decoding
    errors).
    Retries skip the response cache, since the bad response is in it, and run
    together, so in batch mode each round of retries is one more batch job.
    """
    pending = requests
    attempt = 0
    while pending:
        attempt += 1
        failed: list[base.Request] = []
        async for outcome in runner.run(
            pending,
            invalidate_cache=invalidate_cache and not skip_cache,
            skip_cache=skip_cache,
        ):
            match outcome:
                case Failure(request=request, reason=reason):
                    print(f"Attempt {attempt} failed for {describe(request)}: {reason}")
                    failed.append(request)
                case Success(request=request, response=response):
                    try:
                        parsed = parse(request, response)
                    except ValueError as e:
                        print(f"Attempt {attempt} failed for {describe(request)}: {e}")
                        failed.append(request)
                    else:
                        yield request, parsed

        if failed and attempt <= max_retries:
            print(f"Retrying {len(failed)} request(s) after invalid output.")
            pending = failed
            invalidate_cache = True
        else:
            for request in failed:
                print(f"Skipping {describe(request)} after {attempt} attempt(s).")
            pending = ()
