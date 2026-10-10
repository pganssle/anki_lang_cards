"""Running many requests against a backend through the response cache."""

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Self, cast

import attrs

from .. import caches
from . import base


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
    def journal(self) -> base.BatchJournal:
        return base.BatchJournal(self.cacher.cache_dir / "batch-jobs")

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

    A request fails if the backend's response was incomplete or ``parse``
    raises :class:`ValueError` (which includes JSON decoding errors).
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
