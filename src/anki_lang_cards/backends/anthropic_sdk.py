"""Claude models through the Anthropic SDK, with batching and prompt caching."""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final, Self

import anthropic

from .. import caches
from . import base

DEFAULT_MAX_TOKENS: Final[int] = 16_000
"""Room for adaptive thinking as well as the response; thinking counts
against ``max_tokens``."""

_INCOMPLETE_STOP_REASONS: Final = frozenset({"max_tokens", "refusal"})


class AnthropicBackend:
    service: ClassVar[str] = "anthropic"
    features: ClassVar[frozenset[base.Feature]] = frozenset(
        {base.Feature.BATCH, base.Feature.PROMPT_CACHING}
    )
    default_words_per_prompt: ClassVar[int] = 15

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str | None = None,
        prompt_caching: bool = True,
        poll_interval: float = 10.0,
        max_poll_interval: float = 60.0,
        http_client: anthropic.DefaultAsyncHttpxClient | None = None,
    ):
        """
        :param api_key: If omitted, the SDK finds credentials itself (the
            ``ANTHROPIC_API_KEY`` environment variable, an ``ant auth login``
            profile, ...).
        :param effort: The ``output_config.effort`` level, or None for the
            model's default.
        :param poll_interval: Seconds between the first batch status checks;
            this doubles after each check, up to ``max_poll_interval``.
        """
        self.model_name = model_name
        self.max_concurrency: int | None = None
        self.max_tokens = max_tokens
        self.effort = effort
        self.prompt_caching = prompt_caching
        self.poll_interval = poll_interval
        self.max_poll_interval = max_poll_interval
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key, base_url=base_url, http_client=http_client
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._client.close()

    def query_params(self) -> Mapping[str, object]:
        return {
            "max_tokens": self.max_tokens,
            "effort": self.effort,
            "prompt_caching": self.prompt_caching,
        }

    async def query(self, request: base.Request) -> caches.ResponseData:
        message = await self._client.messages.create(
            **self._params(request, batch=False)
        )
        return _response_data(message, batch=False)

    async def query_batch(
        self,
        requests: Sequence[base.Request],
        *,
        resume: Mapping[str, frozenset[str]],
        on_submit: base.SubmitCallback,
    ) -> base.BatchResults:
        by_key = {request.key: request for request in requests}
        unsubmitted = set(by_key)
        jobs: list[str] = []
        for job_id, keys in resume.items():
            if keys & unsubmitted:
                print(f"Resuming batch job {job_id} from an earlier run.")
                jobs.append(job_id)
                unsubmitted -= keys

        if unsubmitted:
            job = await self._client.messages.batches.create(
                requests=[
                    {"custom_id": key, "params": self._params(by_key[key], batch=True)}
                    for key in sorted(unsubmitted)
                ]
            )
            on_submit(job.id, frozenset(unsubmitted))
            print(f"Submitted {len(unsubmitted)} requests as batch job {job.id}.")
            jobs.append(job.id)

        responses: dict[str, caches.ResponseData] = {}
        failures: dict[str, str] = {}
        for job_id in jobs:
            try:
                await self._wait_for(job_id)
                results = await self._client.messages.batches.results(job_id)
                async for item in results:
                    if item.custom_id not in by_key:
                        # A resumed job can hold requests this run no longer
                        # asks for; there is nowhere to put them.
                        continue
                    _collect(item, responses, failures)
            except anthropic.NotFoundError:
                # Batch results expire after 29 days.
                print(
                    f"Batch job {job_id} no longer exists; its requests will be retried."
                )

        for key in by_key.keys() - responses.keys() - failures.keys():
            failures[key] = "no result came back from the batch job"

        return base.BatchResults(
            responses=responses, failures=failures, completed_jobs=frozenset(jobs)
        )

    async def _wait_for(self, job_id: str) -> None:
        interval = self.poll_interval
        while True:
            job = await self._client.messages.batches.retrieve(job_id)
            if job.processing_status == "ended":
                return
            counts = job.request_counts
            print(
                f"Batch job {job_id}: {counts.processing} processing, "
                f"{counts.succeeded} succeeded, {counts.errored} errored."
            )
            await asyncio.sleep(interval)
            interval = min(interval * 2, self.max_poll_interval)

    def _params(self, request: base.Request, *, batch: bool) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system_prompt:
            system_block: dict[str, Any] = {
                "type": "text",
                "text": request.system_prompt,
            }
            if self.prompt_caching:
                # A batch can take longer than the default five-minute cache
                # lifetime to work through, so give it an hour.
                system_block["cache_control"] = (
                    {"type": "ephemeral", "ttl": "1h"}
                    if batch
                    else {"type": "ephemeral"}
                )
            params["system"] = [system_block]
        if self.effort is not None:
            params["output_config"] = {"effort": self.effort}
        return params


def _collect(
    item: anthropic.types.messages.MessageBatchIndividualResponse,
    responses: dict[str, caches.ResponseData],
    failures: dict[str, str],
) -> None:
    result = item.result
    match result.type:
        case "succeeded":
            try:
                responses[item.custom_id] = _response_data(result.message, batch=True)
            except base.IncompleteResponseError as e:
                failures[item.custom_id] = str(e)
        case "errored":
            failures[item.custom_id] = f"errored: {result.error.error.message}"
        case other:
            failures[item.custom_id] = other


def _response_data(
    message: anthropic.types.Message, *, batch: bool
) -> caches.ResponseData:
    if message.stop_reason in _INCOMPLETE_STOP_REASONS:
        raise base.IncompleteResponseError(
            f"The response stopped early (stop_reason={message.stop_reason!r})."
        )
    usage = message.usage
    return {
        "text": "".join(
            block.text for block in message.content if block.type == "text"
        ),
        "usage": {
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "details": {
                "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens,
                "batch": batch,
                "message_id": message.id,
            },
        },
    }
