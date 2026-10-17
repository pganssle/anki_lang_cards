"""Backend for Claude models through the Anthropic SDK.

It supports batching and prompt caching. Model specs written
``anthropic/<model>`` go here.
"""

import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final, Self

import anthropic

from .. import caches, config
from . import base

# Room for adaptive thinking as well as the response, since thinking counts
# against max_tokens.
DEFAULT_MAX_TOKENS: Final[int] = 16_000

_INCOMPLETE_STOP_REASONS: Final = frozenset({"max_tokens", "refusal"})

# The API rejects max_tokens=0 together with structured outputs, so a warm-up
# costs one output token.
WARM_UP_MAX_TOKENS: Final[int] = 1


class AnthropicBackend:
    service: ClassVar[str] = "anthropic"
    claims_prefix: ClassVar[bool] = True
    features: ClassVar[frozenset[base.Feature]] = frozenset(
        {
            base.Feature.BATCH,
            base.Feature.PROMPT_CACHING,
            base.Feature.STRUCTURED_OUTPUT,
        }
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
        structured_output: bool = True,
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
        self.structured_output = structured_output
        self.poll_interval = poll_interval
        self.max_poll_interval = max_poll_interval
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key, base_url=base_url, http_client=http_client
        )

    @classmethod
    def supported_features(cls, model_name: str) -> frozenset[base.Feature]:
        return cls.features

    @classmethod
    def from_config(
        cls,
        model_name: str,
        conf: config.Config,
        features: frozenset[base.Feature],
    ) -> Self:
        anthropic_conf = conf.anthropic or config.AnthropicConfig()
        return cls(
            model_name,
            api_key=anthropic_conf.api_key,
            base_url=anthropic_conf.base_url,
            max_tokens=anthropic_conf.max_tokens or DEFAULT_MAX_TOKENS,
            effort=anthropic_conf.effort,
            prompt_caching=base.Feature.PROMPT_CACHING in features,
            structured_output=base.Feature.STRUCTURED_OUTPUT in features,
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
            "structured_output": self.structured_output,
        }

    async def query(self, request: base.Request) -> caches.ResponseData:
        message = await self._client.messages.create(
            **self._params(request, batch=False)
        )
        _require_complete(message)
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

        warm_ups: Mapping[base.Request, caches.ResponseData] = {}
        if self.prompt_caching:
            warm_ups = await self._warm_cache(
                [by_key[key] for key in sorted(unsubmitted)]
            )

        responses: dict[str, caches.ResponseData] = {}
        failures: dict[str, str] = {}
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
            responses=responses,
            failures=failures,
            completed_jobs=frozenset(jobs),
            warm_ups=warm_ups,
        )

    async def _warm_cache(
        self, requests: Sequence[base.Request]
    ) -> Mapping[base.Request, caches.ResponseData]:
        """Write each shared system prompt to the prompt cache ahead of the batch.

        Requests in a batch are processed concurrently, so without this most
        of them miss the cache and pay to write it; at the hour-long cache
        lifetime a write costs twice as much as uncached input. A throwaway
        request with the same system prompt and settings but a one-token
        output limit writes the entry before the batch starts, for the cost
        of the write alone, so that the batched requests can read it. The
        API only promises best-effort cache hits within a batch, though, and
        in recordings the batch has missed the entry as often as not. A
        system prompt only one request uses is left to the batch.

        Nothing is journaled against a crash between here and submitting the
        batch: the next run's warm-up finds the entry this one wrote and
        pays a cache read for it, a tenth of the price of uncached input.

        :return: The warm-up requests as sent, with their responses.
        """
        by_system_prompt: dict[str, list[base.Request]] = {}
        for request in requests:
            if request.system_prompt:
                by_system_prompt.setdefault(request.system_prompt, []).append(request)

        warm_ups: dict[base.Request, caches.ResponseData] = {}
        for sharing in by_system_prompt.values():
            if len(sharing) < 2:
                continue
            print(
                "Sending a warm-up request ahead of the batch to fill the prompt cache."
            )
            warm_up = _warm_up_request(sharing)
            # The batch parameters, so that it writes the same hour-long
            # cache entry the batch will read.
            params = self._params(warm_up, batch=True)
            params["max_tokens"] = WARM_UP_MAX_TOKENS
            message = await self._client.messages.create(**params)
            warm_ups[warm_up] = _response_data(message, batch=False, warm_up=True)
        return warm_ups

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
        output_config: dict[str, Any] = {}
        if self.effort is not None:
            output_config["effort"] = self.effort
        if self.structured_output and request.json_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": request.json_schema,
            }
        if output_config:
            params["output_config"] = output_config
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
                _require_complete(result.message)
            except base.IncompleteResponseError as e:
                failures[item.custom_id] = str(e)
            else:
                responses[item.custom_id] = _response_data(result.message, batch=True)
        case "errored":
            failures[item.custom_id] = f"errored: {result.error.error.message}"
        case other:
            failures[item.custom_id] = other


def _warm_up_request(sharing: Sequence[base.Request]) -> base.Request:
    """The throwaway request that warms the cache for ``sharing``.

    The prompt after the cached system prompt can be anything; it names the
    batch it warms (by a digest of the requests' keys) so that each batch's
    warm-up is recorded separately in the response cache, where the usage
    report finds it.
    """
    digest = hashlib.md5(
        "".join(sorted(request.key for request in sharing)).encode("ascii"),
        usedforsecurity=False,
    ).hexdigest()
    first = sharing[0]
    return base.Request(
        f"Prompt cache warm-up for batch {digest}.",
        system_prompt=first.system_prompt,
        json_schema=first.json_schema,
    )


def _require_complete(message: anthropic.types.Message) -> None:
    if message.stop_reason in _INCOMPLETE_STOP_REASONS:
        raise base.IncompleteResponseError(
            f"The response stopped early (stop_reason={message.stop_reason!r})."
        )


def _response_data(
    message: anthropic.types.Message, *, batch: bool, warm_up: bool = False
) -> caches.ResponseData:
    usage = message.usage
    details: dict[str, Any] = {
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "batch": batch,
        "message_id": message.id,
    }
    if warm_up:
        details["warm_up"] = True
    if usage.cache_creation is not None:
        # Cache writes cost more for the longer lifetime, so the split is
        # needed to price a run exactly.
        details["cache_creation_5m_input_tokens"] = (
            usage.cache_creation.ephemeral_5m_input_tokens
        )
        details["cache_creation_1h_input_tokens"] = (
            usage.cache_creation.ephemeral_1h_input_tokens
        )
    return {
        "text": "".join(
            block.text for block in message.content if block.type == "text"
        ),
        "usage": {
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "details": details,
        },
    }
