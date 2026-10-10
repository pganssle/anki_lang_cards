"""The interface model backends implement, and the requests they answer."""

import enum
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Protocol, Self

import attrs

from .. import caches, config


class Feature(enum.StrEnum):
    """Optional capabilities a backend may support."""

    BATCH = "batch"
    """Submitting many requests as one discounted, asynchronous job."""

    PROMPT_CACHING = "prompt_caching"
    """Reusing a processed system prompt across requests at a discount."""

    STRUCTURED_OUTPUT = "structured_output"
    """Constraining the response to a request's JSON schema."""


@attrs.frozen
class Request:
    """One prompt to send to a model."""

    prompt: str
    system_prompt: str = ""
    json_schema: Mapping[str, Any] | None = attrs.field(
        default=None, eq=False, kw_only=True
    )
    """The JSON schema the response should follow, for backends that can
    enforce one. The prompt must still ask for the same format, since not
    every backend can. It is not part of a request's identity: the schema
    follows from the kind of prompt."""

    @property
    def key(self) -> str:
        """A stable identifier, shared with the response cache's filenames."""
        return caches.prompt_hash(self.prompt, self.system_prompt)


class ConfigurationError(ValueError):
    """The config file lacks something a backend needs."""


class IncompleteResponseError(ValueError):
    """The model stopped before finishing, e.g. on its output token limit.

    These are retried like unparseable output, but never written to the
    response cache.
    """


class Backend(Protocol):
    """A model endpoint that answers one :class:`Request` at a time."""

    service: ClassVar[str]
    """The name used to select this backend and to key its response cache."""

    claims_prefix: ClassVar[bool]
    """Whether model specs written ``<service>/<model>`` go to this backend.

    A backend that routes to many providers itself, like the ``llm``
    fallback, leaves this off: its model names already contain slashes.
    """

    @classmethod
    def supported_features(cls, model_name: str) -> frozenset[Feature]:
        """The features this backend supports for a model."""
        ...

    features: frozenset[Feature]
    """The features this instance supports."""

    default_words_per_prompt: ClassVar[int]
    """How many words to put in one sentence-generation prompt."""

    model_name: str
    """The model as the backend's API names it."""

    max_concurrency: int | None
    """A cap on simultaneous requests the endpoint can take, if it has one."""

    @classmethod
    def from_config(
        cls, model_name: str, conf: config.Config, features: frozenset[Feature]
    ) -> Self:
        """Build the backend for a model from the config file's settings.

        :param features: The optional features to use; a subset of
            :meth:`supported_features` for the model.
        :raises ConfigurationError: If the config lacks something required.
        """
        ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None: ...

    async def query(self, request: Request) -> caches.ResponseData:
        """Send one request and wait for the response.

        :raises IncompleteResponseError: If the model's response was cut off.
        """
        ...

    def query_params(self) -> Mapping[str, object]:
        """The request parameters to record alongside cached responses."""
        ...


@attrs.frozen
class BatchResults:
    """What came back from one or more batch jobs."""

    responses: Mapping[str, caches.ResponseData]
    """Responses keyed by :attr:`Request.key`."""

    failures: Mapping[str, str]
    """Why each failed request failed, keyed by :attr:`Request.key`."""

    completed_jobs: frozenset[str]
    """The provider's IDs for the jobs whose results are all collected."""


class SubmitCallback(Protocol):
    def __call__(self, job_id: str, request_keys: frozenset[str]) -> None: ...


class BatchBackend(Backend, Protocol):
    """A backend that also supports :attr:`Feature.BATCH`."""

    async def query_batch(
        self,
        requests: Sequence[Request],
        *,
        resume: Mapping[str, frozenset[str]],
        on_submit: SubmitCallback,
    ) -> BatchResults:
        """Run requests as batch jobs and wait for all of them to finish.

        :param resume: Jobs submitted by an earlier, interrupted run, mapping
            each job ID to the request keys it holds. Requests already in one
            of these are collected from it rather than submitted again.
        :param on_submit: Called with each new job's ID and request keys as
            soon as it is submitted, so it can be resumed if this run dies.
        """
        ...
