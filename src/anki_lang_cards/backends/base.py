"""The interface every model backend implements."""

import enum
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import ClassVar, Protocol, Self

import attrs

from .. import caches


class Feature(enum.StrEnum):
    """Optional capabilities a backend may support."""

    BATCH = "batch"
    """Submitting many requests as one discounted, asynchronous job."""

    PROMPT_CACHING = "prompt_caching"
    """Reusing a processed system prompt across requests at a discount."""


@attrs.frozen
class Request:
    """One prompt to send to a model."""

    prompt: str
    system_prompt: str = ""

    @property
    def key(self) -> str:
        """A stable identifier, shared with the response cache's filenames."""
        return caches.prompt_hash(self.prompt, self.system_prompt)


class IncompleteResponseError(ValueError):
    """The model stopped before finishing, e.g. on its output token limit.

    These are retried like unparseable output, but never written to the
    response cache.
    """


class Backend(Protocol):
    """A model endpoint that answers one :class:`Request` at a time."""

    service: ClassVar[str]
    """The name used to select this backend and to key its response cache."""

    features: ClassVar[frozenset[Feature]]

    default_words_per_prompt: ClassVar[int]
    """How many words to put in one sentence-generation prompt."""

    model_name: str
    """The model as the backend's API names it."""

    max_concurrency: int | None
    """A cap on simultaneous requests the endpoint can take, if it has one."""

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
