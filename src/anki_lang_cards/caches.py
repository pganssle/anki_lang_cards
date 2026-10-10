import datetime
import enum
import functools
import hashlib
import json
import operator
import pathlib
import typing
from collections.abc import Callable, Mapping
from typing import Any, Final, TypedDict

import attrs

DEFAULT_CACHE_LOC: Final[pathlib.Path] = pathlib.Path(".cache")


class CacheTypes(enum.StrEnum):
    RESPONSES = "responses"
    ANNOTATIONS = "annotations"
    SENTENCES = "sentences"


class PromptInput(TypedDict):
    model_name: str
    model_service: str
    prompt: str
    system_prompt: str | None
    query_params: Mapping
    timestamp: str


# TODO: Migrate these to attrs classes
class UsageData(TypedDict):
    input: int
    output: int
    details: Mapping[str, Any] | None


class ResponseData(TypedDict):
    text: str
    usage: UsageData


class CacheResult(TypedDict):
    input: PromptInput
    response: ResponseData


def prompt_hash(prompt: str, system_prompt: str | None) -> str:
    if system_prompt is not None:
        prompt = system_prompt + prompt
    return hashlib.md5(prompt.encode("utf-8"), usedforsecurity=False).hexdigest()


def datetime_filename(
    prompt: str, system_propmt: str | None, *, prefix: str = "response-"
) -> str:
    timestamp: str = (
        datetime.datetime.now().isoformat(timespec="seconds").replace(":", "")
    )
    return f"{prefix}{timestamp}"


@attrs.frozen
class QueryCacher:
    cache_type: CacheTypes
    model_name: str
    model_service: str
    cache_loc: pathlib.Path = DEFAULT_CACHE_LOC
    by_service: bool = True
    by_model: bool = True
    filename_generator: Callable[[str, str | None], str] = prompt_hash

    @functools.cached_property
    def cache_dir(self) -> pathlib.Path:
        components: list[str | pathlib.Path] = [self.cache_loc]
        components.append(str(self.cache_type))

        if self.by_service:
            components.append(self.model_service)
        if self.by_model:
            components.append(self.model_name)

        return typing.cast(pathlib.Path, functools.reduce(operator.truediv, components))

    def llm_cache_file_loc(
        self, prompt: str, system_prompt: str | None
    ) -> pathlib.Path:
        """Retrieve the location of the cache for a particular prompt.

        The filename is based on the hash of the prompt."""

        filename_base = self.filename_generator(prompt, system_prompt)
        return self.cache_dir / f"{filename_base}.json"

    def load_from_cache_file(self, cache_file_loc: pathlib.Path) -> CacheResult:
        """Loads the contents of a cache file.

        The cache file must exist to use this function!

        Arguments:
            cache_file_loc: The file on disk representing a JSON-encoded
                CacheResult object.

        Returns:
            The CacheResult object decoded from the cache file.
        """
        return typing.cast(CacheResult, json.loads(cache_file_loc.read_text()))

    def load_from_cache(
        self, prompt: str, system_prompt: str | None = None
    ) -> CacheResult | None:
        """Loads a cached response from the cache directory, or None.

        If the cache file does not exist, this returns `None`.
        """
        file_loc = self.llm_cache_file_loc(prompt, system_prompt)
        if not file_loc.exists():
            return None
        return self.load_from_cache_file(file_loc)

    def load_latest_cache(self) -> CacheResult | None:
        """Loads the most recently cached response (sorted by mtime)

        Returns:
            The response text if any responses have been cached, otherwise
            :py:obj:`None`.
        """
        if not self.cache_dir.exists() or not any(self.cache_dir.iterdir()):
            # No cached responses found
            return None

        # Get all files in the cache directory.
        files = list(self.cache_dir.glob("*.json"))

        if not files:
            return None

        # Sort files by last modified time, in descending order.
        latest_file = max(files, key=lambda f: f.stat().st_mtime)

        # Read and return the contents of the most recent file.
        return self.load_from_cache_file(latest_file)

    def cache_result(
        self,
        prompt: str,
        response: ResponseData,
        query_params: Mapping | None = None,
        system_prompt: str | None = None,
    ) -> pathlib.Path:
        """Add result into the cache."""
        if query_params is None:
            query_params = {}

        cache_data: CacheResult = {
            "input": {
                "model_name": self.model_name,
                "model_service": self.model_service,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "query_params": query_params,
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "response": response,
        }

        cache_file = self.llm_cache_file_loc(prompt, system_prompt)
        cache_file.parent.mkdir(exist_ok=True, parents=True)
        cache_file.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2))
        return cache_file
