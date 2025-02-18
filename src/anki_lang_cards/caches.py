import datetime
import enum
import functools
import hashlib
import json
import pathlib
from collections.abc import Mapping
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
    details: str | None


class ResponseData(TypedDict):
    text: str
    usage: UsageData


class CacheResult(TypedDict):
    input: PromptInput
    response: ResponseData


@attrs.frozen
class QueryCacher:
    cache_type: CacheTypes
    model_name: str
    model_service: str
    cache_loc: pathlib.Path = DEFAULT_CACHE_LOC

    def cache_response(self, response_text: str, filename: str | None = None) -> None:
        if not self.cache_loc.exists():
            self.cache_loc.mkdir()

        response_dir = self.response_dir
        if not response_dir.exists():
            response_dir.mkdir()

        if filename is None:
            filename = (
                f"response-{datetime.datetime.now().isoformat(timespec='seconds')}.csv"
            )

        cache_path = response_dir / filename

        with open(cache_path, "wt") as f:
            f.write(response_text)

    def load_latest_cache(self) -> str | None:
        response_dir = self.response_dir
        if not response_dir.exists() or not any(response_dir.iterdir()):
            raise RuntimeError("No cached responses found")

        # Get all files in the cache directory.
        files = list(response_dir.glob("*.csv"))

        if not files:
            return None

        # Sort files by last modified time, in descending order.
        latest_file = max(files, key=lambda f: f.stat().st_mtime)

        # Read and return the contents of the most recent file.
        with open(latest_file, "rt") as f:
            return f.read()

    def llm_cache_file_loc(self, prompt: str, system_prompt: str |  None) -> pathlib.Path:
        """Retrieve the location of the cache for a particular prompt.

        The filename is based on the hash of the prompt."""
        if system_prompt is not None:
            prompt = system_prompt + prompt

        prompt_hash: str = hashlib.md5(
            prompt.encode("utf-8"), usedforsecurity=False
        ).hexdigest()

        return (
            self.cache_loc
            / self.model_service
            / self.model_name
            / str(self.cache_type)
            / f"{prompt_hash}.json"
        )

    def load_from_cache(self, prompt: str, system_prompt: str | None = None) -> CacheResult | None:
        """Loads a cached response from the cache directory, or None.

        If the cache file does not exist, this returns `None`.
        """
        file_loc = self.llm_cache_file_loc(prompt, system_prompt)
        if not file_loc.exists():
            return None
        return json.loads(file_loc.read_text())

    def cache_result(self, prompt: str, response: Mapping[Any, Any], query_params: Mapping | None = None, system_prompt: str | None = None) -> None:
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
