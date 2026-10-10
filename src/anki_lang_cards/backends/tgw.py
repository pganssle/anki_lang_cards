"""Backend for models served by Text Generation WebUI.

It talks to TGW's OpenAI-style completions API. Model specs written
``tgw/<model>`` go here.
"""

import urllib.parse
from collections.abc import Mapping
from typing import Any, ClassVar, Self

import aiohttp
import backoff

from .. import _types as types
from .. import caches, config
from . import base


def _backoff_logger(details: Mapping[str, Any]) -> None:
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling the Text Generation WebUI API".format(**details)
    )


class TGWBackend:
    service: ClassVar[str] = "tgw"
    claims_prefix: ClassVar[bool] = True
    features: ClassVar[frozenset[base.Feature]] = frozenset()
    default_words_per_prompt: ClassVar[int] = 2

    def __init__(
        self,
        model_name: str,
        *,
        api_base_url: types.Url,
        max_connections: int = 1,
        request_timeout: int = 360,
        max_tokens: int = 8192,
    ):
        self.model_name = model_name
        self.max_concurrency: int | None = max_connections
        self.api_base_url = api_base_url
        self._params: Mapping[str, str | int | float] = {
            "max_tokens": max_tokens,
            "request_timeout": request_timeout,
        }

    @classmethod
    def from_config(
        cls,
        model_name: str,
        conf: config.Config,
        features: frozenset[base.Feature],
    ) -> Self:
        if conf.tgw is None:
            raise base.ConfigurationError(
                "A tgw model was selected, but there is no [tgw] config."
            )
        return cls(
            model_name,
            api_base_url=conf.tgw.api_base_url,
            max_connections=conf.tgw.max_connections,
            request_timeout=conf.tgw.request_timeout,
            max_tokens=conf.tgw.max_tokens,
        )

    async def __aenter__(self) -> Self:
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200))
        await self._load_model()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._session.__aexit__(exc_type, exc_val, exc_tb)

    def query_params(self) -> Mapping[str, object]:
        return self._params

    @backoff.on_exception(
        backoff.expo,
        aiohttp.client_exceptions.ClientResponseError,
        on_backoff=[_backoff_logger],
    )
    async def query(self, request: base.Request) -> caches.ResponseData:
        data = {"prompt": request.system_prompt + request.prompt, **self._params}
        async with self._session.post(
            self._endpoint("completions"), json=data
        ) as response:
            response.raise_for_status()
            resp = await response.json()

        choice = resp["choices"][0]
        if "text" not in choice:
            raise base.IncompleteResponseError(
                f"No text in the response; finish_reason={choice['finish_reason']}"
            )
        if choice["finish_reason"] != "stop":
            print(f"Query finished for reason {choice['finish_reason']}")
        usage = resp.get("usage", {})
        return {
            "text": choice["text"],
            "usage": {
                "input": usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
                "details": None,
            },
        }

    async def _load_model(self) -> None:
        """Make the server load this backend's model if it hasn't already."""
        async with self._session.get(self._endpoint("internal/model/info")) as resp:
            resp.raise_for_status()
            if (await resp.json())["model_name"] == self.model_name:
                return

        async with self._session.post(
            self._endpoint("internal/model/load"),
            json={"model_name": self.model_name},
        ) as resp:
            resp.raise_for_status()
            print(await resp.json())

    def _endpoint(self, path: str) -> str:
        return urllib.parse.urljoin(self.api_base_url, path)
