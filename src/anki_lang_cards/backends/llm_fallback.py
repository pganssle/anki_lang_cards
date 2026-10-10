"""Any model the ``llm`` library can reach through its installed plugins."""

from collections.abc import Mapping
from typing import ClassVar, Self

import llm

from .. import caches
from . import base


class LLMBackend:
    service: ClassVar[str] = "llm"
    features: ClassVar[frozenset[base.Feature]] = frozenset()
    default_words_per_prompt: ClassVar[int] = 15

    def __init__(self, model_name: str, *, keys: Mapping[str, str] | None = None):
        """
        :param keys: API keys by the name ``llm`` files them under (e.g.
            ``"anthropic"``). A model whose key is not here falls back on
            ``llm``'s own key store and environment variables.
        """
        self.model_name = model_name
        self.max_concurrency: int | None = None
        self._model = llm.get_async_model(model_name)
        needs_key = getattr(self._model, "needs_key", None)
        if keys and needs_key is not None and needs_key in keys:
            self._model.key = keys[needs_key]

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def query_params(self) -> Mapping[str, object]:
        return {}

    async def query(self, request: base.Request) -> caches.ResponseData:
        response = await self._model.prompt(
            request.prompt, system=request.system_prompt or None
        )
        text = await response.text()
        usage = await response.usage()
        return {
            "text": text,
            "usage": {
                "input": usage.input,
                "output": usage.output,
                "details": usage.details,
            },
        }
