"""Backend for any model the ``llm`` library can reach through its plugins.

Model specs that no other backend claims come here, so ``llm`` model names
(``gpt-5``, ``anthropic/claude-sonnet-5``, ...) work unchanged.
"""

from collections.abc import Mapping
from typing import ClassVar, Self

import llm

from .. import caches, config
from . import base


class LLMBackend:
    service: ClassVar[str] = "llm"
    claims_prefix: ClassVar[bool] = False
    default_words_per_prompt: ClassVar[int] = 15

    def __init__(
        self,
        model_name: str,
        *,
        keys: Mapping[str, str] | None = None,
        structured_output: bool = True,
    ):
        """
        :param keys: API keys by the name ``llm`` files them under (e.g.
            ``"anthropic"``). A model whose key is not here falls back on
            ``llm``'s own key store and environment variables.
        """
        self.model_name = model_name
        self.max_concurrency: int | None = None
        self._model = llm.get_async_model(model_name)
        self.features = _model_features(self._model)
        self.structured_output = (
            structured_output and base.Feature.STRUCTURED_OUTPUT in self.features
        )
        needs_key = getattr(self._model, "needs_key", None)
        if keys and needs_key is not None and needs_key in keys:
            self._model.key = keys[needs_key]

    @classmethod
    def supported_features(cls, model_name: str) -> frozenset[base.Feature]:
        return _model_features(llm.get_async_model(model_name))

    @classmethod
    def from_config(
        cls,
        model_name: str,
        conf: config.Config,
        features: frozenset[base.Feature],
    ) -> Self:
        keys = {}
        if conf.anthropic is not None and conf.anthropic.api_key is not None:
            keys["anthropic"] = conf.anthropic.api_key
        return cls(
            model_name,
            keys=keys,
            structured_output=base.Feature.STRUCTURED_OUTPUT in features,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def query_params(self) -> Mapping[str, object]:
        return {"structured_output": self.structured_output}

    async def query(self, request: base.Request) -> caches.ResponseData:
        schema = request.json_schema if self.structured_output else None
        try:
            response = await self._model.prompt(
                request.prompt,
                system=request.system_prompt or None,
                schema=dict(schema) if schema is not None else None,
            )
            text = await response.text()
            usage = await response.usage()
        except llm.ModelError as e:
            raise base.RequestError(f"{type(e).__name__}: {e}") from e
        except Exception as e:
            # Plugins mostly let their provider's SDK errors through. Those
            # carry an HTTP status in one of two spellings (the Anthropic and
            # OpenAI SDKs' status_code, aiohttp's status); anything without
            # one is not known to be about this request, so it propagates.
            status = _http_status(e)
            if status is None:
                raise
            raise base.RequestError(f"{type(e).__name__}: {e}", status=status) from e
        return {
            "text": text,
            "usage": {
                "input": usage.input,
                "output": usage.output,
                "details": usage.details,
            },
        }


def _http_status(error: Exception) -> int | None:
    for attribute in ("status_code", "status"):
        status = getattr(error, attribute, None)
        if isinstance(status, int):
            return status
    return None


def _model_features(model: llm.AsyncModel) -> frozenset[base.Feature]:
    if model.supports_schema:
        return frozenset({base.Feature.STRUCTURED_OUTPUT})
    return frozenset()
