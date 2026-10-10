"""Config handler"""

import pathlib
import tomllib
from collections.abc import Mapping, Sequence
from typing import Any, Final, Self

import attrs

from . import _types as types


@attrs.frozen
class AnthropicConfig:
    api_key: str | None = None
    """If unset, the SDK looks for credentials itself (ANTHROPIC_API_KEY, ...)."""

    default_model: str | None = None
    """Deprecated in favor of ``[models] default_model``, which wins if set."""

    effort: str | None = None
    """The ``output_config.effort`` level; unset leaves the model's default."""

    max_tokens: int | None = None
    base_url: str | None = None

    @classmethod
    def from_values(cls, vals: Mapping[str, Any]) -> Self:
        return cls(**vals)


@attrs.frozen
class TGWConfig:
    api_base_url: types.Url
    max_connections: int = 1
    request_timeout: int = 360
    max_tokens: int = 8192

    @classmethod
    def from_values(cls, vals: Mapping[str, Any]) -> Self:
        return cls(**vals)


DEFAULT_DIALECT: Final[str] = "international Spanish"


@attrs.frozen
class ModelsConfig:
    """Which model to use, and which optional backend features to use with it.

    A feature left unset is used wherever the backend supports it; one set
    to true is treated as a requirement, and the user is asked before
    running without it.
    """

    default_model: str | None = None
    batch: bool | None = None
    prompt_caching: bool | None = None
    structured_output: bool | None = None

    @classmethod
    def from_values(cls, vals: Mapping[str, Any]) -> Self:
        return cls(**vals)


@attrs.frozen
class SentencesConfig:
    """Config options for LLM sentence generation."""

    dialect: str = DEFAULT_DIALECT

    @classmethod
    def from_values(cls, vals: Mapping[str, Any]) -> Self:
        return cls(**vals)


@attrs.frozen
class AnnotationsConfig:
    """Default config options for annotations."""

    dictionaries: Sequence[pathlib.Path] = ()

    @classmethod
    def from_values(cls, vals: Mapping[str, Any]) -> Self:
        if "dictionaries" in vals:
            new_vals = {**vals}
            dictionaries = tuple(map(pathlib.Path, vals["dictionaries"]))
            new_vals["dictionaries"] = dictionaries
            vals = new_vals
        return cls(**vals)


@attrs.frozen
class Config:
    annotations: AnnotationsConfig
    tgw: TGWConfig | None = None
    anthropic: AnthropicConfig | None = None
    sentences: SentencesConfig = SentencesConfig()
    models: ModelsConfig = ModelsConfig()

    @classmethod
    def from_file(cls, path: pathlib.Path) -> Self:
        with open(path, "rb") as f:
            config = tomllib.load(f)

        if (tgw_config_values := config.get("tgw", None)) is not None:
            tgw_config = TGWConfig.from_values(tgw_config_values)
        else:
            tgw_config = None

        if (anthropic_config_values := config.get("anthropic", None)) is not None:
            anthropic_config = AnthropicConfig.from_values(anthropic_config_values)
        else:
            anthropic_config = None

        if (annotations_values := config.get("annotations", None)) is not None:
            annotations_config = AnnotationsConfig.from_values(annotations_values)
        else:
            annotations_config = AnnotationsConfig()

        if (sentences_values := config.get("sentences", None)) is not None:
            sentences_config = SentencesConfig.from_values(sentences_values)
        else:
            sentences_config = SentencesConfig()

        if (models_values := config.get("models", None)) is not None:
            models_config = ModelsConfig.from_values(models_values)
        else:
            models_config = ModelsConfig()

        return cls(
            models=models_config,
            tgw=tgw_config,
            anthropic=anthropic_config,
            annotations=annotations_config,
            sentences=sentences_config,
        )
