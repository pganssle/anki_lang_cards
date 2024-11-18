"""Config handler"""

import pathlib
import tomllib
from collections.abc import Mapping
from typing import Any, Self

import attrs

from . import _types as types


@attrs.frozen
class AnthropicConfig:
    api_key: str

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


@attrs.frozen
class Config:
    tgw: TGWConfig | None = None
    anthropic: AnthropicConfig | None = None

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
        return cls(tgw=tgw_config, anthropic=anthropic_config)
