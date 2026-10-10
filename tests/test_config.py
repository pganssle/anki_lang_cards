"""Tests for config file parsing."""

import pathlib

import pytest

from anki_lang_cards import config

MINIMAL_CONFIG = """\
[anthropic]
api_key="fake-key"
"""

CONFIG_WITH_DIALECT = """\
[anthropic]
api_key="fake-key"

[sentences]
dialect = "peninsular Spanish (the variety spoken in Spain)"
"""


@pytest.mark.parametrize(
    ("config_text", "expected_dialect"),
    [
        (MINIMAL_CONFIG, "international Spanish"),
        (
            CONFIG_WITH_DIALECT,
            "peninsular Spanish (the variety spoken in Spain)",
        ),
    ],
)
def test_sentences_dialect(
    tmp_path: pathlib.Path, config_text: str, expected_dialect: str
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_text, encoding="utf-8")

    conf = config.Config.from_file(config_path)

    assert conf.sentences.dialect == expected_dialect


@pytest.mark.parametrize(
    ("config_text", "expected_models", "expected_anthropic"),
    [
        (
            MINIMAL_CONFIG,
            config.ModelsConfig(),
            config.AnthropicConfig(api_key="fake-key"),
        ),
        (
            """\
[models]
default_model = "anthropic/claude-sonnet-5"
batch = false
prompt_caching = true

[anthropic]
effort = "medium"
max_tokens = 32000
""",
            config.ModelsConfig(
                default_model="anthropic/claude-sonnet-5",
                batch=False,
                prompt_caching=True,
            ),
            config.AnthropicConfig(effort="medium", max_tokens=32000),
        ),
        ("[models]\n", config.ModelsConfig(), None),
    ],
)
def test_models_config(
    tmp_path: pathlib.Path,
    config_text: str,
    expected_models: config.ModelsConfig,
    expected_anthropic: config.AnthropicConfig | None,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_text, encoding="utf-8")

    conf = config.Config.from_file(config_path)

    assert conf.models == expected_models
    assert conf.anthropic == expected_anthropic
