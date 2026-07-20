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
