"""Library for dictionary headword stemming and dictionary lookup."""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

import attrs
import hunspell
import pystardict
from lxml import html


@attrs.frozen
class Definition:
    meaning: str
    pos: str | None


# Custom types
Definitions: TypeAlias = Sequence[Definition]


@attrs.frozen
class DictionaryEntry:
    headword: str
    definitions: Definitions


def _parse_wiktionary_entry(content: str) -> Sequence[Definition]:
    tree = html.fromstring(f"<div>{content}</div>")
    definitions: list[Definition] = []
    current_pos = None

    for element in tree.xpath("./*"):
        if element.tag == "i":
            current_pos = element.text
        elif element.tag == "ol" and current_pos:
            for li in element.xpath(".//li"):
                definitions.append(Definition(pos=current_pos, meaning=li.text.strip()))

    return tuple(definitions)


def _parse_simple_entry(content: str) -> Sequence[Definition]:
    if not content:
        return ()

    content = re.sub(r"<k>.*?</k>\n?", "", content)
    meanings = [s.strip() for s in content.split(",")]
    pos = "v" if any(m.startswith("to ") for m in meanings) else None

    return tuple(
        Definition(pos=pos, meaning=meaning) for meaning in meanings if meaning
    )


class DictionaryEngine:
    def __init__(
        self, hunspell_dict_path: Path, stardict_paths: Sequence[Path]
    ) -> None:
        self._speller = hunspell.HunSpell(
            str(hunspell_dict_path.with_suffix(".dic")),
            str(hunspell_dict_path.with_suffix(".aff")),
        )
        self._dictionaries = [
            pystardict.Dictionary(str(path)) for path in stardict_paths
        ]

        self._lookup_cache: dict[str, Sequence[Definition]] = {}

    def _get_stems(self, word: str) -> Sequence[str]:
        """Get all possible stems for a word."""
        stems = [x.decode("utf-8") for x in self._speller.stem(word)]
        if word not in stems:
            stems.append(word)
        # Add original word as potential stem
        return tuple(stems)

    def _get_raw_definitions(self, headword: str) -> Sequence[Definition]:
        """Get all definitions for a headword from all dictionaries."""
        if headword in self._lookup_cache:
            return self._lookup_cache[headword]

        all_definitions: list[Definition] = []
        for dictionary in self._dictionaries:
            if headword in dictionary:
                content = dictionary[headword]
                if "<i>" in content:
                    definitions = _parse_wiktionary_entry(content)
                else:
                    definitions = _parse_simple_entry(content)
                all_definitions.extend(definitions)

                # We'll take the definition from the first dictionary that has
                # one for now.
                break

        definitions = tuple(all_definitions)

        self._lookup_cache[headword] = definitions
        return self._lookup_cache[headword]

    def lookup(self, word: str) -> Sequence[DictionaryEntry]:
        """
        Look up all possible dictionary entries for a word.

        Args:
            word: Spanish word to look up

        Returns:
            Sequence of DictionaryEntry objects, one per possible headword
        """
        stems = self._get_stems(word.lower())
        entries = []

        for stem in stems:
            definitions = self._get_raw_definitions(stem)
            if definitions:
                entries.append(DictionaryEntry(headword=stem, definitions=definitions))

        return tuple(entries)


if __name__ == "__main__":
    stardict_paths = [
        Path(
            "input/dicts/Spanish-English Wiktionary dictionary stardict/Spanish-English Wiktionary dictionary"
        ),
        Path("input/dicts/stardict-spanish-english-2.4.2/spanish-english"),
    ]

    hunspell_dict_path = Path("/usr/share/hunspell/es_ES.dic")
    engine = DictionaryEngine(
        hunspell_dict_path=hunspell_dict_path, stardict_paths=stardict_paths
    )

    dic_entry = engine.lookup("cariño")

    print(dic_entry)
