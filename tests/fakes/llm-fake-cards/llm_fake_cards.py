"""A fake ``llm`` model plugin for anki_lang_cards end-to-end tests.

The model recognizes the two prompt shapes used by the card-generation
pipeline (the book-annotation RAG prompt and the multi-sentence prompt),
parses the JSON out of the INPUT section, and fabricates deterministic
responses. The generated sentences deliberately include characters that have
historically been mangled somewhere between the LLM and Anki: straight and
Spanish quotation marks, inverted punctuation, em-dashes, ampersands, and
non-ASCII letters.

Trigger words embedded in a highlight/term switch on specific misbehaviors
so tests can exercise the error-handling paths:

- ``fence``: the response is wrapped in a ```` ```json ```` fence.
- ``badjson``: the first response to a given prompt is not JSON at all;
  subsequent identical prompts get a valid response.
- ``badsource``: examples cite a ``source_id`` that does not exist.
- ``renamed``: the term is reported without its ``/a`` suffix, so it cannot
  be matched to the input word.
- ``wordkey``: the response uses a ``"word"`` key instead of ``"term"``
  (a mistake the real prompt's own examples used to contain).
- ``apierror``: the model raises ``llm.ModelError`` every time, as a
  plugin does when its provider rejects the request.
"""

import hashlib
import json
import re

import llm
import pydantic

MODEL_ID = "fake-cards"

_INPUT_RE = re.compile(r"INPUT:\n-+\n(?P<json>.*)\n+OUTPUT:\n-+", re.DOTALL)

# Prompts that have already been served an invalid response once.
_invalid_served: set[str] = set()


class _FakeOptions(pydantic.BaseModel):
    max_tokens: int | None = None


def _parse_input(prompt_text: str):
    match = _INPUT_RE.search(prompt_text)
    if match is None:
        raise ValueError(f"Fake model got an unrecognized prompt:\n{prompt_text}")
    raw = match.group("json")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # The multi-sentence prompt sends a comma-joined sequence of objects
        # without the enclosing list brackets; tolerate it like a real LLM.
        return json.loads(f"[{raw}]")


def _context_snippet(highlight: str, context: str) -> str:
    """A sentence extracted from the real context around the highlight."""
    idx = context.find(highlight)
    if idx == -1:
        return f"No encontré «{highlight}» en el texto."
    start = context.rfind(" ", 0, max(0, idx - 60)) + 1
    end = context.find(" ", idx + len(highlight) + 60)
    if end == -1:
        end = len(context)
    return context[start:end].strip()


def _annotation_entry(highlight, source_id: int) -> dict:
    if isinstance(highlight, dict):
        text = highlight["highlight"]
        note = highlight.get("note")
    else:
        text = highlight
        note = None

    headword = text if note is None else f"{text} ({note})"
    if "badsource" in text:
        source_id = 999999

    return {
        "highlight": text,
        "headword": headword,
        "part_of_speech": "nm",
        "meaning": f'a fake "meaning" of {text}',
        "article": "el",
        "examples": [
            {
                "spanish": f"—¿Has visto «{text}»? —preguntó María—. ¡Qué palabra!",
                "english": f'"Have you seen \\"{text}\\"?" asked María. "What a word!"',
                "source_id": source_id,
            },
            {
                "spanish": f"El niño escribió {text} & más palabras en su cuaderno.",
                "english": f"The boy wrote {text} & more words in his notebook.",
                "source_id": source_id,
            },
        ],
    }


def _respond_annotations(entries) -> list[dict]:
    out = []
    for entry in entries:
        context = entry["context"]
        for highlight in entry["highlights"]:
            item = _annotation_entry(highlight, entry["source_id"])
            item["examples"].insert(
                0,
                {
                    "spanish": _context_snippet(item["highlight"], context),
                    "english": f"A snippet from the book about {item['highlight']}.",
                    "source_id": item["examples"][0]["source_id"],
                },
            )
            out.append(item)
    return out


def _respond_sentences(entries) -> list[dict]:
    out = []
    for entry in entries:
        term = entry["term"]
        reported_term = term.split("/")[0] if "renamed" in term else term
        item = {
            "term": reported_term,
            "meaning": entry["meaning"],
            "examples": [
                {
                    "spanish": f'La palabra "{term}" apareció en el libro.',
                    "english": f'The word "{term}" appeared in the book.',
                },
                {
                    "spanish": f"Los expertos dicen: «{term}» es fundamental.",
                    "english": f"The experts say: '{term}' is fundamental.",
                },
                {
                    "spanish": f"¡{term}! ¿Quién lo diría?",
                    "english": f"{term}! Who would have thought?",
                },
                {
                    "spanish": f"Ayer aprendí {term} — una palabra muy útil.",
                    "english": f"Yesterday I learned {term} — a very useful word.",
                },
            ],
        }
        if "wordkey" in term:
            item["word"] = item.pop("term")
        out.append(item)
    return out


def _generate_response(prompt_text: str) -> str:
    entries = _parse_input(prompt_text)

    flat_terms: list[str] = []
    for entry in entries:
        if "highlights" in entry:
            for highlight in entry["highlights"]:
                if isinstance(highlight, dict):
                    flat_terms.append(highlight["highlight"])
                else:
                    flat_terms.append(highlight)
        else:
            flat_terms.append(entry["term"])

    if any("apierror" in term for term in flat_terms):
        raise llm.ModelError("the fake provider rejected this request")

    if any("badjson" in term for term in flat_terms):
        key = hashlib.md5(
            prompt_text.encode("utf-8"), usedforsecurity=False
        ).hexdigest()
        if key not in _invalid_served:
            _invalid_served.add(key)
            return "I'm sorry, I can't produce JSON today."

    if entries and "highlights" in entries[0]:
        response = _respond_annotations(entries)
    else:
        response = _respond_sentences(entries)

    text = json.dumps({"words": response}, ensure_ascii=False, indent=2)
    if any("fence" in term for term in flat_terms):
        text = f"```json\n{text}\n```"
    return text


class FakeCards(llm.Model):
    model_id = MODEL_ID
    can_stream = False

    class Options(_FakeOptions): ...

    def execute(self, prompt, stream, response, conversation):
        text = _generate_response(prompt.prompt)
        response.set_usage(input=len(prompt.prompt) // 4, output=len(text) // 4)
        yield text


class AsyncFakeCards(llm.AsyncModel):
    model_id = MODEL_ID
    can_stream = False

    class Options(_FakeOptions): ...

    async def execute(self, prompt, stream, response, conversation):
        text = _generate_response(prompt.prompt)
        response.set_usage(input=len(prompt.prompt) // 4, output=len(text) // 4)
        yield text


@llm.hookimpl
def register_models(register, model_aliases):
    register(FakeCards(), AsyncFakeCards())
