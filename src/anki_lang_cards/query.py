import asyncio
import json
import random
import urllib.parse
from collections.abc import AsyncIterable, Mapping, Sequence
from typing import Any, ClassVar, Final, Protocol, Self

import aiohttp

from . import _types as types


class PromptHandler(Protocol):
    prompt_template: str

    def make_prompt(self, word: types.Word) -> str:
        ...


class ModelAPIBase(Protocol):
    api_base_url: types.Url
    semaphore: asyncio.Semaphore
    session: aiohttp.ClientSession
    model_params: Mapping[str, str | int | float]
    headers: ClassVar[Mapping[str, str]] = {"Content-Type": "application/json"}

    async def __aenter__(self) -> Self:
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.session.__aexit__(exc_type, exc_val, exc_tb)

    def api_endpoint(self, path: str) -> str:
        """Get an API endpoint url."""
        return urllib.parse.urljoin(self.api_base_url, path)

    async def change_model(self, model_name: str) -> None:
        """Change the loaded model.

        :raises OperationUnsupported: If the operation is not supported by the
        model server.
        """
        raise OperationNotSupported("Model changes not supported by this server.")

    async def query_model(self, prompt: str) -> Mapping:
        """Fetch data from the API endpoint.

        :param propmt: The prompt to send to the model.

        :return: Returns the request response as a decoded JSON object.
        """
        data = {"prompt": prompt, **self.params}
        async with self.semaphore:
            async with self.session.post(
                self.api_endpoint("completions"), headers=self.headers, json=data
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def make_queries(
        self,
        words: Sequence[types.Word],
        prompt: PromptHandler,
        *,
        max_invalid_output_retries=3,
        model_name: str | None = None,
    ) -> AsyncIterable[types.Word]:
        if model_name is not None:
            await self.change_model(model_name=model_name)

        retries: dict[str, int] = {}
        query_futures = {}
        for word in words:
            task = asyncio.create_task(
                self.query_model(prompt=prompt.make_prompt(word))
            )
            query_futures[task] = word

        pending = set(query_futures)
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )

            for future in done:
                word = query_futures.pop(future)
                try:
                    parsed, model = parse_results(await future)
                except (ValueError, json.decoder.JSONDecodeError):
                    retries[word.word] = retries.get(word.word, 0) + 1
                    if retries[word.word] <= max_invalid_output_retries:
                        print(f"Retrying {word.word} after invalid output")
                        task = asyncio.create_task(
                            self.query_model(prompt=prompt.make_prompt(word)),
                        )
                        pending.add(task)
                        query_futures[task] = word
                    else:
                        print(
                            f"Skipping {word.word} after {retries[word.word]} retries."
                        )
                    continue

                word.model_used = model
                yield result_to_word(word, parsed)


class TGWModelAPI(ModelAPIBase):
    """Model API for Text Generation WebUI."""

    def __init__(
        self,
        api_base_url: types.Url,
        max_connections: int = 1,
        query_params: Mapping[str, str | int | float] | None = None,
    ):
        self.semaphore = asyncio.Semaphore(max_connections)
        params = {"max_tokens": 8192, "request_timeout": 360}
        if query_params is not None:
            params.update(query_params)
        self.params: Mapping[str, str | int | float] = params
        self.api_base_url = api_base_url

    async def change_model(
        self,
        model_name: str,
    ) -> None:
        async with self.semaphore:
            async with self.session.get(
                self.api_endpoint("internal/model/info")
            ) as resp:
                resp.raise_for_status()
                model_info = await resp.json()
                if model_info["model_name"] == model_name:
                    return

            async with self.session.post(
                self.api_endpoint("internal/model/load"),
                json={"model_name": model_name},
            ) as resp:
                resp.raise_for_status()
                json_resp = await resp.json()
                print(json_resp)


class SentencePromptHandler(PromptHandler):
    def __init__(self, prompt_template: str):
        self.prompt_template = prompt_template

    def make_prompt(self, word: types.Word) -> str:
        return self.prompt_template % {
            "term": word.word,
            "meaning": word.meaning,
            "pos": word.part_of_speech,
        }


def parse_results(resp: Mapping[str, Any]) -> tuple[Mapping[str, str], str]:
    choices = resp["choices"]
    choice = choices[0]
    if "text" not in choice:
        raise ValueError(
            f"Failed to retrieve message for reason {choice['finish_reason']}"
        )
    elif choice["finish_reason"] != "stop":
        print(f"Query finished for reason {choice['finish_reason']}")
    try:
        return json.loads(choice["text"]), resp["model"]
    except json.decoder.JSONDecodeError:
        print(f"Failed to decode text:\n{choice['text']}")
        raise


def result_to_word(
    word: types.Word, parsed: Mapping[str, str | list[str]]
) -> types.Word:
    if not word.spanish_sentence:
        example_sentence = parsed["examples"].pop(
            random.randint(0, len(parsed["examples"]) - 1)
        )
        word.spanish_sentence = example_sentence["spanish"]
        word.english_sentence = example_sentence["english"]
    word.example_sentences = parsed["examples"]
    return word


class OperationUnsupported(NotImplementedError):
    """If an operation is not supported by an API endpoint."""
