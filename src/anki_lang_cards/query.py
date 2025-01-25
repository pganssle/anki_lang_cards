import asyncio
import itertools
import json
import random
import urllib.parse
from collections.abc import AsyncIterable, Iterable, Mapping, Sequence
from typing import Any, ClassVar, Final, Protocol, Self

import aiohttp
import attrs
import backoff

from . import _types as types


def _backoff_word_logger(details: Mapping[str, Any]) -> None:
    prompt: str = details["kwargs"].get("prompt", "")
    if len(prompt) > 100:
        prompt = "..." + prompt[-100:]

    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} and prompt:\n{prompt}".format(
            **details, prompt=prompt
        )
    )


class PromptHandler(Protocol):
    prompt_template: str

    def make_prompt(self, words: Iterable[types.Word]) -> str: ...


class ModelAPIBase(Protocol):
    api_base_url: types.Url
    semaphore: asyncio.Semaphore
    session: aiohttp.ClientSession
    model_params: Mapping[str, str | int | float]
    headers: ClassVar[Mapping[str, str]] = {"Content-Type": "application/json"}

    async def __aenter__(self) -> Self:
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.session.__aexit__(exc_type, exc_val, exc_tb)

    def api_endpoint(self, path: str) -> str:
        """Get an API endpoint url."""
        return urllib.parse.urljoin(self.api_base_url, path)

    def parse_results(
        self, resp: Mapping[str, Any]
    ) -> tuple[Mapping[str, str], str]: ...

    async def change_model(self, model_name: str) -> None:
        """Change the loaded model.

        :raises OperationUnsupported: If the operation is not supported by the
        model server.
        """

        # TODO: Change this to mutate the API maybe
        raise OperationNotSupported("Model changes not supported by this server.")

    def batch_words(
        self, words: Iterable[types.Word]
    ) -> Iterable[Iterable[types.Word]]: ...

    @backoff.on_exception(
        backoff.expo,
        aiohttp.client_exceptions.ClientResponseError,
        on_backoff=[_backoff_word_logger],
    )
    async def query_model(self, prompt: str, system_prompt: str) -> Mapping:
        """Fetch data from the API endpoint.

        :param prompt: The prompt to send to the model.

        :return: Returns the request response as a decoded JSON object.
        """
        data = {"prompt": system_prompt + prompt, **self.params}
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
        max_invalid_output_retries=0,
    ) -> AsyncIterable[types.Word]:
        if getattr(self, "model_name", None) is not None:
            await self.change_model(model_name=self.model_name)

        retries: dict[str, int] = {}
        query_futures = {}
        for word_subset in map(tuple, self.batch_words(words)):
            task = asyncio.create_task(
                self.query_model(
                    prompt=prompt.make_prompt(word_subset),
                    system_prompt=prompt.system_prompt,
                )
            )
            query_futures[task] = word_subset

        pending = set(query_futures)
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )

            for future in done:
                word_subset = query_futures.pop(future)
                try:
                    parsed, model = self.parse_results(await future)
                except (ValueError, json.decoder.JSONDecodeError):
                    key = tuple(word.word for word in word_subset)
                    retries[key] = retries.get(key, 0) + 1
                    if retries[key] <= max_invalid_output_retries:
                        print(f"Retrying {key} after invalid output")
                        task = asyncio.create_task(
                            self.query_model(prompt=prompt.make_prompt(word_subset)),
                        )
                        pending.add(task)
                        query_futures[task] = word_subset
                    else:
                        print(f"Skipping {key} after {retries[key]} retries.")
                    continue

                citation = types.ModelCitation(model=model)
                def _add_citation(response_data: dict) -> dict:
                    new_examples = []
                    for example in response_data["examples"]:
                        new_example = {**example, "source": citation}
                        new_examples.append(new_example)
                    return {**response_data, "examples": new_examples}

                parsed_with_sources = [_add_citation(entry) for entry in parsed]

                for word in result_to_words(word_subset, parsed_with_sources):
                    word.model_used = model
                    yield word
                # for word in word_subset:
                #     word.model_used = model
                # yield from result_to_words(word_subset, parsed)
                # yield result_to_word(word, parsed)


class TGWModelAPI(ModelAPIBase):
    """Model API for Text Generation WebUI."""

    DEFAULT_MODEL: str = "Meta-Llama-3-70B-Instruct-IQ1_M.gguf"

    def __init__(
        self,
        api_base_url: types.Url,
        max_connections: int = 1,
        query_params: Mapping[str, str | int | float] | None = None,
        request_timeout: int = 360,
        max_tokens: int = 8192,
        batch_size: int = 2,
        model_name: str | None = None,
    ):
        self.semaphore = asyncio.Semaphore(max_connections)
        params: dict[str, str | int | float] = {
            "max_tokens": max_tokens,
            "request_timeout": request_timeout,
        }
        if query_params is not None:
            params.update(query_params)
        self.params: Mapping[str, str | int | float] = params
        self.api_base_url = api_base_url
        self.batch_size = batch_size
        if model_name is None:
            model_name = self.DEFAULT_MODEL
        self.model_name = model_name

    def parse_results(self, resp: Mapping[str, Any]) -> tuple[Mapping[str, str], str]:
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

    def batch_words(
        self, words: Iterable[types.Word]
    ) -> Iterable[Iterable[types.Word]]:
        return _batched(words, self.batch_size)

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


class AnthropicAPI(ModelAPIBase):
    DEFAULT_MODEL: str = "claude-3-5-sonnet-20241022"

    def __init__(
        self,
        api_key: str,
        max_connections: int = 1,
        model_name: str | None = None,
        query_params: Mapping[str, str | int | float] | None = None,
        batch_size: int = 15,
        max_tokens: int = 8192,
    ):
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        if model_name is None:
            model_name = self.DEFAULT_MODEL
        self.model = model_name
        self.semaphore = asyncio.Semaphore(max_connections)
        self.query_params = query_params or {}
        self.query_params["max_tokens"] = max_tokens
        self.batch_size = batch_size

    def batch_words(
        self, words: Iterable[types.Word]
    ) -> Iterable[Iterable[types.Word]]:
        return _batched(words, self.batch_size)

    async def query_model(self, prompt: str, system_prompt: str) -> Mapping:
        # TODO: Cache and log results to avoid duplicate successful queries
        async with self.semaphore:
            message = await self._client.messages.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
                system=system_prompt,
                **self.query_params,
            )

        usage = message.usage  # input_tokens, output_tokens
        model = message.model
        response_text = message.content[0].text
        if message.stop_reason != "end_turn":
            print(f"Warning: stop_reason is {message.stop_reason}")
        return message

    def parse_results(self, resp: Mapping[str, Any]) -> tuple[Mapping[str, str], str]:

        response_content = resp.content[0].text
        try:
            return json.loads(response_content), resp.model
        except json.decoder.JSONDecodeError:
            print(f"Failed to decode text:\n{response_content}")
            raise


class SentencePromptHandler(PromptHandler):
    def __init__(self, prompt_template: str):
        self.prompt_template = prompt_template

    def make_prompt(self, words: Iterable[types.Word]) -> str:
        word = words[0]
        return self.prompt_template % {
            "term": word.word,
            "meaning": word.meaning,
            "pos": word.part_of_speech,
        }


if not hasattr(itertools, "batched"):

    def _batched(iterable, n):
        # batched('ABCDEFG', 3) → ABC DEF G
        if n < 1:
            raise ValueError("n must be at least one")
        iterator = iter(iterable)
        while batch := tuple(itertools.islice(iterator, n)):
            yield batch

else:
    _batched = itertools.batched


class MultiSentencePromptHandler(PromptHandler):
    def __init__(self, prompt_template: str, system_prompt: str = ""):
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt

    def make_prompt(self, words: Iterable[types.Word]) -> str:
        prompt_data = [
            f"""\
    {{
        "term": "{word.word}",
        "part_of_speech": "{word.part_of_speech}",
        "meaning": "{word.meaning}"
    }}"""
            for word in words
        ]
        return self.prompt_template % {"data": ",\n".join(prompt_data)}


def result_to_word(
    word: types.Word, parsed: Mapping[str, str | list[str]]
) -> types.Word:

    if not word.spanish_sentence:
        example_sentence = parsed["examples"].pop(
            random.randint(0, len(parsed["examples"]) - 1)
        )
        word.spanish_sentence = example_sentence["spanish"]
        word.english_sentence = example_sentence["english"]
        word.sentence_source = example_sentence["source"]
    word.example_sentences = parsed["examples"]
    return word


def result_to_words(
    words: Sequence[types.Word],
    parsed: Sequence[Mapping[str, Any]]
) -> Iterable[types.Word]:
    word_map = {word.word: word for word in words}
    for parsed_word in parsed:
        try:
            yield result_to_word(word_map[parsed_word["term"]], parsed_word)
        except KeyError:
            print(f"Term mismatch for {parsed_word}, skipping.")


class OperationUnsupported(NotImplementedError):
    """If an operation is not supported by an API endpoint."""
