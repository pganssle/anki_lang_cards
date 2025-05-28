import hashlib
import json
import os
import pathlib
import tempfile
import time
from collections.abc import Callable, Sequence

import hypothesis
import pytest
import time_machine
from hypothesis import strategies as st

from anki_lang_cards import caches


@pytest.fixture
def query_cacher_default(tmp_path: pathlib.Path) -> caches.QueryCacher:
    """Default QueryCacher fixture."""
    cache_loc = tmp_path / ".caches"

    # Use gpt4 because it's the least typing 😅
    return caches.QueryCacher(
        cache_type=caches.CacheTypes.RESPONSES,
        model_name="gpt4",
        model_service="openai",
        cache_loc=cache_loc,
    )


@pytest.fixture
def query_cacher_custom(tmp_path: pathlib.Path) -> caches.QueryCacher:
    """QueryCacher with custom settings."""
    return caches.QueryCacher(
        cache_type=caches.CacheTypes.ANNOTATIONS,
        model_name="claude-3",
        model_service="anthropic",
        cache_loc=tmp_path,
        by_service=False,
        by_model=False,
        filename_generator=caches.datetime_filename,
    )


@st.composite
def query_cacher_constructor_strategy(
    draw, filename_generators: Sequence[Callable[[str, str | None], str]] = ()
) -> Callable[[pathlib.Path], caches.QueryCacher]:
    """Returns a function that makes a random QueryCacher"""
    filename_char_strategy = st.characters(codec="utf-8", exclude_characters=";:/\x00")
    filename_text_strategy = st.text(
        alphabet=filename_char_strategy, min_size=1, max_size=128
    )
    cache_type = draw(st.sampled_from(list(caches.CacheTypes)))
    model_name = draw(filename_text_strategy)
    model_service = draw(filename_text_strategy)
    by_service = draw(st.booleans())
    by_model = draw(st.booleans())

    if filename_generators:
        filename_generator = draw(st.sampled_from(filename_generators))
    else:
        filename_generator = caches.prompt_hash

    def make_query_cacher(
        cache_loc: pathlib.Path,
        cache_type=cache_type,
        model_name=model_name,
        model_service=model_service,
        by_service=by_service,
        by_model=by_model,
        filename_generator=filename_generator,
    ) -> caches.QueryCacher:
        return caches.QueryCacher(
            cache_type=cache_type,
            cache_loc=cache_loc,
            model_name=model_name,
            model_service=model_service,
            by_service=by_service,
            by_model=by_model,
            filename_generator=filename_generator,
        )

    return make_query_cacher


def test_prompt_hash() -> None:
    """Test the prompt_hash function with and without system prompt."""
    # TODO: Replace this with something where the hash function can be specified
    # Test with no system prompt
    prompt = "What is the capital of France?"
    expected_hash = hashlib.md5(
        prompt.encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    assert caches.prompt_hash(prompt, None) == expected_hash

    # Test with system prompt
    system_prompt = "You are a helpful assistant."
    combined = system_prompt + prompt
    expected_hash_with_system = hashlib.md5(
        combined.encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    assert caches.prompt_hash(prompt, system_prompt) == expected_hash_with_system


@time_machine.travel("2023-01-01T12:00:00")
@pytest.mark.parametrize(
    ("prompt", "system_prompt"),
    [
        ("Test Prompt", None),
        ("Test Prompt", "You are a helpful assistant"),
    ],
)
@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        (None, "response-2023-01-01T120000"),
        ("custom-", "custom-2023-01-01T120000"),
    ],
)
def test_datetime_filename(
    prompt: str, system_prompt: str | None, prefix: str | None, expected: str
) -> None:
    """Test datetime_filename with fixed time."""
    if prefix is None:
        kwargs = {}
    else:
        kwargs = {"prefix": prefix}

    assert caches.datetime_filename(prompt, system_prompt, **kwargs) == expected


def test_cache_dir_structure(tmp_path: pathlib.Path) -> None:
    """Test that cache_dir property creates the expected structure."""
    # Override the default cache location for testing
    cacher = caches.QueryCacher(
        cache_type=caches.CacheTypes.RESPONSES,
        model_name="gpt-4",
        model_service="openai",
        cache_loc=tmp_path,
    )

    expected_dir = tmp_path / "responses" / "openai" / "gpt-4"
    assert cacher.cache_dir == expected_dir


@pytest.mark.parametrize(
    ("model_name", "model_service", "by_model", "by_service", "expected"),
    [
        ("gpt-4", "openai", False, False, pathlib.Path("responses")),
        ("claude-3", "anthropic", False, False, pathlib.Path("responses")),
        ("gpt-4", "openai", True, False, pathlib.Path("responses") / "gpt-4"),
        ("claude-3", "anthropic", True, False, pathlib.Path("responses") / "claude-3"),
        ("gpt-4", "openai", False, True, pathlib.Path("responses") / "openai"),
        ("claude-3", "anthropic", False, True, pathlib.Path("responses") / "anthropic"),
        ("gpt-4", "openai", True, True, pathlib.Path("responses") / "openai" / "gpt-4"),
        (
            "claude-3",
            "anthropic",
            True,
            True,
            pathlib.Path("responses") / "anthropic" / "claude-3",
        ),
    ],
)
def test_cache_dir_structure_options(
    tmp_path: pathlib.Path,
    model_name: str,
    model_service: str,
    by_model: bool,
    by_service: bool,
    expected: pathlib.Path,
) -> None:
    """Test cache_dir with different by_service and by_model options."""

    expected_dir = tmp_path / expected

    cacher = caches.QueryCacher(
        cache_type=caches.CacheTypes.RESPONSES,
        cache_loc=tmp_path,
        model_name=model_name,
        model_service=model_service,
        by_model=by_model,
        by_service=by_service,
    )

    assert cacher.cache_dir == expected_dir


def test_llm_cache_file_loc(query_cacher_default: caches.QueryCacher) -> None:
    """Test that llm_cache_file_loc generates the expected path."""
    prompt = "What is the weather today?"
    system_prompt = "You are a helpful assistant."

    # Get expected hash
    hash_value = caches.prompt_hash(prompt, system_prompt)
    expected_path = (
        query_cacher_default.cache_loc
        / "responses"
        / "openai"
        / "gpt4"
        / f"{hash_value}.json"
    )

    assert (
        query_cacher_default.llm_cache_file_loc(prompt, system_prompt) == expected_path
    )


def test_llm_cache_file_loc_custom_generator(tmp_path: pathlib.Path) -> None:
    """Test llm_cache_file_loc with a custom filename generator."""

    def custom_generator(prompt, system_prompt):
        return f"custom-{len(prompt)}"

    cacher = caches.QueryCacher(
        cache_type=caches.CacheTypes.RESPONSES,
        model_name="gpt-4",
        model_service="openai",
        cache_loc=tmp_path,
        filename_generator=custom_generator,
    )

    prompt = "What is the weather today?"
    expected_path = tmp_path / "responses" / "openai" / "gpt-4" / "custom-26.json"

    assert cacher.llm_cache_file_loc(prompt, None) == expected_path


@pytest.mark.parametrize(
    ("enum_obj", "str_repr"),
    [
        (caches.CacheTypes.RESPONSES, "responses"),
        (caches.CacheTypes.ANNOTATIONS, "annotations"),
        (caches.CacheTypes.SENTENCES, "sentences"),
    ],
)
def test_cache_type_enum_values(enum_obj: caches.CacheTypes, str_repr: str) -> None:
    """Test if CacheTypes enum has the expected values."""
    assert enum_obj == str_repr


@pytest.mark.parametrize(
    ("cache_type", "expected_component"),
    [
        (caches.CacheTypes.RESPONSES, "responses"),
        (caches.CacheTypes.ANNOTATIONS, "annotations"),
        (caches.CacheTypes.SENTENCES, "sentences"),
    ],
)
def test_cache_dir_loc_by_type(
    cache_type: caches.CacheTypes, expected_component: str, tmp_path: pathlib.Path
) -> None:
    """Test that the cache directory depends on the cache type."""
    cacher = caches.QueryCacher(
        cache_type=cache_type,
        cache_loc=tmp_path,
        model_name="gpt-4",
        model_service="openai",
        by_model=False,
        by_service=False,
    )

    expected = tmp_path / expected_component
    assert cacher.cache_dir == expected


def test_load_from_cache_nonexistent(query_cacher_default: caches.QueryCacher) -> None:
    """Test that load_from_cache returns None for nonexistent cache files."""
    result = query_cacher_default.load_from_cache("nonexistent prompt")
    assert result is None


@hypothesis.given(
    prompt=st.text(min_size=1, max_size=10000),
    system_prompt=st.one_of(st.text(min_size=1, max_size=10000), st.none()),
    response_text=st.text(min_size=1, max_size=10000),
    query_cacher_fn=query_cacher_constructor_strategy(),
)
def test_cache_file_round_trip(
    prompt: str,
    system_prompt: str | None,
    response_text: str,
    query_cacher_fn: Callable[[pathlib.Path], caches.QueryCacher],
) -> None:
    # TODO: Add query parameters

    # We need to use a temporary directory context manager here because function-scoped
    # fixtures in pytest are not reset for each hypothesis test.
    with tempfile.TemporaryDirectory() as td:
        cacher = query_cacher_fn(pathlib.Path(td))
        response = {
            "text": response_text,
            "usage": {
                "input": int(
                    (len(prompt) + len(system_prompt) if system_prompt else 0) / 4
                ),
                "output": int(len(response_text) / 4),
                "details": None,
            },
        }

        # Write to disk
        cacher.cache_result(
            prompt=prompt, response=response, system_prompt=system_prompt
        )

        # Read from disk
        cached_result = cacher.load_from_cache(prompt, system_prompt)

        assert cached_result["response"] == response


@hypothesis.given(
    prompts_with_responses=st.lists(
        st.tuples(
            st.text(min_size=1, max_size=10000),
            st.text(min_size=1, max_size=10000),
        ),
        min_size=1,
        max_size=10,
    ),
    system_prompt=st.one_of(st.text(min_size=1, max_size=10000), st.none()),
    query_cacher_fn=query_cacher_constructor_strategy(
        filename_generators=[caches.prompt_hash, caches.datetime_filename]
    ),
)
def test_cache_file_round_trip_time_based(
    prompts_with_responses: Sequence[tuple[str, str]],
    system_prompt: str | None,
    query_cacher_fn: Callable[[pathlib.Path], caches.QueryCacher],
) -> None:
    # TODO: Add query parameters

    # We need to use a temporary directory context manager here because function-scoped
    # fixtures in pytest are not reset for each hypothesis test.
    with tempfile.TemporaryDirectory() as td:
        cacher = query_cacher_fn(pathlib.Path(td))
        for prompt, response_text in prompts_with_responses:
            response = {
                "text": response_text,
                "usage": {
                    "input": int(
                        (len(prompt) + len(system_prompt) if system_prompt else 0) / 4
                    ),
                    "output": int(len(response_text) / 4),
                    "details": None,
                },
            }

            # Write to disk
            cacher.cache_result(
                prompt=prompt, response=response, system_prompt=system_prompt
            )

        # Read only the latest one from disk
        cached_result = cacher.load_latest_cache()

        assert cached_result["input"]["prompt"] == prompt
        assert cached_result["input"].get("system_prompt") == system_prompt
        assert cached_result["response"] == response


@hypothesis.given(
    prompt=st.text(min_size=1, max_size=1000),
    system_prompt=st.one_of(st.none(), st.text(min_size=1, max_size=1000)),
)
def test_prompt_hash_consistency(prompt, system_prompt):
    """Test that prompt_hash produces consistent results for the same inputs."""
    hash1 = caches.prompt_hash(prompt, system_prompt)
    hash2 = caches.prompt_hash(prompt, system_prompt)
    assert hash1 == hash2


@hypothesis.given(
    prompt=st.text(min_size=1, max_size=1000),
    system_prompt=st.one_of(st.none(), st.text(min_size=1, max_size=1000)),
    prefix=st.text(min_size=1, max_size=20),
)
def test_datetime_filename_starts_with_prefix(prompt, system_prompt, prefix):
    """Test that datetime_filename always starts with the provided prefix."""
    filename = caches.datetime_filename(prompt, system_prompt, prefix=prefix)
    assert filename.startswith(prefix)


def test_load_latest_cache_no_json_files(
    query_cacher_custom: caches.QueryCacher, tmp_path: pathlib.Path
) -> None:
    """Test load_latest_cache with directory containing no JSON files."""
    # Create directory with non-JSON files
    tmp_path.mkdir(exist_ok=True, parents=True)
    (tmp_path / "file.txt").write_text("This is not a JSON file")

    assert query_cacher_custom.cache_loc == tmp_path
    assert query_cacher_custom.load_latest_cache() is None


def test_load_latest_cache_multiple_files(
    query_cacher_custom: caches.QueryCacher, tmp_path: pathlib.Path
) -> None:
    """Test load_latest_cache returns the most recently modified file."""

    # Create directory with multiple JSON files with different mtimes
    cache_dir = query_cacher_custom.cache_dir
    cache_dir.mkdir(exist_ok=True, parents=True)

    # Create sample cache data
    cache_data1 = {
        "input": {"prompt": "First prompt", "timestamp": "2023-01-01T12:00:00"},
        "response": {"text": "First response"},
    }

    cache_data2 = {
        "input": {"prompt": "Second prompt", "timestamp": "2023-01-02T12:00:00"},
        "response": {"text": "Second response"},
    }

    # Write files with different modification times
    file1 = cache_dir / "file1.json"
    file2 = cache_dir / "file2.json"

    file1.write_text(json.dumps(cache_data1))
    # Ensure a different mtime in the files
    f1_stat = file1.stat()
    os.utime(file1, (f1_stat.st_atime - 1, f1_stat.st_mtime - 1))
    file2.write_text(json.dumps(cache_data2))

    # Load the latest cache
    result = query_cacher_custom.load_latest_cache()

    # Verify we got the second (more recent) file
    assert result == cache_data2


def test_cache_result_creates_directories(tmp_path: pathlib.Path) -> None:
    """Test that cache_result creates necessary directories."""
    # Mock cache_dir to use a deep path in tmp_path
    deep_path = tmp_path / "deep" / "nested" / "dir"
    query_cacher = caches.QueryCacher(
        cache_type=caches.CacheTypes.ANNOTATIONS,
        model_name="claude-3",
        model_service="anthropic",
        cache_loc=deep_path,
        by_service=False,
        by_model=False,
        filename_generator=caches.datetime_filename,
    )

    # Test data
    prompt = "Test prompt"
    response_data = {
        "text": "Test response",
        "usage": {"input": 10, "output": 12, "details": None},
    }

    # Cache the result
    query_cacher.cache_result(prompt, response_data)

    # Verify the directory was created
    assert deep_path.exists()
    assert deep_path.is_dir()


@time_machine.travel("2023-01-01T12:00:00")
def test_cache_result_writes_correct_data(query_cacher_default):
    """Test that cache_result writes the expected data to the cache file."""
    # Test data
    prompt = "Test prompt"
    system_prompt = "System prompt"
    response_data = {
        "text": "Test response",
        "usage": {"input": 10, "output": 12, "details": None},
    }
    query_params = {"temperature": 0.7, "max_tokens": 100}

    # Get expected file location
    file_loc = query_cacher_default.llm_cache_file_loc(prompt, system_prompt)

    # Cache the result
    query_cacher_default.cache_result(
        prompt, response_data, query_params, system_prompt
    )

    # Verify the file exists
    assert file_loc.exists()

    # Load and verify the cached data
    cached_data = json.loads(file_loc.read_text())

    # Check structure and content
    assert "input" in cached_data
    assert "response" in cached_data
    assert cached_data["input"]["model_name"] == query_cacher_default.model_name
    assert cached_data["input"]["model_service"] == query_cacher_default.model_service
    assert cached_data["input"]["prompt"] == prompt
    assert cached_data["input"]["system_prompt"] == system_prompt
    assert cached_data["input"]["query_params"] == query_params
    assert cached_data["input"]["timestamp"] == "2023-01-01T12:00:00"
    assert cached_data["response"] == response_data


def test_cache_result_default_query_params(
    query_cacher_default: caches.QueryCacher,
) -> None:
    """Test that cache_result uses empty dict for default query_params."""
    # Test data
    prompt = "Test prompt"
    response_data = {
        "text": "Test response",
        "usage": {"input": 10, "output": 12, "details": None},
    }

    # Cache the result without query_params
    query_cacher_default.cache_result(prompt, response_data)

    # Get file location and load cached data
    file_loc = query_cacher_default.llm_cache_file_loc(prompt, None)
    cached_data = json.loads(file_loc.read_text())

    # Verify query_params is an empty dict
    assert cached_data["input"]["query_params"] == {}


# Integration tests that cover multiple components
def test_full_cache_and_load_cycle(tmp_path: pathlib.Path) -> None:
    """Test the full cycle of caching and loading a result."""
    # Create a cacher
    cacher = caches.QueryCacher(
        cache_type=caches.CacheTypes.RESPONSES,
        model_name="gpt-4",
        model_service="openai",
        cache_loc=tmp_path,
    )

    # Test data
    prompt = "What is the capital of France?"
    system_prompt = "You are a helpful assistant."
    response_data = {
        "text": "The capital of France is Paris.",
        "usage": {"input": 12, "output": 6, "details": None},
    }

    # Cache the result
    cacher.cache_result(prompt, response_data, system_prompt=system_prompt)

    # Load from cache
    loaded_result = cacher.load_from_cache(prompt, system_prompt)

    # Verify loaded data matches original
    assert loaded_result is not None
    assert loaded_result["response"] == response_data
    assert loaded_result["input"]["prompt"] == prompt
    assert loaded_result["input"]["system_prompt"] == system_prompt


def test_load_latest_with_multiple_results(tmp_path):
    """Test that load_latest_cache correctly identifies the most recent result."""
    # Create a cacher
    cacher = caches.QueryCacher(
        cache_type=caches.CacheTypes.RESPONSES,
        model_name="gpt-4",
        model_service="openai",
        cache_loc=tmp_path,
        filename_generator=caches.datetime_filename,
    )

    # Create multiple results with delays to ensure different mtimes
    prompts = ["First", "Second", "Third"]
    responses = []

    # Starting utime is 100 seconds ago, will increment by 0.1s each time
    utime: int = time.time_ns() - 100_000_000_000
    for i, prompt in enumerate(prompts):
        response_data = {
            "text": f"Response {i+1}",
            "usage": {"input": 10, "output": 10, "details": None},
        }
        responses.append(response_data)

        # Cache the result
        cache_file = cacher.cache_result(prompt, response_data)

        os.utime(cache_file, ns=(utime, utime))
        utime += 100_000_000

    # Load the latest result
    latest = cacher.load_latest_cache()

    # Verify we got the most recent one
    assert latest is not None
    assert latest["response"] == responses[-1]
    assert latest["input"]["prompt"] == prompts[-1]
