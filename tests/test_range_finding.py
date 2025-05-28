"""Tests for the logic that finds the highest rank.

This is technically testing the private interface, which is not ideal,
but it's a slightly tricky function. The problem is that all fields in the
Anki deck are strings, and I need to find the highest number by selecting
all the cards within certain ranges. To do that, I am generating a regular
expression that matches strings of numbers within a certain range.
"""

import re

import hypothesis
import pytest
from hypothesis import strategies as st

from anki_lang_cards import anki_import

NUM_STRATEGY = st.integers(min_value=1, max_value=1_000_000)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (1, 1000, {"[1-9]", "[1-9][0-9]", "[1-9][0-9][0-9]"}),
        (2512, 2519, {"251[2-8]"}),
        (5800, 7200, {"5[8-9][0-9][0-9]", "6[0-9][0-9][0-9]", "7[0-1][0-9][0-9]"}),
        (1900, 4300, {"19[0-9][0-9]", "[2-3][0-9][0-9][0-9]", "4[0-2][0-9][0-9]"}),
        (1477, 1478, {"1477"}),
    ],
)
def test_regex_patterns(left: int, right: int, expected: set[str]):
    """Test that fixed ranges generate specific patterns."""
    pattern = anki_import._generate_regex_pattern(left, right)
    actual = set(pattern.strip("()").split("|"))
    assert actual == expected


@hypothesis.given(
    numbers=st.sets(NUM_STRATEGY, max_size=1_000_000),
    regex_range=st.tuples(NUM_STRATEGY, NUM_STRATEGY).map(sorted),
)
def test_regex_generation(numbers: set[int], regex_range: tuple[int, int]):
    """Hypothesis test to verify that the regex generation matches the right range.

    This should generate a range to select (left-open interval), generate a
    regex pattern that matches it, then test it it against a randomly-generated
    list of numbers.
    """
    # Determine the valid range
    n, m = regex_range

    # Generate the regex pattern for the given range
    regex_pattern = anki_import._generate_regex_pattern(n, m)
    regex = re.compile(regex_pattern)

    # Filter numbers that fall within the range (n, m]
    expected_subset = {num for num in numbers if n <= num < m}

    # Match numbers using the regex
    matched_subset = {
        number for number in numbers if (regex.fullmatch(str(number))) is not None
    }

    # Validate that the matched numbers are exactly the expected subset
    assert matched_subset == expected_subset, f"Regex = {regex_pattern}"
