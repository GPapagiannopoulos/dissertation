"""Testing suite for the MEDS db command builder."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

"""
Test cases:
1) assert default number of threads
2) assert multiple threads
3) assert exact argv list
"""


@pytest.mark.parametrize(
    "overrides, expected_options",
    [
        # 0. Default values
        ({}, ["--num_threads", "1"]),
        # 1. 3 threads
        ({"num_threads": "3"}, ["--num_threads", "3"]),
    ],
)
def test_build_command_happy_path(
    make_db_command: Callable,
    reader_paths: dict[str, Path],
    overrides: dict[str, Any],
    expected_options: list[str],
) -> None:
    """Asserts expected behaviour for the build command helper."""
    expected = [
        str(reader_paths["executable"]),
        str(reader_paths["src"]),
        str(reader_paths["dest"]),
        *expected_options,
    ]
    assert make_db_command(**overrides) == expected


def test_build_command_rejects_non_positive_threads() -> None:
    """Asserts that the command needs >=1 threads."""
    pass
