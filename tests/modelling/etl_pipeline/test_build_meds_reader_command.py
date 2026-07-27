"""Testing suite for the MEDS db command builder."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


@pytest.mark.parametrize(
    "overrides, expected_options",
    [
        # 0. Default values
        ({}, ["--num_threads", "1"]),
        # 1. 3 threads
        ({"num_threads": 3}, ["--num_threads", "3"]),
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


@pytest.mark.parametrize(
    "threads",
    [
        # 0. 0 threads raises
        {"num_threads": 0},
        # 1. Negative threads raises
        {"num_threads": -2},
    ],
)
def test_build_command_rejects_non_positive_threads(
    make_db_command: Callable, threads: dict[str, int]
) -> None:
    """Asserts that the command needs >=1 threads."""
    with pytest.raises(
        ValueError,
        match=f"The minimum number of threads is 1. Received {threads['num_threads']}",
    ):
        make_db_command(**threads)


def test_build_command_returns_strings(make_db_command: Callable) -> None:
    """Asserts that the arg list is all strings."""
    assert all(isinstance(arg, str) for arg in make_db_command())
