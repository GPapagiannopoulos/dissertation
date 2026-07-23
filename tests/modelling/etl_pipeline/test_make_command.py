"""Testing suite for the build_command helper function."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


@pytest.mark.parametrize(
    "overrides, expected_options",
    [
        # 0. Defaults produce the documented argv
        (
            {},
            ["--num_shards", "100", "--num_proc", "1", "--backend", "cpp"],
        ),
        # 1. num_shards is forwarded as a string
        (
            {"num_shards": 10},
            ["--num_shards", "10", "--num_proc", "1", "--backend", "cpp"],
        ),
        # 2. num_proc is forwarded as a string
        (
            {"num_proc": 4},
            ["--num_shards", "100", "--num_proc", "4", "--backend", "cpp"],
        ),
        # 3. The polars backend is selectable
        (
            {"backend": "polars"},
            ["--num_shards", "100", "--num_proc", "1", "--backend", "polars"],
        ),
        # 4. Every option can be overridden at once
        (
            {"num_shards": 200, "num_proc": 2, "backend": "polars"},
            ["--num_shards", "200", "--num_proc", "2", "--backend", "polars"],
        ),
    ],
)
def test_build_command_happy_path(
    make_command: Callable,
    command_paths: dict[str, Path],
    overrides: dict[str, Any],
    expected_options: list[str],
) -> None:
    """Asserts the full argv is assembled exactly as meds_etl_mimic expects."""
    expected = [
        str(command_paths["executable"]),
        str(command_paths["src"]),
        str(command_paths["dest"]),
        *expected_options,
    ]

    assert make_command(**overrides) == expected


def test_build_command_orders_positionals_source_before_destination(
    make_command: Callable, command_paths: dict[str, Path]
) -> None:
    """Asserts src precedes dest, since swapping them reads the output as input."""
    args = make_command()

    assert args[1] == str(command_paths["src"])
    assert args[2] == str(command_paths["dest"])


def test_build_command_returns_only_strings(make_command: Callable) -> None:
    """Asserts no Path survives into the argv, which subprocess would mishandle."""
    args = make_command(num_shards=10, num_proc=2, backend="polars")

    assert all(isinstance(arg, str) for arg in args)


@pytest.mark.parametrize(
    "backend",
    [
        # 0. The empty string is not a backend
        "",
        # 1. Backend matching is case sensitive
        "CPP",
        # 2. Unknown backends are rejected rather than forwarded
        "rust",
        # 3. None is not a stand-in for the default
        None,
    ],
)
def test_build_command_rejects_unknown_backend(
    make_command: Callable, backend: str | None
) -> None:
    """Asserts an unsupported backend fails here rather than at subprocess time."""
    with pytest.raises(ValueError):
        make_command(backend=backend)


@pytest.mark.parametrize(
    "overrides",
    [
        # 0. Zero shards produces no output
        {"num_shards": 0},
        # 1. Negative shards are meaningless
        {"num_shards": -1},
        # 2. Zero processes cannot do the work
        {"num_proc": 0},
        # 3. Negative processes are meaningless
        {"num_proc": -1},
    ],
)
def test_build_command_rejects_non_positive_counts(
    make_command: Callable, overrides: dict[str, int]
) -> None:
    """Asserts shard and process counts are validated before the ETL is launched."""
    with pytest.raises(ValueError):
        make_command(**overrides)
