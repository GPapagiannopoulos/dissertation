"""Testing suite for the diagnoses caching module."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from thesis.feature_engineering.diagnoses_cache import ENRICH_VERSION, _fingerprint


@pytest.fixture
def make_uo_file(tmp_path: Path) -> Callable[[str, str], Path]:
    """Returns a factory that writes a mock UO parquet and hands back its path.

    The fingerprinting function only checks the .stat() properties of the files
    at the 'uo_sources' paths. To mock the object we only need to write bytes.

    Args:
        tmp_path (Path): temporary directory path cleaned up automatically
            by Pytest at the end of the test.

    Returns:
        Path: the path to the mock file.
    """

    def _make(name: str, data: str) -> Path:
        data_path = tmp_path / f"{name}.parquet"
        data_path.write_bytes(data.encode())
        return data_path

    return _make


@pytest.fixture
def make_base_sidecar(tmp_path: Path) -> Callable[[dict], Path]:
    """Returns a factory that writes a mock sidecar object for the base dataset.

    The sidecar needs to be valid JSON. We write our mock data as a stringified
    JSON object at the tmp_path.

    Args:
         tmp_path (Path): temporary directory path cleaned up automatically
            by Pytest at the end of the test.

    Returns:
        Path: the path to the mock sidecar
    """

    def _make(contents: dict) -> Path:
        sidecar_path = tmp_path / "sidecar.parquet"
        sidecar_path.write_text(json.dumps(contents))
        return sidecar_path

    return _make


def test_fingerprint_embeds_pipeline_version(
    make_uo_file: Callable, make_base_sidecar: Callable
) -> None:
    """Asserts that part of the fingerprint is the current cache version."""
    weight_data_path = make_uo_file("weight_data", "mock data")
    uo_data_path = make_uo_file("uo_data", "mock data")
    base_sidecar_path = make_base_sidecar({"category": "data"})

    assert (
        _fingerprint(base_sidecar_path, [weight_data_path, uo_data_path])[
            "enrich_version"
        ]
        == ENRICH_VERSION
    )


def test_fingerprint_embeds_base_sidecar(
    make_uo_file: Callable, make_base_sidecar: Callable
) -> None:
    """Asserts that the base dataset sidecar is embedded by fingerprint."""
    weight_data_path = make_uo_file("weight_data", "mock data")
    uo_data_path = make_uo_file("uo_data", "mock data")
    base_sidecar_path = make_base_sidecar({"version": 2, "manifest": "abc"})

    assert _fingerprint(base_sidecar_path, [weight_data_path, uo_data_path])[
        "base"
    ] == {"version": 2, "manifest": "abc"}


def test_fingerprint_embeds_uo_metadata(
    make_uo_file: Callable, make_base_sidecar: Callable
) -> None:
    """Asserts that metadata is embedded by _fingerprint."""
    weight_data_path = make_uo_file("weight_data", "mock data")
    uo_data_path = make_uo_file("uo_data", "mock data")
    base_sidecar_path = make_base_sidecar({"version": 2, "manifest": "abc"})

    assert (
        len(
            _fingerprint(base_sidecar_path, [weight_data_path, uo_data_path])[
                "uo_sources"
            ]
        )
        == 2
    )
