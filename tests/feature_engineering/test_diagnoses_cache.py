"""Testing suite for the diagnoses caching module."""

import datetime
import json
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.feature_engineering import diagnoses_cache
from thesis.feature_engineering.diagnoses_cache import (
    ENRICH_VERSION,
    _fingerprint,
    compose_enriched,
)


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


def _mutate_base_sidecar(
    make_uo_file: Callable, make_base_sidecar: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overwrites the base sidecar with different JSON (a base rebuild)."""
    make_base_sidecar({"version": 999})


def _mutate_uo_source(
    make_uo_file: Callable, make_base_sidecar: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grows a UO source file so its st_size changes (re-derived data)."""
    make_uo_file("weight_data", "mock data with many more bytes")


def _mutate_enrich_version(
    make_uo_file: Callable, make_base_sidecar: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bumps the module-level ENRICH_VERSION (a diagnosis-logic change)."""
    monkeypatch.setattr(diagnoses_cache, "ENRICH_VERSION", 999)


@pytest.mark.parametrize(
    "mutate",
    [_mutate_base_sidecar, _mutate_uo_source, _mutate_enrich_version],
    ids=["base_sidecar_changed", "uo_source_grew", "enrich_version_bumped"],
)
def test_fingerprint_detects_staleness(
    make_uo_file: Callable,
    make_base_sidecar: Callable,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable,
) -> None:
    """Asserts any tracked input changing produces a different fingerprint."""
    weight_data_path = make_uo_file("weight_data", "mock data")
    uo_data_path = make_uo_file("uo_data", "mock data")
    base_sidecar_path = make_base_sidecar({"version": 2, "manifest": "abc"})
    uo_sources = [weight_data_path, uo_data_path]

    fingerprint_before = _fingerprint(base_sidecar_path, uo_sources)
    mutate(make_uo_file, make_base_sidecar, monkeypatch)
    fingerprint_after = _fingerprint(base_sidecar_path, uo_sources)

    assert fingerprint_before != fingerprint_after


def test_fingerprint_stable_when_unchanged(
    make_uo_file: Callable, make_base_sidecar: Callable
) -> None:
    """Asserts an unchanged cache recomputes to an identical fingerprint."""
    weight_data_path = make_uo_file("weight_data", "mock data")
    uo_data_path = make_uo_file("uo_data", "mock data")
    base_sidecar_path = make_base_sidecar({"version": 2, "manifest": "abc"})
    uo_sources = [weight_data_path, uo_data_path]

    assert _fingerprint(base_sidecar_path, uo_sources) == _fingerprint(
        base_sidecar_path, uo_sources
    )


def test_fingerprint_survives_json_round_trip(
    make_uo_file: Callable, make_base_sidecar: Callable
) -> None:
    """Asserts the fingerprint equals its own JSON round-trip."""
    weight_data_path = make_uo_file("weight_data", "mock data")
    uo_data_path = make_uo_file("uo_data", "mock data")
    base_sidecar_path = make_base_sidecar({"version": 2, "manifest": "abc"})

    fingerprint = _fingerprint(base_sidecar_path, [weight_data_path, uo_data_path])

    assert fingerprint == json.loads(json.dumps(fingerprint))


@pytest.fixture
def make_base_parquet(tmp_path: Path) -> Callable[..., Path]:
    """Returns a factory writing a wide base-events parquet, returning its path.

    The base carries a ``labevents/valuenum`` column absent from the diagnoses
    frame, so the diagonal concat must null-fill it on the diagnosis rows.
    """

    def _make(**overrides: list) -> Path:
        defaults = {
            "event_type": ["labevents", "labevents"],
            "patient_id": ["1", "2"],
            "hadm_id": ["1", "2"],
            "timestamp": pl.Series(
                [
                    datetime.datetime(2025, 1, 1, 0),
                    datetime.datetime(2025, 1, 2, 0),
                ],
                dtype=pl.Datetime("ns"),
            ),
            "labevents/valuenum": [1.25, 2.5],
        }
        defaults.update(overrides)
        path = tmp_path / "events.parquet"
        pl.DataFrame(defaults).write_parquet(path)
        return path

    return _make


@pytest.fixture
def make_diagnoses_parquet(tmp_path: Path) -> Callable[..., Path]:
    """Returns a factory writing a narrow diagnoses parquet, returning its path.

    Carries a ``diagnosis_made/diagnosis`` column absent from the base frame,
    so the diagonal concat must null-fill it on the base rows.
    """

    def _make(**overrides: list) -> Path:
        defaults = {
            "event_type": ["diagnosis_made"],
            "patient_id": ["1"],
            "hadm_id": ["1"],
            "timestamp": pl.Series(
                [datetime.datetime(2025, 1, 3, 0)], dtype=pl.Datetime("ns")
            ),
            "diagnosis_made/diagnosis": ["Acute Kidney Injury"],
        }
        defaults.update(overrides)
        path = tmp_path / "diagnoses.parquet"
        pl.DataFrame(defaults).write_parquet(path)
        return path

    return _make


def test_compose_enriched_unions_base_and_diagnoses(
    make_base_parquet: Callable, make_diagnoses_parquet: Callable
) -> None:
    """Asserts the diagonal concat unions both event types and null-fills columns."""
    base_parquet = make_base_parquet()
    diagnoses_parquet = make_diagnoses_parquet()

    expected_lf = pl.LazyFrame(
        {
            "event_type": ["labevents", "labevents", "diagnosis_made"],
            "patient_id": ["1", "2", "1"],
            "hadm_id": ["1", "2", "1"],
            "timestamp": pl.Series(
                [
                    datetime.datetime(2025, 1, 1, 0),
                    datetime.datetime(2025, 1, 2, 0),
                    datetime.datetime(2025, 1, 3, 0),
                ],
                dtype=pl.Datetime("ns"),
            ),
            "labevents/valuenum": [1.25, 2.5, None],
            "diagnosis_made/diagnosis": [None, None, "Acute Kidney Injury"],
        }
    )

    assert_frame_equal(
        compose_enriched(base_parquet, diagnoses_parquet),
        expected_lf,
        check_row_order=False,
    )


def test_compose_enriched_preserves_shared_timestamp_dtype(
    make_base_parquet: Callable, make_diagnoses_parquet: Callable
) -> None:
    """Guards the shared timestamp column keeps a single dtype."""
    base_parquet = make_base_parquet()
    diagnoses_parquet = make_diagnoses_parquet()

    schema = compose_enriched(base_parquet, diagnoses_parquet).collect_schema()

    assert schema["timestamp"] == pl.Datetime("ns")
