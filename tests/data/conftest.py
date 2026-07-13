"""Shared fixtures for the data-source test suite."""

from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from thesis.data.sources import PolarsEDASource


@pytest.fixture
def events_lf():
    """Factory for a valid PyHealth-shaped events frame with overrides.

    Returns a builder. Calling without args returns a valid default
    frame. Pass column=values or drop=[...] to modify.
    """

    def _build(drop: tuple[str, ...] = (), **overrides: list) -> pl.LazyFrame:
        defaults = {
            "patient_id": ["1", "2", "2"],
            "event_type": ["patients", "patients", "admissions"],
        }
        defaults.update(overrides)
        for col in drop:
            defaults.pop(col, None)
        return pl.LazyFrame(defaults)

    return _build


@pytest.fixture
def make_source(events_lf):
    """Factory for returning a constructed PolarsEDASource over a valid frame."""

    def _make(**kwargs) -> PolarsEDASource:
        return PolarsEDASource(events_lf(**kwargs))

    return _make


@pytest.fixture
def make_eav_source(make_source):
    """Source shaped for numeric/EAV describe tests."""

    def _make(
        event_type: str = "prescriptions", **columns: pl.Series
    ) -> PolarsEDASource:
        n = len(next(iter(columns.values())))
        frame = {f"{event_type}/{name}": vals for name, vals in columns.items()}
        frame["event_type"] = pl.Series([event_type] * n, dtype=pl.String)
        frame["patient_id"] = pl.Series([str(i) for i in range(n)], dtype=pl.String)
        return make_source(**frame)

    return _make


@pytest.fixture
def mapping_csv(tmp_path) -> Callable:
    """Write a mapping DataFrame to a temp CSV and return its path."""

    def _write(data: dict, name: str = "mapping.csv") -> Path:
        path = tmp_path / name
        pl.LazyFrame(data).sink_csv(path)
        return path

    return _write


@pytest.fixture
def source_lazyframe() -> Callable:
    """Factory for a LazyFrame with overrides."""

    def _build(event_type: str, **columns: pl.Series) -> pl.LazyFrame:
        frame = {f"{event_type}/{name}": vals for name, vals in columns.items()}

        return pl.LazyFrame(frame)

    return _build
