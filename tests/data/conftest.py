"""Shared fixtures for the data-source test suite."""

import polars as pl
import pytest


@pytest.fixture
def events_df():
    """Factory for a valid PyHealth-shaped events frame with overrides.

    Returns a builder. Calling without args returns a valid default
    frame. Pass column=values or drop=[...] to modify.
    """

    def _build(drop: tuple[str, ...] = (), **overrides: list) -> pl.DataFrame:
        defaults = {
            "patient_id": ["1", "2", "2"],
            "event_type": ["patients", "patients", "admissions"],
        }
        defaults.update(overrides)
        for col in drop:
            defaults.pop(col, None)
        return pl.DataFrame(defaults)

    return _build
