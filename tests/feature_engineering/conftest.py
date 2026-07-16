"""Shared fixtures for the feature engineering test suite."""

import datetime

import polars as pl
import pytest


@pytest.fixture
def labevents_lf():
    """Factory for creating a valid LazyFrame with overrides."""

    def _build(
        drop: tuple[str, ...] = (), hour_step: int = 12, **overrides: list
    ) -> pl.LazyFrame:
        defaults = {
            "patient_id": ["1"] * 14,
            "hadm_id": ["1"] * 14,
            "event_type": ["labevents"] * 14,
            "timestamp": [
                (
                    datetime.datetime(
                        2025,
                        1,
                        1,
                        0,
                    )
                    + datetime.timedelta(hours=i * hour_step)
                )
                for i in range(14)
            ],
            "labevents/label": ["Creatinine"] * 14,
            "labevents/valuenum": [1.25] * 14,
            "labevents/valueuom": ["mg/dL"] * 14,
            "admissions/admittime": [datetime.datetime(2025, 1, 1, 0)] + [None] * 13,
        }
        defaults.update(overrides)
        for col in drop:
            defaults.pop(col, None)
        return pl.LazyFrame(defaults)

    return _build
