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
            "admissions/admittime": [datetime.datetime(2025, 1, 1, 0)] * 14,
        }
        defaults.update(overrides)
        for col in drop:
            defaults.pop(col, None)
        return pl.LazyFrame(defaults)

    return _build


@pytest.fixture
def chartevents_lf():
    """Factory for creating a valid chartevents LazyFrame with overrides."""

    def _build(**overrides: dict[str, list]) -> pl.LazyFrame:
        defaults = {
            "subject_id": ["1"] * 6,
            "hadm_id": ["1"] * 6,
            "stay_id": ["1"] * 6,
            "charttiime": [
                datetime.datetime(2025, 1, 1, 0) + datetime.timedelta(hours=i)
                for i in range(6)
            ],
            "itemid": [226531] + [224639] * 5,
            "valuenum": [199.8] + [90.627756] * 5,
            "valueuom": [""] + ["kg"] * 5,
        }
        defaults.update(**overrides)
        return pl.LazyFrame(defaults)

    return _build
