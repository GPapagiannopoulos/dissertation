"""Test suite for the calculate_urine_output_rate helper function."""

import datetime
from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.feature_engineering.urine_output import calculate_urine_output_rate


def _at(hour: int) -> datetime.datetime:
    return datetime.datetime(2025, 1, 1, hour)


@pytest.mark.parametrize(
    "weight_overrides, urine_overrides, expected_data",
    [
        # 0. Rate is the windowed volume over the observed span and the weight
        (
            {},
            {
                "subject_id": ["1"] * 3,
                "hadm_id": ["1"] * 3,
                "stay_id": ["1"] * 3,
                "charttime": [_at(0), _at(1), _at(2)],
                "valuenum": [30.0] * 3,
            },
            {
                "subject_id": pl.Series(["1"] * 3, dtype=pl.String),
                "hadm_id": pl.Series(["1"] * 3, dtype=pl.String),
                "stay_id": pl.Series(["1"] * 3, dtype=pl.String),
                "charttime": pl.Series([_at(0), _at(1), _at(2)], dtype=pl.Datetime),
                "rate": pl.Series([None, 60 / 60 / 1, 90 / 60 / 2], dtype=pl.Float64),
                "window_hours": pl.Series([0.0, 1.0, 2.0], dtype=pl.Float64),
                "n_events": pl.Series([1, 2, 3], dtype=pl.UInt32),
            },
        ),
        # 1. Rate is null when the admission has no weight
        (
            {},
            {
                "subject_id": ["1", "1"],
                "hadm_id": ["2", "2"],
                "stay_id": ["2", "2"],
                "charttime": [_at(0), _at(1)],
                "valuenum": [30.0, 30.0],
            },
            {
                "subject_id": pl.Series(["1", "1"], dtype=pl.String),
                "hadm_id": pl.Series(["2", "2"], dtype=pl.String),
                "stay_id": pl.Series(["2", "2"], dtype=pl.String),
                "charttime": pl.Series([_at(0), _at(1)], dtype=pl.Datetime),
                "rate": pl.Series([None, None], dtype=pl.Float64),
                "window_hours": pl.Series([0.0, 1.0], dtype=pl.Float64),
                "n_events": pl.Series([1, 2], dtype=pl.UInt32),
            },
        ),
        # 2. The window does not bleed across stays of the same admission
        (
            {},
            {
                "subject_id": ["1"] * 4,
                "hadm_id": ["1"] * 4,
                "stay_id": ["1", "1", "2", "2"],
                "charttime": [_at(0), _at(1), _at(2), _at(3)],
                "valuenum": [30.0, 30.0, 100.0, 100.0],
            },
            {
                "subject_id": pl.Series(["1"] * 4, dtype=pl.String),
                "hadm_id": pl.Series(["1"] * 4, dtype=pl.String),
                "stay_id": pl.Series(["1", "1", "2", "2"], dtype=pl.String),
                "charttime": pl.Series(
                    [_at(0), _at(1), _at(2), _at(3)], dtype=pl.Datetime
                ),
                "rate": pl.Series(
                    [None, 60 / 60 / 1, None, 200 / 60 / 1], dtype=pl.Float64
                ),
                "window_hours": pl.Series([0.0, 1.0, 0.0, 1.0], dtype=pl.Float64),
                "n_events": pl.Series([1, 2, 1, 2], dtype=pl.UInt32),
            },
        ),
        # 3. A negative windowed volume yields a negative rate (irrigation artifact)
        (
            {},
            {
                "subject_id": ["1", "1"],
                "hadm_id": ["1", "1"],
                "stay_id": ["1", "1"],
                "charttime": [_at(0), _at(1)],
                "valuenum": [30.0, -100.0],
            },
            {
                "subject_id": pl.Series(["1", "1"], dtype=pl.String),
                "hadm_id": pl.Series(["1", "1"], dtype=pl.String),
                "stay_id": pl.Series(["1", "1"], dtype=pl.String),
                "charttime": pl.Series([_at(0), _at(1)], dtype=pl.Datetime),
                "rate": pl.Series([None, -70 / 60 / 1], dtype=pl.Float64),
                "window_hours": pl.Series([0.0, 1.0], dtype=pl.Float64),
                "n_events": pl.Series([1, 2], dtype=pl.UInt32),
            },
        ),
        # 4. Measurements older than the window drop out of the trailing sum
        (
            {},
            {
                "subject_id": ["1"] * 3,
                "hadm_id": ["1"] * 3,
                "stay_id": ["1"] * 3,
                "charttime": [_at(0), _at(3), _at(7)],
                "valuenum": [1000.0, 30.0, 30.0],
            },
            {
                "subject_id": pl.Series(["1"] * 3, dtype=pl.String),
                "hadm_id": pl.Series(["1"] * 3, dtype=pl.String),
                "stay_id": pl.Series(["1"] * 3, dtype=pl.String),
                "charttime": pl.Series([_at(0), _at(3), _at(7)], dtype=pl.Datetime),
                "rate": pl.Series([None, 1030 / 60 / 3, 60 / 60 / 4], dtype=pl.Float64),
                "window_hours": pl.Series([0.0, 3.0, 4.0], dtype=pl.Float64),
                "n_events": pl.Series([1, 2, 2], dtype=pl.UInt32),
            },
        ),
    ],
)
def test_calculate_urine_output_rate_happy_path(
    weight_frame: Callable,
    net_urine_frame: Callable,
    weight_overrides: dict[str, list],
    urine_overrides: dict[str, list],
    expected_data: dict[str, pl.Series],
) -> None:
    """Asserts expected behaviour for the calculate_urine_output_rate helper."""
    weight = weight_frame(**weight_overrides)
    urine = net_urine_frame(**urine_overrides)
    expected = pl.LazyFrame(expected_data)
    assert_frame_equal(calculate_urine_output_rate(weight, urine), expected)
