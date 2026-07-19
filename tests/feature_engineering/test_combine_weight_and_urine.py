"""Test suite for the _combine_weight_and_urine helper function."""

import datetime
from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.feature_engineering.urine_output import _combine_weight_and_urine


def _at(hour: int) -> datetime.datetime:
    return datetime.datetime(2025, 1, 1, hour)


@pytest.mark.parametrize(
    "weight_overrides, urine_overrides, expected_data",
    [
        # 0. Weight is fed forward: each urine takes the most recent prior weight
        (
            {
                "subject_id": ["1", "1"],
                "hadm_id": ["1", "1"],
                "stay_id": ["1", "1"],
                "charttime": [_at(0), _at(2)],
                "valuenum": [60.0, 62.0],
            },
            {},
            {
                "subject_id": pl.Series(["1"] * 4, dtype=pl.String),
                "hadm_id": pl.Series(["1"] * 4, dtype=pl.String),
                "stay_id": pl.Series(["1"] * 4, dtype=pl.String),
                "charttime": pl.Series(
                    [_at(0), _at(1), _at(2), _at(3)], dtype=pl.Datetime
                ),
                "valuenum": pl.Series([30.0] * 4, dtype=pl.Float64),
                "weight": pl.Series([60.0, 60.0, 62.0, 62.0], dtype=pl.Float64),
            },
        ),
        # 1. Urine before the first weight is backfilled with that first weight
        (
            {
                "charttime": [_at(1)],
                "valuenum": [60.0],
            },
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
                "valuenum": pl.Series([30.0] * 3, dtype=pl.Float64),
                "weight": pl.Series([60.0, 60.0, 60.0], dtype=pl.Float64),
            },
        ),
        # 2. An admission with no weight stays null
        # (weight does not leak across admissions)
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
                "valuenum": pl.Series([30.0, 30.0], dtype=pl.Float64),
                "weight": pl.Series([None, None], dtype=pl.Float64),
            },
        ),
        # 3. Weight is scoped to the admission, so it feeds across stays within it
        (
            {},
            {
                "subject_id": ["1", "1"],
                "hadm_id": ["1", "1"],
                "stay_id": ["2", "2"],
                "charttime": [_at(1), _at(2)],
                "valuenum": [30.0, 30.0],
            },
            {
                "subject_id": pl.Series(["1", "1"], dtype=pl.String),
                "hadm_id": pl.Series(["1", "1"], dtype=pl.String),
                "stay_id": pl.Series(["2", "2"], dtype=pl.String),
                "charttime": pl.Series([_at(1), _at(2)], dtype=pl.Datetime),
                "valuenum": pl.Series([30.0, 30.0], dtype=pl.Float64),
                "weight": pl.Series([60.0, 60.0], dtype=pl.Float64),
            },
        ),
        # 4. Output is sorted by subject, admission, and charttime
        (
            {},
            {
                "subject_id": ["1"] * 3,
                "hadm_id": ["1"] * 3,
                "stay_id": ["1"] * 3,
                "charttime": [_at(2), _at(0), _at(1)],
                "valuenum": [3.0, 1.0, 2.0],
            },
            {
                "subject_id": pl.Series(["1"] * 3, dtype=pl.String),
                "hadm_id": pl.Series(["1"] * 3, dtype=pl.String),
                "stay_id": pl.Series(["1"] * 3, dtype=pl.String),
                "charttime": pl.Series([_at(0), _at(1), _at(2)], dtype=pl.Datetime),
                "valuenum": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
                "weight": pl.Series([60.0, 60.0, 60.0], dtype=pl.Float64),
            },
        ),
    ],
)
def test_combine_weight_and_urine_happy_path(
    weight_frame: Callable,
    net_urine_frame: Callable,
    weight_overrides: dict[str, list],
    urine_overrides: dict[str, list],
    expected_data: dict[str, pl.Series],
) -> None:
    """Asserts expected behaviour for the _combine_weight_and_urine helper."""
    weight = weight_frame(**weight_overrides)
    urine = net_urine_frame(**urine_overrides)
    expected = pl.LazyFrame(expected_data)
    assert_frame_equal(_combine_weight_and_urine(weight, urine), expected)
