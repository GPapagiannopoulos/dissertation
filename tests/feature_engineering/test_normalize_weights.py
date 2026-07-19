"""Testing suite for the normalize_weights helper function."""

import datetime
from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.feature_engineering.urine_output import normalize_weights


@pytest.mark.parametrize(
    "overrides, expected_lf_data",
    [
        # 0. Correctly converts lbs to kg
        (
            {},
            {
                "subject_id": ["1"] * 6,
                "hadm_id": ["1"] * 6,
                "stay_id": ["1"] * 6,
                "charttiime": [
                    datetime.datetime(2025, 1, 1, 0) + datetime.timedelta(hours=i)
                    for i in range(6)
                ],
                "valuenum": [90.627756] * 6,
            },
        ),
        # 1. Result is ordered by subject id
        (
            {
                "subject_id": ["2"] * 3 + ["1"] * 3,
                "charttiime": [datetime.datetime(2025, 1, 1, 0)] * 6,
            },
            {
                "subject_id": ["1"] * 3 + ["2"] * 3,
                "hadm_id": ["1"] * 6,
                "stay_id": ["1"] * 6,
                "charttiime": [datetime.datetime(2025, 1, 1, 0)] * 6,
                "valuenum": [90.627756] * 6,
            },
        ),
        # 2. Output is subsequently sorted by hadm_id
        (
            {
                "subject_id": ["2"] * 3 + ["1"] * 3,
                "hadm_id": ["3", "2", "1"] * 2,
                "charttiime": [datetime.datetime(2025, 1, 1, 0)] * 6,
            },
            {
                "subject_id": ["1"] * 3 + ["2"] * 3,
                "hadm_id": ["1", "2", "3"] * 2,
                "stay_id": ["1"] * 6,
                "charttiime": [datetime.datetime(2025, 1, 1, 0)] * 6,
                "valuenum": [90.627756] * 6,
            },
        ),
    ],
)
def test_normalize_weights_happy_path(
    chartevents_lf: Callable,
    overrides: dict[str, pl.Series],
    expected_lf_data: dict[str, pl.Series],
) -> None:
    """Asserts expected behaviour for the helper function."""
    source = chartevents_lf(**overrides)
    source = normalize_weights(source)
    expected = pl.LazyFrame(expected_lf_data)
    assert_frame_equal(source, expected)
