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
                "charttime": [
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
                "charttime": [datetime.datetime(2025, 1, 1, 0)] * 6,
            },
            {
                "subject_id": ["1"] * 3 + ["2"] * 3,
                "hadm_id": ["1"] * 6,
                "stay_id": ["1"] * 6,
                "charttime": [datetime.datetime(2025, 1, 1, 0)] * 6,
                "valuenum": [90.627756] * 6,
            },
        ),
        # 2. Output is subsequently sorted by hadm_id
        (
            {
                "subject_id": ["2"] * 3 + ["1"] * 3,
                "hadm_id": ["3", "2", "1"] * 2,
                "charttime": [datetime.datetime(2025, 1, 1, 0)] * 6,
            },
            {
                "subject_id": ["1"] * 3 + ["2"] * 3,
                "hadm_id": ["1", "2", "3"] * 2,
                "stay_id": ["1"] * 6,
                "charttime": [datetime.datetime(2025, 1, 1, 0)] * 6,
                "valuenum": [90.627756] * 6,
            },
        ),
        # 3. Output is finally sorted by datetime
        (
            {
                "subject_id": ["2"] * 3 + ["1"] * 3,
                "charttime": [
                    datetime.datetime(2025, 1, 1, 4),
                    datetime.datetime(2025, 1, 1, 0),
                    datetime.datetime(2025, 1, 1, 8),
                ]
                * 2,
            },
            {
                "subject_id": ["1"] * 3 + ["2"] * 3,
                "hadm_id": ["1"] * 6,
                "stay_id": ["1"] * 6,
                "charttime": [
                    datetime.datetime(2025, 1, 1, 0),
                    datetime.datetime(2025, 1, 1, 4),
                    datetime.datetime(2025, 1, 1, 8),
                ]
                * 2,
                "valuenum": [90.627756] * 6,
            },
        ),
        # 4. Weight not converted unless specific itemid
        (
            {"itemid": ["224639"] * 6},
            {
                "subject_id": ["1"] * 6,
                "hadm_id": ["1"] * 6,
                "stay_id": ["1"] * 6,
                "charttime": [
                    datetime.datetime(2025, 1, 1, 0) + datetime.timedelta(hours=i)
                    for i in range(6)
                ],
                "valuenum": [199.8] + [90.627756] * 5,
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


@pytest.mark.parametrize(
    "drop",
    [
        # 0. itemid
        ["itemid"],
        # 1. hadm_id
        ["hadm_id"],
        # 2. subject_id
        ["subject_id"],
        # 3. charrtime
        ["charttime"],
        # 4. valuenum
        ["valuenum"],
    ],
)
def test_normalize_weights_raises_if_missing_col(
    chartevents_lf: Callable, drop: list[str]
) -> None:
    """Asserts guard behaviour against missing core cols."""
    with pytest.raises(KeyError, match=f"'{drop[0]}' is missing."):
        source = chartevents_lf(drop)
        normalize_weights(source)


@pytest.mark.parametrize(
    "overrides",
    [
        # 0. charrtime isn't datetime
        {"charttime": pl.Series(["2025-01-01 00:00:00"] * 6, dtype=pl.String)},
        # 1. valuenum isn't numeric
        {"valuenum": pl.Series(["1"] * 6, dtype=pl.String)},
    ],
)
def test_normalize_weights_raises_if_wrong_dtype(
    chartevents_lf: Callable, overrides: dict[str, pl.Series]
) -> None:
    """Asserts guard behaviour against wrong dtypes."""
    with pytest.raises(ValueError):
        source = chartevents_lf(**overrides)
        normalize_weights(source)
