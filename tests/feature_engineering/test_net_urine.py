"""Test suite for the net_urine helper function."""

import datetime
from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.feature_engineering.urine_output import net_urine


@pytest.mark.parametrize(
    "overrides, expected_lf_data",
    [
        # 0. Returns the subject and admission sum for each charttime
        (
            {
                "charttime": pl.Series(
                    [datetime.datetime(2025, 1, 1, 0)] * 6, dtype=pl.Datetime
                )
            },
            {
                "subject_id": pl.Series(["1"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
                "stay_id": pl.Series(["1"], dtype=pl.String),
                "charttime": pl.Series(
                    [datetime.datetime(2025, 1, 1, 0)], dtype=pl.Datetime
                ),
                "valuenum": pl.Series([600], dtype=pl.Float64),
            },
        ),
        # 1. Differentiates between subjects
        (
            {
                "subject_id": pl.Series(["1"] * 3 + ["2"] * 3, dtype=pl.String),
                "charttime": pl.Series(
                    [datetime.datetime(2025, 1, 1, 0)] * 6, dtype=pl.Datetime
                ),
            },
            {
                "subject_id": pl.Series(["1", "2"], dtype=pl.String),
                "hadm_id": pl.Series(["1"] * 2, dtype=pl.String),
                "stay_id": pl.Series(["1"] * 2, dtype=pl.String),
                "charttime": pl.Series(
                    [datetime.datetime(2025, 1, 1, 0)] * 2, dtype=pl.Datetime
                ),
                "valuenum": pl.Series([300, 300], dtype=pl.Float64),
            },
        ),
        # 2. Differentiates between admissions of the same subject
        (
            {
                "hadm_id": pl.Series(["1"] * 3 + ["2"] * 3, dtype=pl.String),
                "charttime": pl.Series(
                    [datetime.datetime(2025, 1, 1, 0)] * 6, dtype=pl.Datetime
                ),
            },
            {
                "subject_id": pl.Series(["1"] * 2, dtype=pl.String),
                "hadm_id": pl.Series(["1", "2"], dtype=pl.String),
                "stay_id": pl.Series(["1"] * 2, dtype=pl.String),
                "charttime": pl.Series(
                    [datetime.datetime(2025, 1, 1, 0)] * 2, dtype=pl.Datetime
                ),
                "valuenum": pl.Series([300, 300], dtype=pl.Float64),
            },
        ),
        # 3. Aggregation is at the timestamp level
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
                "valuenum": [100.0] * 6,
            },
        ),
        # 4. Null values get dropped, not replaced
        (
            {
                "valuenum": pl.Series([100] * 5 + [None], dtype=pl.Float64),
                "charttime": pl.Series(
                    [datetime.datetime(2025, 1, 1, 0)] * 5
                    + [datetime.datetime(2025, 1, 1, 1)],
                    dtype=pl.Datetime,
                ),
            },
            {
                "subject_id": pl.Series(["1"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
                "stay_id": pl.Series(["1"], dtype=pl.String),
                "charttime": pl.Series(
                    [datetime.datetime(2025, 1, 1, 0)], dtype=pl.Datetime
                ),
                "valuenum": pl.Series([500], dtype=pl.Float64),
            },
        ),
        # 5. Output is sorted in order of subject_id, hadm_id, and charttime
        (
            {
                "subject_id": pl.Series(
                    ["2", "1", "1", "1", "2", "1"], dtype=pl.String
                ),
                "hadm_id": pl.Series(["1", "2", "1", "1", "1", "2"], dtype=pl.String),
                "charttime": pl.Series(
                    [
                        datetime.datetime(2025, 1, 1, 0),
                        datetime.datetime(2025, 1, 1, 0),
                        datetime.datetime(2025, 1, 1, 1),
                        datetime.datetime(2025, 1, 1, 0),
                        datetime.datetime(2025, 1, 1, 1),
                        datetime.datetime(2025, 1, 1, 1),
                    ],
                    dtype=pl.Datetime,
                ),
                "valuenum": pl.Series([10, 20, 30, 40, 50, 60], dtype=pl.Float64),
            },
            {
                "subject_id": pl.Series(
                    ["1", "1", "1", "1", "2", "2"], dtype=pl.String
                ),
                "hadm_id": pl.Series(["1", "1", "2", "2", "1", "1"], dtype=pl.String),
                "stay_id": pl.Series(["1"] * 6, dtype=pl.String),
                "charttime": pl.Series(
                    [
                        datetime.datetime(2025, 1, 1, 0),
                        datetime.datetime(2025, 1, 1, 1),
                        datetime.datetime(2025, 1, 1, 0),
                        datetime.datetime(2025, 1, 1, 1),
                        datetime.datetime(2025, 1, 1, 0),
                        datetime.datetime(2025, 1, 1, 1),
                    ],
                    dtype=pl.Datetime,
                ),
                "valuenum": pl.Series([40, 30, 20, 60, 10, 50], dtype=pl.Float64),
            },
        ),
        # 6. Irrigant volume is subtracted from the total
        (
            {
                "itemid": pl.Series(["227488"] + ["226560"] * 5, dtype=pl.String),
                "valuenum": pl.Series([300] + [100] * 5, dtype=pl.Float64),
                "charttime": pl.Series(
                    [datetime.datetime(2025, 1, 1, 0)] * 6, dtype=pl.Datetime
                ),
            },
            {
                "subject_id": pl.Series(["1"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
                "stay_id": pl.Series(["1"], dtype=pl.String),
                "charttime": pl.Series(
                    [datetime.datetime(2025, 1, 1, 0)], dtype=pl.Datetime
                ),
                "valuenum": pl.Series([200], dtype=pl.Float64),
            },
        ),
        # 7. Irrigant volume gets signed reversed only if positive
        (
            {
                "itemid": pl.Series(["227488"] + ["226560"] * 5, dtype=pl.String),
                "valuenum": pl.Series([-300] + [100] * 5, dtype=pl.Float64),
                "charttime": pl.Series(
                    [datetime.datetime(2025, 1, 1, 0)] * 6, dtype=pl.Datetime
                ),
            },
            {
                "subject_id": pl.Series(["1"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
                "stay_id": pl.Series(["1"], dtype=pl.String),
                "charttime": pl.Series(
                    [datetime.datetime(2025, 1, 1, 0)], dtype=pl.Datetime
                ),
                "valuenum": pl.Series([200], dtype=pl.Float64),
            },
        ),
    ],
)
def test_net_urine_happy_path(
    outputevents_lf: Callable,
    overrides: dict[str, pl.Series],
    expected_lf_data: dict[str, pl.Series],
) -> None:
    """Asserts expected behaviour for net_urine helper."""
    source = outputevents_lf(**overrides)
    expected_lf = pl.LazyFrame(expected_lf_data)
    assert_frame_equal(net_urine(source), expected_lf)


@pytest.mark.parametrize(
    "overrides",
    [
        # 0. charttime isn't datetime
        {"charttime": pl.Series(["2025-01-01 00:00:00"] * 6, dtype=pl.String)},
        # 1. valuenum isn't numeric
        {"valuenum": pl.Series(["1"] * 6, dtype=pl.String)},
    ],
)
def test_net_urine_raises_if_wrong_dtype(
    outputevents_lf: Callable, overrides: dict[str, pl.Series]
) -> None:
    """Asserts guard behaviour against wrong dtypes."""
    with pytest.raises(ValueError):
        source = outputevents_lf(**overrides)
        net_urine(source)
