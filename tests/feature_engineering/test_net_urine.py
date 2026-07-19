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
                "charrtime": pl.Series(
                    [datetime.datetime(2025, 1, 1, 0)] * 6, dtype=pl.Datetime
                )
            },
            {
                "subject_id": pl.Series(["1"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
                "charttime": pl.Series(
                    [datetime.datetime(2025, 1, 1, 0)], dtype=pl.Datetime
                ),
                "valuenum": pl.Series([600], dtype=pl.Float64),
            },
        )
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
