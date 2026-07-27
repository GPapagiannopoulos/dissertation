"""Testing suite for the code_inventory helper."""

from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.modelling.etl_pipeline.coverage import code_inventory


@pytest.mark.parametrize(
    "drops, overrides, expected",
    [
        # 0. Standard happy path
        (
            [],
            {},
            {
                "code": pl.Series(["a", "b", "c"], dtype=pl.String),
                "count": pl.Series([1] * 3, dtype=pl.UInt32),
            },
        ),
        # 1. Correctly counts duplicates
        (
            [],
            {"code": pl.Series(["a"] * 3, dtype=pl.String)},
            {
                "code": pl.Series(["a"], dtype=pl.String),
                "count": pl.Series([3], dtype=pl.UInt32),
            },
        ),
        # 2. 'Null' codes get their own rows
        (
            [],
            {"code": pl.Series(["a"] * 2 + [None], dtype=pl.String)},
            {
                "code": pl.Series(["a", None], dtype=pl.String),
                "count": pl.Series([2, 1], dtype=pl.UInt32),
            },
        ),
        # 3. An empty LF returns an empty LF
        (
            [],
            {"code": pl.Series([]), "subject_id": pl.Series([])},
            {"code": pl.Series([]), "count": pl.Series([], dtype=pl.UInt32)},
        ),
    ],
)
def test_code_inventory_happy_path(
    meds_outputs: Callable,
    drops: list[str],
    overrides: dict[str, pl.Series],
    expected: dict[str, pl.Series],
) -> None:
    """Asserts standard intended behaviour for code_inventory."""
    data_lf = pl.LazyFrame(meds_outputs(drops, **overrides))
    assert_frame_equal(code_inventory(data_lf), pl.LazyFrame(expected))
