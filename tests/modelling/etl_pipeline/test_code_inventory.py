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
        )
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
