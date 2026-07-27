"""Testing suite for the code_inventory helper."""

import polars as pl
from polars.testing import assert_frame_equal

from thesis.modelling.etl_pipeline.coverage import code_inventory


def test_code_inventory_groups_by_code() -> None:
    """Asserts standard intended behaviour for code_inventory."""
    data = {
        "subject_id": pl.Series(["1", "2", "3"], dtype=pl.String),
        "code": pl.Series(["a", "b", "c"], dtype=pl.String),
    }
    expected = {
        "code": pl.Series(["a", "b", "c"], dtype=pl.String),
        "count": pl.Series([1] * 3, dtype=pl.UInt32),
    }
    data_lf = pl.LazyFrame(data)
    assert_frame_equal(code_inventory(data_lf), pl.LazyFrame(expected))
