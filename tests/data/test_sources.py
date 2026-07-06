"""Test suite for the EDASource class methods and helper functions."""

import datetime as dt

import polars as pl
import pytest
from polars.testing import assert_series_equal

from thesis.data.sources import cast_frame


@pytest.mark.parametrize(
    "data, schema_overrides, expected",
    [
        (
            {"col": pl.Series(["a", "b", "c"], dtype=pl.String)},
            {"col": "String"},
            pl.Series("col", ["a", "b", "c"], dtype=pl.String),
        ),
        (
            {"col": pl.Series(["1", "2", "3"], dtype=pl.String)},
            {"col": "UInt"},
            pl.Series("col", [1, 2, 3], dtype=pl.UInt16),
        ),
        (
            {"col": pl.Series(["1.1", "2.2", "3.3"], dtype=pl.String)},
            {"col": "Float"},
            pl.Series("col", [1.1, 2.2, 3.3], dtype=pl.Float32),
        ),
        (
            {"col": pl.Series(["1", "0", "0"], dtype=pl.String)},
            {"col": "Boolean"},
            pl.Series("col", [True, False, False], dtype=pl.Boolean),
        ),
        (
            {"col": pl.Series(["2025-01-01"], dtype=pl.String)},
            {"col": "Date"},
            pl.Series("col", [dt.date(2025, 1, 1)], pl.Date),
        ),
        (
            {
                "col": pl.Series(
                    [
                        "2025-01-01T12:15:00",
                    ],
                    dtype=pl.String,
                )
            },
            {"col": "DateTime"},
            pl.Series("col", [dt.datetime(2025, 1, 1, 12, 15, 0)], pl.Datetime),
        ),
    ],
)
def test_happy_path_conversions(
    data: dict[str, pl.Series], schema_overrides: dict[str, str], expected: pl.Series
) -> None:
    """Asserts that simple dtype castings work as intended."""
    lf = cast_frame(pl.LazyFrame(data), schema_overrides).collect()
    assert_series_equal(lf["col"], expected)
