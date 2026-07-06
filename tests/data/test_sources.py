"""Test suite for the EDASource class methods and helper functions."""

import datetime as dt

import polars as pl
import pytest
from polars.testing import assert_series_equal

from thesis.data.sources import cast_frame


@pytest.mark.parametrize(
    "data, schema_overrides, expected",
    [
        # 0. String to string casts are skipped
        (
            {"col": pl.Series(["a", "b", "c"], dtype=pl.String)},
            {"col": "String"},
            pl.Series("col", ["a", "b", "c"], dtype=pl.String),
        ),
        # 1. UInt casts to unsigned int (16bit)
        (
            {"col": pl.Series(["1", "2", "3"], dtype=pl.String)},
            {"col": "UInt"},
            pl.Series("col", [1, 2, 3], dtype=pl.UInt16),
        ),
        # 2. Float casts to Float (32bit)
        (
            {"col": pl.Series(["1.1", "2.2", "3.3"], dtype=pl.String)},
            {"col": "Float"},
            pl.Series("col", [1.1, 2.2, 3.3], dtype=pl.Float32),
        ),
        # 3. Boolean casts to appropriate boolean values
        (
            {"col": pl.Series(["1", "0", "0"], dtype=pl.String)},
            {"col": "Boolean"},
            pl.Series("col", [True, False, False], dtype=pl.Boolean),
        ),
        # 4. Date casts to Date objects
        (
            {"col": pl.Series(["2025-01-01"], dtype=pl.String)},
            {"col": "Date"},
            pl.Series("col", [dt.date(2025, 1, 1)], pl.Date),
        ),
        # 5. DateTime objects gets parsed and cast correctly
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
        # Empty string values in the csv get converted to None values via PyHealth
        # 6. None values are skipped, non-null values get cast
        (
            {"col": pl.Series(["1", None, "2"], dtype=pl.String)},
            {"col": "UInt"},
            pl.Series("col", [1, None, 2], dtype=pl.UInt16),
        ),
        # 7. None only fields loaded as a pl.String col cast correctly
        (
            {"col": pl.Series([None, None, None], dtype=pl.String)},
            {"col": "UInt"},
            pl.Series("col", [None, None, None], dtype=pl.UInt16),
        ),
        # 8. None only fields loaded as a pl.Null col cast correctly
        (
            {"col": pl.Series([None, None, None], dtype=pl.Null)},
            {"col": "UInt"},
            pl.Series("col", [None, None, None], dtype=pl.UInt16),
        ),
        # 9.
        (
            {"col": pl.Series([], dtype=pl.String)},
            {"col": "UInt"},
            pl.Series("col", [], dtype=pl.UInt16),
        ),
    ],
)
def test_cast_frame_happy_path_conversions(
    data: dict[str, pl.Series], schema_overrides: dict[str, str], expected: pl.Series
) -> None:
    """Asserts that simple dtype castings work as intended.

    Args:
        data (dict[str, pl.Series]): Mock data representing a column in a polars lf.
        schema_overrides (dict[str, str]): A col_name:dtype dictionary representing
        the casts.
        expected (pl.Series): A series representing the expected column after the cast

    Returns:
        None

    Raises:
        AssertionError: if the cast fails
    """
    lf = cast_frame(pl.LazyFrame(data), schema_overrides).collect()
    assert_series_equal(lf["col"], expected)
