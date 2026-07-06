"""Test suite for the EDASource class methods and helper functions."""

import datetime as dt

import polars as pl
import pytest
from polars.exceptions import InvalidOperationError
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


def test_cast_frame_without_fields_raises():
    """Asserts that a lazyframe without any fields raises an error.

    Attempting to cast a LazyFrame of shape 0 x N is valid. However,
    attempting to cast a LazyFrame of shape 0 x 0 is not a logical
    operation and needs to raise a ValueError.

    Returns:
        None

    Raises:
        AssertionError: if the cast operation doesn't raise a ValueError
    """
    with pytest.raises(ValueError, match="LazyFrame contains no fields"):
        lf = pl.LazyFrame()
        lf = cast_frame(lf, {"col": "UInt"})


def test_cast_frame_with_empty_overrides_raises():
    """Asserts that cast_frame will raise an error if the map is empty.

    Attempting to cast with an empty mapping is definitely an error.

    Returns:
        None

    Raises:
        AssertionError: if the cast operation doesn't raise a ValueError
    """
    with pytest.raises(ValueError, match="Map contains no key:value pairs"):
        lf = pl.LazyFrame({"col": pl.Series(["a"], dtype=pl.String)})
        lf = cast_frame(lf, {})


def test_cast_frame_empty_lf_raises_first():
    """Asserts that an empty LazyFrame raises an error before an empty map."""
    with pytest.raises(ValueError, match="LazyFrame contains no fields"):
        lf = pl.LazyFrame()
        lf = cast_frame(lf, {})


def test_cast_frame_raises_if_col_skipped():
    """Asserts that cast_frame will raise if it doesn't find a col in the lazyframe.

    The function iterates over the keys in the map to find the fields it needs to
    cast. If one of the keys doesn't exist, it is likely because of programmer error.

    Returns:
        None

    Raises:
        AssertionError: if the cast operation doesn't raise a ValueError
    """
    with pytest.raises(ValueError, match="wrong_col is not an available field"):
        lf = pl.LazyFrame({"col": pl.Series([])})
        map = {"wrong_col": "UInt"}
        lf = cast_frame(lf, map)


@pytest.mark.parametrize(
    "data, dtype_map",
    [
        # 0. Unparseable str->int conversion
        ({"col": pl.Series(["1", "abc", "3"], dtype=pl.String)}, {"col": "UInt"}),
        # 1. Narrowing conversion from float to int
        ({"col": pl.Series(["1.0", "2.5", "3.9"], dtype=pl.String)}, {"col": "UInt"}),
        # 2. Negative values to unsigned
        ({"col": pl.Series(["-1.0", "-2.5", "3.9"], dtype=pl.String)}, {"col": "UInt"}),
        # 3. Impossible date
        ({"col": pl.Series(["2025-42-99"], dtype=pl.String)}, {"col": "Date"}),
    ],
)
def test_cast_frame_raises_on_unconvertible_values(
    data: dict[str, pl.Series], dtype_map: dict[str, str]
):
    """Asserts that cast_frame raises an error if the conversion is invalid.

    MIMIC-IV is a high quality dataset that should not suffer from excessive
    data quality errors. Failed casts due to unparseable data are either outliers
    or programmer errors. Regardless they require the programmer's attention.

    Args:
        data (dict[str, pl.Series]): mock data to be converted
        dtype_map (dict[str, str]): mapping of mock data cols to dtype strings

    Returns:
        None

    Raises:
        AssertionError: if collection doesn't raise an InvalidOperationError
    """
    lf = pl.LazyFrame(data)
    lf = cast_frame(lf, dtype_map)
    with pytest.raises(InvalidOperationError):
        lf.collect()


def test_cast_frame_raises_if_dtype_unrecognized():
    """Asserts that the mapping is to a valid string.

    To decouple the exploration from the implementation details, mapping uses
    a generic str that matches a polars dtype. The valid strings are found
    under constants and offer a single point of change if the underlying
    dtype needs to change. If the user passes an invalid string, it should
    raise.

    Returns:
        None

    Raises:
        AssertionError: if the cast doesn't raise a KeyError
    """
    with pytest.raises(KeyError, match="integer is not a valid key"):
        lf = pl.LazyFrame({"col": pl.Series(["1", "2", "3"], dtype=pl.String)})
        map = {"col": "integer"}
        lf = cast_frame(lf, map)
