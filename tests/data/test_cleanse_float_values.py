"""Testing suite for the cleanse_float_values helper function."""

import polars as pl
from polars.testing import assert_frame_equal

from thesis.data.sources import cleanse_float_values


def test_commas_removed_from_string() -> None:
    """Asserts that the function removes commas in the body of the string."""
    data = {"col": pl.Series(["13,000"])}
    df = cleanse_float_values(pl.DataFrame(data), "col")
    expected = pl.DataFrame({"col": pl.Series(["13000"], dtype=pl.String)})
    assert_frame_equal(df, expected)


def test_all_commas_removed_from_string() -> None:
    """Asserts that the function removes all commas in the body of the string."""
    data = {"col": pl.Series(["13,000,000"])}
    df = cleanse_float_values(pl.DataFrame(data), "col")
    expected = pl.DataFrame({"col": pl.Series(["13000000"], dtype=pl.String)})
    assert_frame_equal(df, expected)
