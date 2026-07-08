"""Testing suite for the cleanse_float_values helper function."""

import polars as pl
from polars.testing import assert_frame_equal

from thesis.data.sources import cleanse_float_values


def test_commas_removed_from_string() -> None:
    """Asserts that the function removes commas in the body of the string."""
    data = {"col": pl.Series(["13,000", "15,000"])}
    df = cleanse_float_values(pl.DataFrame(data), ["col"])
    expected = pl.DataFrame({"col": pl.Series(["13000.0", "15000.0"], dtype=pl.String)})
    assert_frame_equal(df, expected)


def test_all_commas_removed_from_string() -> None:
    """Asserts that the function removes all commas in the body of the string."""
    data = {"col": pl.Series(["13,000,000"])}
    df = cleanse_float_values(pl.DataFrame(data), ["col"])
    expected = pl.DataFrame({"col": pl.Series(["13000000.0"], dtype=pl.String)})
    assert_frame_equal(df, expected)


def test_hyphenated_entries_replaced_with_mean() -> None:
    """Asserts that the function replaces hyphenated records with mean."""
    data = {"col": pl.Series(["1-3"], dtype=pl.String)}
    df = cleanse_float_values(pl.DataFrame(data), ["col"])
    expected = pl.DataFrame({"col": pl.Series(["2.0"], dtype=pl.String)})
    assert_frame_equal(df, expected)


def test_already_castable_strings_are_unaffected() -> None:
    """Asserts that non-hyphenated, non-comma containing values are not cast."""
    data = {"col": pl.Series(["13,000", "1-6", "2"], dtype=pl.String)}
    df = cleanse_float_values(pl.DataFrame(data), ["col"])
    expected = pl.DataFrame({"col": pl.Series(["13000.0", "3.5", "2.0"])})
    assert_frame_equal(df, expected)


def test_function_affects_multiple_cols() -> None:
    """Asserts that all specified fields are cleansed."""
    data = {
        "col_a": pl.Series(["1-3"], dtype=pl.String),
        "col_b": pl.Series(["2-6"], dtype=pl.String),
    }
    df = cleanse_float_values(pl.DataFrame(data), ["col_a", "col_b"])
    expected = pl.DataFrame(
        {
            "col_a": pl.Series(["2.0"], dtype=pl.String),
            "col_b": pl.Series(["4.0"], dtype=pl.String),
        }
    )
    assert_frame_equal(df, expected)
