"""Testing suite for the cleanse_float_values helper function."""

import polars as pl
import pytest
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal

from thesis.data.sources import cleanse_float_values


def test_commas_removed_from_string() -> None:
    """Asserts that the function removes commas in the body of the string."""
    data = {"col": pl.Series(["13,000", "15,000"])}
    df = cleanse_float_values(pl.LazyFrame(data), ["col"])
    expected = pl.LazyFrame({"col": pl.Series(["13000.0", "15000.0"], dtype=pl.String)})
    assert_frame_equal(df, expected)


def test_all_commas_removed_from_string() -> None:
    """Asserts that the function removes all commas in the body of the string."""
    data = {"col": pl.Series(["13,000,000"])}
    df = cleanse_float_values(pl.LazyFrame(data), ["col"])
    expected = pl.LazyFrame({"col": pl.Series(["13000000.0"], dtype=pl.String)})
    assert_frame_equal(df, expected)


def test_hyphenated_entries_replaced_with_mean() -> None:
    """Asserts that the function replaces hyphenated records with mean."""
    data = {"col": pl.Series(["1-3"], dtype=pl.String)}
    df = cleanse_float_values(pl.LazyFrame(data), ["col"])
    expected = pl.LazyFrame({"col": pl.Series(["2.0"], dtype=pl.String)})
    assert_frame_equal(df, expected)


def test_already_castable_strings_are_unaffected() -> None:
    """Asserts that non-hyphenated, non-comma containing values are not cast."""
    data = {"col": pl.Series(["13,000", "1-6", "2"], dtype=pl.String)}
    df = cleanse_float_values(pl.LazyFrame(data), ["col"])
    expected = pl.LazyFrame({"col": pl.Series(["13000.0", "3.5", "2.0"])})
    assert_frame_equal(df, expected)


def test_function_affects_multiple_cols() -> None:
    """Asserts that all specified fields are cleansed."""
    data = {
        "col_a": pl.Series(["1-3"], dtype=pl.String),
        "col_b": pl.Series(["2-6"], dtype=pl.String),
    }
    df = cleanse_float_values(pl.LazyFrame(data), ["col_a", "col_b"])
    expected = pl.LazyFrame(
        {
            "col_a": pl.Series(["2.0"], dtype=pl.String),
            "col_b": pl.Series(["4.0"], dtype=pl.String),
        }
    )
    assert_frame_equal(df, expected)


def test_function_only_affects_target_cols() -> None:
    """Asserts that non-specified fields are not modified."""
    data = {
        "col_a": pl.Series(["1-3"], dtype=pl.String),
        "col_b": pl.Series(["2-6"], dtype=pl.String),
    }
    df = cleanse_float_values(pl.LazyFrame(data), ["col_a"])
    expected = pl.LazyFrame(
        {
            "col_a": pl.Series(["2.0"], dtype=pl.String),
            "col_b": pl.Series(["2-6"], dtype=pl.String),
        }
    )
    assert_frame_equal(df, expected)


def test_function_raises_if_col_not_str() -> None:
    """Asserts that an error is raised if the dtype isn't string."""
    data = {"col": pl.Series([1, 2, 3], dtype=pl.Int16)}
    with pytest.raises(InvalidOperationError, match="expected String type"):
        cleanse_float_values(pl.LazyFrame(data), ["col"])


def test_function_raises_if_any_of_target_fields_not_strings() -> None:
    """Asserts that an error is raised if any of the fields are not strings."""
    data = {
        "col_a": pl.Series([1, 2, 3], dtype=pl.Int16),
        "col_b": pl.Series(["1", "2", "3"], dtype=pl.String),
    }
    with pytest.raises(InvalidOperationError, match="expected String type"):
        cleanse_float_values(pl.LazyFrame(data), ["col_a", "col_b"])


def test_function_no_raise_if_non_string_field_not_passed() -> None:
    """Asserts that an error is only raised if the non-string fields are targeted."""
    data = {
        "col_a": pl.Series([1, 2, 3], dtype=pl.Int16),
        "col_b": pl.Series(["1", "2", "3"], dtype=pl.String),
    }
    df = cleanse_float_values(pl.LazyFrame(data), ["col_b"])
    expected = pl.LazyFrame(
        {
            "col_a": pl.Series([1, 2, 3], dtype=pl.Int16),
            "col_b": pl.Series(["1.0", "2.0", "3.0"], dtype=pl.String),
        }
    )

    assert_frame_equal(df, expected)


def test_function_discerns_sign_vs_range() -> None:
    """Asserts that the function only removes hyphens in the body of the string."""
    data = {"col": pl.Series(["-2", "1-3"], dtype=pl.String)}
    expected_df = pl.LazyFrame({"col": pl.Series(["-2.0", "2.0"], dtype=pl.String)})
    df = cleanse_float_values(pl.LazyFrame(data), ["col"])
    assert_frame_equal(df, expected_df)


@pytest.mark.parametrize(
    "data, expected",
    [
        # 0. Single whitespace replaced with 'None'
        (
            {"col": pl.Series([" "], dtype=pl.String)},
            {"col": pl.Series([None], dtype=pl.String)},
        ),
        # 1. None values are unaffected
        (
            {"col": pl.Series([None], dtype=pl.String)},
            {"col": pl.Series([None], dtype=pl.String)},
        ),
        # 2. Empty strings get converted to 'None'
        (
            {"col": pl.Series([""], dtype=pl.String)},
            {"col": pl.Series([None], dtype=pl.String)},
        ),
        # 3. Single dash gets converted to 'None'
        (
            {"col": pl.Series(["-"], dtype=pl.String)},
            {"col": pl.Series([None], dtype=pl.String)},
        ),
        # 4. Other single punctuation gets converted to 'None'
        (
            {"col": pl.Series([","], dtype=pl.String)},
            {"col": pl.Series([None], dtype=pl.String)},
        ),
        # 5. Removes whitespace between hyphen
        (
            {"col": pl.Series(["1 - 3"], dtype=pl.String)},
            {"col": pl.Series(["2.0"], dtype=pl.String)},
        ),
    ],
)
def test_function_handles_absence_indicators(
    data: dict[str, pl.Series], expected: dict[str, pl.Series]
) -> None:
    """Asserts that whitespace is correctly removed."""
    lf = cleanse_float_values(pl.LazyFrame(data), ["col"])
    expected_lf = pl.LazyFrame(expected)
    assert_frame_equal(lf, expected_lf)
