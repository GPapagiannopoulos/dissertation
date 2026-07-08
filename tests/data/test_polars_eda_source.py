"""Testing suite for the methods of the PolarsEDASource class."""

from collections.abc import Callable

import polars as pl
import pytest

from thesis.data.sources import PolarsEDASource


@pytest.mark.parametrize(
    "data, expected_list",
    [
        # 0. Extracts the list from the event type
        (
            {"event_type": pl.Series(["a", "b", "c"], dtype=pl.String)},
            ["a", "b", "c"],
        ),
        # 1. Correctly sorts the list in ascending order
        (
            {"event_type": pl.Series(["c", "b", "a"], dtype=pl.String)},
            ["a", "b", "c"],
        ),
        # 2. Removes duplicates
        (
            {"event_type": pl.Series(["a", "a", "b"], dtype=pl.String)},
            ["a", "b"],
        ),
        # 3. No records returns an empty list
        (
            {
                "event_type": pl.Series([], dtype=pl.String),
                "patient_id": pl.Series([], dtype=pl.String),
            },
            [],
        ),
    ],
)
def test_polars_eda_event_types_method_happy_path(
    make_source: Callable,
    data: dict[str, pl.Series],
    expected_list: list[str],
) -> None:
    """Asserts that simple cases are handled correctly.

    Args:
        make_source (Callable): factory fixture to build PolarsEDASource instance
        data (dict[str, pl.Series]): a dictionary containing the
        dataframe data
        expected_list (list[str]): a list of the expected event types
    """
    source = make_source(**data)
    assert source.event_types() == expected_list


@pytest.mark.parametrize(
    "df",
    [
        # 0. Singular Null value
        pl.DataFrame(
            {
                "event_type": pl.Series(["a", "b", None], dtype=pl.String),
                "patient_id": pl.Series(["a", "b", "c"], dtype=pl.String),
            }
        ),
        # 1. Only Null values
        pl.DataFrame(
            {
                "event_type": pl.Series([None, None, None], dtype=pl.String),
                "patient_id": pl.Series(["a", "b", "c"], dtype=pl.String),
            }
        ),
        # 2. event_type field missing
        pl.DataFrame(
            {
                "col": pl.Series(["a", "b", "c"], dtype=pl.String),
                "patient_id": pl.Series(["a", "b", "c"], dtype=pl.String),
            }
        ),
    ],
)
def test_constructor_raises_if_invalid_df(df: pl.DataFrame) -> None:
    """Asserts that retrieving event_types with None entries raises.

    patient_id and event_type form the core of
    PyHealth's MIMIC-IV loader. Both fields are mandatory
    and should not contain None values. If they do there is
    programmer error or a malformed DataFrame.
    """
    with pytest.raises(ValueError):
        PolarsEDASource(df)
