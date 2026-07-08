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
    "drop, overrides, error_message",
    [
        # 0. Singular Null value
        (
            [],
            {
                "event_type": pl.Series(["a", "b", None], dtype=pl.String),
            },
            "'None' value in 'event_type' detected.",
        ),
        # 1. Only Null values in event_type
        (
            [],
            {
                "event_type": pl.Series([None, None, None], dtype=pl.String),
            },
            "'None' value in 'event_type' detected.",
        ),
        # 2. event_type field missing
        (["event_type"], {}, "Missing 'event_type' column"),
        # 3. None in patient_id
        (
            [],
            {
                "patient_id": pl.Series(["a", None, "c"], dtype=pl.String),
            },
            "'None' value in 'patient_id' detected.",
        ),
        # 4. Only None values in patient_id
        (
            [],
            {
                "patient_id": pl.Series([None, None, None], dtype=pl.String),
            },
            "'None' value in 'patient_id' detected.",
        ),
        # 5. Missing patient_id col
        (["patient_id"], {}, "Missing 'patient_id' column"),
    ],
)
def test_constructor_raises_if_invalid_df(
    events_df: Callable,
    drop: list[str],
    overrides: dict[str, pl.Series],
    error_message: str,
) -> None:
    """Asserts that retrieving event_types with None entries raises.

    patient_id and event_type form the core of
    PyHealth's MIMIC-IV loader. Both fields are mandatory
    and should not contain None values. If they do there is
    programmer error or a malformed DataFrame.
    """
    with pytest.raises(ValueError, match=error_message):
        PolarsEDASource(events_df(drop=drop, **overrides))
