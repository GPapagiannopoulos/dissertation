"""Testing suite for the methods of the PolarsEDASource class."""

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
            {"event_type": pl.Series(["a", "a", "b", "b", "c"], dtype=pl.String)},
            ["a", "b", "c"],
        ),
        # 3. No records returns an empty list
        ({"event_type": pl.Series([], dtype=pl.String)}, []),
    ],
)
def test_polars_eda_event_types_method_happy_path(
    data: dict[str, pl.Series], expected_list: list[str]
) -> None:
    """Asserts that simple cases are handled correctly.

    Args:
        data (dict[str, pl.Series]): a dictionary containing the
        dataframe data
        expected_list (list[str]): a list of the expected event types
    """
    source = PolarsEDASource(pl.DataFrame(data))
    assert source.event_types() == expected_list
