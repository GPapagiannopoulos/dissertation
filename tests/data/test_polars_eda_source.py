"""Testing suite for the methods of the PolarsEDASource class."""

from collections.abc import Callable

import polars as pl
import pytest
from polars.exceptions import ColumnNotFoundError
from polars.testing import assert_frame_equal

from thesis.data.eda_source import MixedUnitsError
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


@pytest.mark.parametrize(
    "event_type, number",
    [
        # 0. Happy path
        ("patients", 2),
        # 1. Event_type not found
        ("missing_type", 0),
    ],
)
def test_polars_eda_n_events_method_happy_path(
    make_source: Callable, event_type: str, number: int
) -> None:
    """Asserts that n_events returns the correct number of events."""
    source = make_source()
    assert source.n_events(event_type) == number


@pytest.mark.parametrize(
    "overrides, event_type, expected_number",
    [
        # 0. Simple retrieval by event_type
        ({}, "admissions", 1),
        # 1. Does not count duplicate patient ids
        ({"patient_id": pl.Series(["1", "1", "1"], dtype=pl.String)}, "patients", 1),
        # 2. Doesn't count 'None' values
        # Not using patient_id because constructor raises
        ({"col": pl.Series([None, None, None], dtype=pl.String)}, "col", 0),
        # 3. Returns 0 if event_type not found
        ({}, "wrong_event_type", 0),
    ],
)
def test_polars_eda_n_patients_method_happy_path(
    make_source: Callable,
    overrides: dict[str, pl.Series],
    event_type: str,
    expected_number: int,
) -> None:
    """Asserts that n_patients returns the correct number o patients."""
    source = make_source(**overrides)
    assert source.n_patients(event_type) == expected_number


@pytest.mark.parametrize(
    "overrides, event_type, expected_list",
    [
        # 0. Correctly extracts single field
        (
            {"patients/col": pl.Series(["a", "b", None], dtype=pl.String)},
            "patients",
            ["patients/col"],
        ),
        # 1. Correct extracts multiple fields
        (
            {
                "patients/col_a": pl.Series(["a", "b", None], dtype=pl.String),
                "patients/col_b": pl.Series(["1", "2", None], dtype=pl.String),
            },
            "patients",
            ["patients/col_a", "patients/col_b"],
        ),
        # 2. Ignores other fields
        (
            {
                "patients/col": pl.Series(["a", "b", None], dtype=pl.String),
                "admissions/col": pl.Series(["1", "2", None], dtype=pl.String),
            },
            "admissions",
            ["admissions/col"],
        ),
        # 3. If field name is empty, returns prefix followed by empty string
        (
            {"patients/": pl.Series(["a", "b", None], dtype=pl.String)},
            "patients",
            ["patients/"],
        ),
        # 4. If no fields belonging to that event_type, returns an empty list
        ({}, "diagnosis", []),
        # 5. List is sorted in ascending order
        (
            {
                "patients/col_b": pl.Series(["a", "b", None], dtype=pl.String),
                "patients/col_a": pl.Series(["1", "2", None], dtype=pl.String),
            },
            "patients",
            ["patients/col_a", "patients/col_b"],
        ),
    ],
)
def test_polars_eda_fields_method_happy_path(
    make_source: Callable,
    overrides: dict[str, pl.Series],
    event_type: str,
    expected_list: list[str],
) -> None:
    """Asserts that fields method retrieves the correct fields."""
    source = make_source(**overrides)
    assert source.fields(event_type) == expected_list


@pytest.mark.parametrize(
    "overrides, event_type, expected_df_data",
    [
        # 0. Retrieves the correct field names
        # Implicitly excludes non-patients fields from the default df
        (
            {"patients/col": pl.Series([1, 2, None], dtype=pl.UInt16)},
            "patients",
            {"field": "patients/col", "dtype": "UInt16"},
        ),
        # 1. Resulting df is sorted in ascending order
        (
            {
                "patients/col_b": pl.Series(["1", "2", None], dtype=pl.String),
                "patients/col_a": pl.Series(["a", None, "b"], dtype=pl.String),
            },
            "patients",
            {
                "field": ["patients/col_a", "patients/col_b"],
                "dtype": ["String", "String"],
            },
        ),
        # 2. Empty df if no relevant cols
        (
            {"patients/col_a": pl.Series(["a", "b", None], dtype=pl.String)},
            "admissions",
            {
                "field": [],
                "dtype": [],
            },
        ),
        # 3. dtype matches the column for which it was extracted
        (
            {
                "patients/col_c": pl.Series(["a", "b", None], dtype=pl.String),
                "patients/col_a": pl.Series([1, 2, None], dtype=pl.UInt16),
                "patients/col_d": pl.Series([True, False, None], dtype=pl.Boolean),
                "patients/col_b": pl.Series([1.0, 2.0, None], dtype=pl.Float32),
            },
            "patients",
            {
                "field": [
                    "patients/col_a",
                    "patients/col_b",
                    "patients/col_c",
                    "patients/col_d",
                ],
                "dtype": ["UInt16", "Float32", "String", "Boolean"],
            },
        ),
    ],
)
def test_polars_eda_fields_dtypes_method_happy_path(
    make_source: Callable,
    overrides: dict[str, pl.Series],
    event_type: str,
    expected_df_data: dict[str, pl.Series],
) -> None:
    """Asserts that the method returns the correct dtypes.

    Selection of the correct fields has been tested separately.
    """
    source = make_source(**overrides)
    expectation = pl.DataFrame(expected_df_data)
    assert_frame_equal(source.field_dtypes(event_type), expectation)


@pytest.mark.parametrize(
    "overrides, event_type, expected_df_data",
    [
        # 0. Outputs head of target event type
        (
            {"patients/col": pl.Series([1, 2, 3], dtype=pl.UInt16)},
            "patients",
            {"patients/col": pl.Series([1, 2, 3], dtype=pl.UInt16)},
        ),
        # 1. Head contains all appropriate fields
        (
            {
                "patients/col_a": pl.Series(["a", "b", "c"], dtype=pl.String),
                "patients/col_b": pl.Series([1, 2, 3], dtype=pl.UInt16),
            },
            "patients",
            {
                "patients/col_a": pl.Series(["a", "b", "c"], dtype=pl.String),
                "patients/col_b": pl.Series([1, 2, 3], dtype=pl.UInt16),
            },
        ),
        # 2. Only records not meeting the cutoff excluded
        (
            {
                "patients/col_a": pl.Series(["a", "b", None], dtype=pl.String),
                "patients/col_b": pl.Series([1, 2, 3], dtype=pl.UInt16),
            },
            "patients",
            {
                "patients/col_a": pl.Series(["a", "b"], dtype=pl.String),
                "patients/col_b": pl.Series([1, 2], dtype=pl.UInt16),
            },
        ),
        # 3. Records exactly at the cutoff excluded
        (
            {
                "patients/col_a": pl.Series(["a", "b", "c"], dtype=pl.String),
                "patients/col_b": pl.Series(["a", "b", "c"], dtype=pl.String),
                "patients/col_c": pl.Series(["a", "b", "c"], dtype=pl.String),
                "patients/col_d": pl.Series(["a", "b", None], dtype=pl.String),
                "patients/col_e": pl.Series(["a", "b", None], dtype=pl.String),
            },
            "patients",
            {
                "patients/col_a": pl.Series(["a", "b"], dtype=pl.String),
                "patients/col_b": pl.Series(["a", "b"], dtype=pl.String),
                "patients/col_c": pl.Series(["a", "b"], dtype=pl.String),
                "patients/col_d": pl.Series(["a", "b"], dtype=pl.String),
                "patients/col_e": pl.Series(["a", "b"], dtype=pl.String),
            },
        ),
        # 4. Preview is empty if no records fulfill criteria
        (
            {"patients/col_a": pl.Series([None, None, None], dtype=pl.String)},
            "patients",
            {"patients/col_a": pl.Series([], dtype=pl.String)},
        ),
    ],
)
def test_polars_eda_preview_method_happy_path(
    make_source: Callable,
    overrides: dict[str, pl.Series],
    event_type: str,
    expected_df_data: dict[str, pl.Series],
) -> None:
    """Asserts that the preview method returns only valid records."""
    source = make_source(**overrides)
    expected_df = pl.DataFrame(expected_df_data)
    assert_frame_equal(source.preview_table(event_type), expected_df)


@pytest.mark.parametrize(
    "overrides, expected_df, target_field",
    [
        # 0. Correctly calculates a single value
        (
            {"patients/col_a": pl.Series(["a", "a", "a"], dtype=pl.String)},
            {
                "patients/col_a": pl.Series(["a"], dtype=pl.String),
                "proportion": pl.Series([1.0], dtype=pl.Float64),
            },
            "patients/col_a",
        ),
        # 1. Correctly calculates the proportion of multiple event types
        (
            {
                "event_type": pl.Series(["event"] * 3, dtype=pl.String),
                "event/col": pl.Series(["a", "b", "c"], dtype=pl.String),
            },
            {
                "event/col": pl.Series(["a", "b", "c"], dtype=pl.String),
                "proportion": pl.Series([1 / 3, 1 / 3, 1 / 3], dtype=pl.Float64),
            },
            "event/col",
        ),
        # 2. Counts proportion of null values
        (
            {
                "event_type": pl.Series(["event"] * 3, dtype=pl.String),
                "event/col": pl.Series([None] * 3, dtype=pl.String),
            },
            {
                "event/col": pl.Series([None], dtype=pl.String),
                "proportion": pl.Series([1.0], dtype=pl.Float64),
            },
            "event/col",
        ),
        # 3. Does not count target_field nulls belonging to a different event_type
        (
            {
                "event_type": pl.Series(["event", "event", "other"], dtype=pl.String),
                "event/col": pl.Series(["a", "a", None], dtype=pl.String),
            },
            {
                "event/col": pl.Series(["a"], dtype=pl.String),
                "proportion": pl.Series([1.0], dtype=pl.Float64),
            },
            "event/col",
        ),
    ],
)
def test_polars_eda_describe_categorical_field_happy_path(
    make_source: Callable,
    overrides: dict[str, pl.Series],
    expected_df: dict[str, pl.Series],
    target_field: str,
):
    """Asserts standard behaviour for describe_categorical_field method."""
    source = make_source(**overrides)
    expectation = pl.DataFrame(expected_df)
    assert_frame_equal(source.describe_categorical_field(target_field), expectation)


def test_polars_eda_describe_categorical_field_raise_if_missing_field(
    make_source: Callable,
) -> None:
    """Asserts that the function raises if target field is missing.

    Normal method use is reserved for the dashboard, where the parameter
    is derived from available columns. This guard exists for use outside
    the dashboard.
    """
    source = make_source()
    with pytest.raises(ColumnNotFoundError, match="Unable to find column"):
        source.describe_categorical_field("wrong_col/col")


def test_polars_eda_describe_categorical_field_raise_if_numeric_field(
    make_source: Callable,
) -> None:
    """Asserts that the function raises if target field is numeric."""
    overrides = {"patients/col_a": pl.Series([1, 2, 3], dtype=pl.Int16)}
    source = make_source(**overrides)
    with pytest.raises(ValueError, match="'patients/col_a' is not a categorical field"):
        source.describe_categorical_field("patients/col_a")


@pytest.mark.parametrize(
    "target_field, uom_field, overrides, filters, expected_df",
    [
        # 0. Method correctly retrieves by filter
        (
            "labevents/value",
            "labevents/uom",
            {
                "value": pl.Series([10.0, 20.0, 100.0, 200.0], dtype=pl.Float64),
                "label": pl.Series(
                    ["test_a", "test_a", "test_b", "test_b"], dtype=pl.String
                ),
                "uom": pl.Series(["mg", "mg", "cm", "cm"], dtype=pl.String),
            },
            {"labevents/label": "test_a"},
            {
                "labevents/value": pl.Series([10.0, 20.0], dtype=pl.Float64),
                "labevents/label": pl.Series(["test_a", "test_a"], dtype=pl.String),
                "labevents/uom": pl.Series(["mg", "mg"], dtype=pl.String),
            },
        ),
        # 1. Filters on multiple dimensions with AND semantics
        (
            "prescriptions/dose_val",
            "prescriptions/dose_unit",
            {
                "dose_val": pl.Series([10.0, 20.0, 30.0, 40.0], dtype=pl.Float64),
                "drug": pl.Series(
                    ["aspirin", "aspirin", "aspirin", "warfarin"], dtype=pl.String
                ),
                "route": pl.Series(["PO", "IV", "PO", "PO"], dtype=pl.String),
                "dose_unit": pl.Series(["mg", "mg", "mg", "mg"], dtype=pl.String),
            },
            {"prescriptions/drug": "aspirin", "prescriptions/route": "PO"},
            {
                "prescriptions/dose_val": pl.Series([10.0, 30.0], dtype=pl.Float64),
                "prescriptions/drug": pl.Series(
                    ["aspirin", "aspirin"], dtype=pl.String
                ),
                "prescriptions/route": pl.Series(["PO", "PO"], dtype=pl.String),
                "prescriptions/dose_unit": pl.Series(["mg", "mg"], dtype=pl.String),
            },
        ),
        # 2. Filter matching no rows returns an empty slice with schema intact
        (
            "labevents/value",
            "labevents/uom",
            {
                "value": pl.Series([10.0, 20.0], dtype=pl.Float64),
                "label": pl.Series(["test_a", "test_a"], dtype=pl.String),
                "uom": pl.Series(["mg", "mg"], dtype=pl.String),
            },
            {"labevents/label": "does_not_exist"},
            {
                "labevents/value": pl.Series([], dtype=pl.Float64),
                "labevents/label": pl.Series([], dtype=pl.String),
                "labevents/uom": pl.Series([], dtype=pl.String),
            },
        ),
    ],
)
def test_polars_eda_describe_numerical_field_happy_path(
    make_eav_source: Callable,
    target_field: str,
    uom_field: str | None,
    overrides: dict[str, pl.Series],
    filters: dict[str, str],
    expected_df: dict[str, pl.Series],
) -> None:
    """Asserts standard behaviour for describe_numerical_field method."""
    source = make_eav_source(target_field.split("/")[0], **overrides)
    data_slice = source._numeric_subset(target_field, filters, uom_field)
    expected = pl.DataFrame(expected_df)
    assert_frame_equal(data_slice, expected)


def test_polars_eda_numeric_subset_excludes_other_event_types(
    make_source: Callable,
) -> None:
    """Asserts the slice drops rows belonging to a different event_type.

    Foreign rows are null by design in real data, but here the foreign row
    is given a matching label and a non-null value so that only the
    event_type filter can be responsible for excluding it.
    """
    source = make_source(
        event_type=pl.Series(["labevents", "labevents", "other"], dtype=pl.String),
        patient_id=pl.Series(["1", "2", "3"], dtype=pl.String),
        **{
            "labevents/value": pl.Series([10.0, 20.0, 999.0], dtype=pl.Float64),
            "labevents/label": pl.Series(
                ["test_a", "test_a", "test_a"], dtype=pl.String
            ),
            "labevents/uom": pl.Series(["mg", "mg", "mg"], dtype=pl.String),
        },
    )
    result = source._numeric_subset(
        "labevents/value", {"labevents/label": "test_a"}, "labevents/uom"
    )
    expected = pl.DataFrame(
        {
            "labevents/value": pl.Series([10.0, 20.0], dtype=pl.Float64),
            "labevents/label": pl.Series(["test_a", "test_a"], dtype=pl.String),
            "labevents/uom": pl.Series(["mg", "mg"], dtype=pl.String),
        }
    )
    assert_frame_equal(result, expected)


def test_polars_eda_numeric_subset_uom_field_none_omits_unit(
    make_eav_source: Callable,
) -> None:
    """Asserts that a None uom_field yields no unit column in the slice.

    Some numeric fields have no unit of measurement, so the dashboard passes
    None. The slice should then contain only the target and filter fields.
    """
    source = make_eav_source(
        "labevents",
        value=pl.Series([10.0, 20.0], dtype=pl.Float64),
        label=pl.Series(["test_a", "test_a"], dtype=pl.String),
    )
    result = source._numeric_subset(
        "labevents/value", {"labevents/label": "test_a"}, None
    )
    expected = pl.DataFrame(
        {
            "labevents/value": pl.Series([10.0, 20.0], dtype=pl.Float64),
            "labevents/label": pl.Series(["test_a", "test_a"], dtype=pl.String),
        }
    )
    assert_frame_equal(result, expected)


def _stat(summary_stats: pl.DataFrame, name: str, column: str) -> float:
    """Pull a single statistic value out of a describe() frame."""
    return summary_stats.filter(pl.col("statistic") == name).get_column(column).item()


def test_polars_eda_describe_numeric_field_single_unit(
    make_eav_source: Callable,
) -> None:
    """Asserts stats describe the right column and the unit is returned."""
    source = make_eav_source(
        "labevents",
        value=pl.Series([10.0, 20.0, 30.0], dtype=pl.Float64),
        label=pl.Series(["test_a", "test_a", "test_a"], dtype=pl.String),
        uom=pl.Series(["mg", "mg", "mg"], dtype=pl.String),
    )
    result = source.describe_numeric_field(
        "labevents/value", {"labevents/label": "test_a"}, "labevents/uom"
    )
    assert result.unit == "mg"
    assert _stat(result.stats, "count", "labevents/value") == 3.0
    assert _stat(result.stats, "mean", "labevents/value") == 20.0


def test_polars_eda_describe_numeric_field_mixed_units_raises(
    make_eav_source: Callable,
) -> None:
    """Asserts a cohort spanning multiple units raises MixedUnitsError."""
    source = make_eav_source(
        "labevents",
        value=pl.Series([10.0, 20.0], dtype=pl.Float64),
        label=pl.Series(["test_a", "test_a"], dtype=pl.String),
        uom=pl.Series(["mg", "cm"], dtype=pl.String),
    )
    with pytest.raises(MixedUnitsError, match="mg") as excinfo:
        source.describe_numeric_field(
            "labevents/value", {"labevents/label": "test_a"}, "labevents/uom"
        )
    assert set(excinfo.value.units) == {"mg", "cm"}


def test_polars_eda_describe_numeric_field_no_unit(
    make_eav_source: Callable,
) -> None:
    """Asserts a None uom_field yields a None unit and still summarizes."""
    source = make_eav_source(
        "labevents",
        value=pl.Series([10.0, 20.0], dtype=pl.Float64),
        label=pl.Series(["test_a", "test_a"], dtype=pl.String),
    )
    result = source.describe_numeric_field(
        "labevents/value", {"labevents/label": "test_a"}, None
    )
    assert result.unit is None
    assert _stat(result.stats, "count", "labevents/value") == 2.0


def test_polars_eda_describe_numeric_field_empty_cohort(
    make_eav_source: Callable,
) -> None:
    """Asserts an empty cohort yields a None unit without crashing.

    Extracting a scalar unit from a zero-row slice must not raise; the
    dashboard can hand a filter value that matches no rows.
    """
    source = make_eav_source(
        "labevents",
        value=pl.Series([10.0, 20.0], dtype=pl.Float64),
        label=pl.Series(["test_a", "test_a"], dtype=pl.String),
        uom=pl.Series(["mg", "mg"], dtype=pl.String),
    )
    result = source.describe_numeric_field(
        "labevents/value", {"labevents/label": "does_not_exist"}, "labevents/uom"
    )
    assert result.unit is None
    assert _stat(result.stats, "count", "labevents/value") == 0.0
