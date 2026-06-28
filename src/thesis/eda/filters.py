"""Filters and filtering function used in excluding fields from EDA."""

import polars as pl

from thesis.data.sources import PolarsEDASource


def is_id_field(field_name: str) -> bool:
    """Check if a field is a UID field."""
    return field_name.endswith("_id")


def is_date_field(field_dtype: str) -> bool:
    """Check if a field a date field."""
    return field_dtype.startswith("Date")


def is_unit_field(field_name: str) -> bool:
    """Checks if a field is a unit field."""
    mimic4_ehr_unit_cols = frozenset(
        ["labevents/valueuom", "prescriptions/dose_unit_rx"]
    )
    return field_name in mimic4_ehr_unit_cols


def valid_fields(src: PolarsEDASource, event_type: str) -> list[str]:
    """Returns a list of fields that can be meaningfully summarized."""
    data = src.field_dtypes(event_type)

    return [
        c
        for c in data.select("field").to_series()
        if not is_id_field(c)
        and not is_date_field(data.filter(pl.col("field") == c).select("dtype").item())
        and not is_unit_field(c)
    ]
