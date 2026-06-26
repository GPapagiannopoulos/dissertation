"""Adapter for the MIMIC4Dataset class from PyHealth."""

import polars as pl

from thesis.constants import DTYPE_TO_POLARS_DTYPE_MAP


def cast_frame(lf: pl.LazyFrame, dtype_map: dict[str, str]) -> pl.LazyFrame:
    """Apply dtype casts declared in the manifest to a LazyFrame.

    Columns absent from the frame are silently skipped. String→String
    casts are no-ops and are omitted. Date columns use str.to_date()
    because Polars 1.x does not support cast(pl.Date) from String.

    Args:
        lf: The LazyFrame to cast.
        dtype_map: Mapping of column name to dtype string (e.g. "UInt", "Date").

    Returns:
        LazyFrame with the cast expressions applied.

    """
    available = set(lf.collect_schema().names())
    exprs: list[pl.Expr] = []
    for field, dtype_str in dtype_map.items():
        if field not in available:
            continue
        polars_dtype = DTYPE_TO_POLARS_DTYPE_MAP.get(dtype_str)
        if polars_dtype is None or polars_dtype is pl.String:
            continue
        if polars_dtype is pl.Date:
            exprs.append(pl.col(field).str.to_date(format="%Y-%m-%d", strict=False))
        else:
            exprs.append(pl.col(field).cast(polars_dtype, strict=False))
    if not exprs:
        return lf
    return lf.with_columns(exprs)


class PolarsEDASource:
    """EDA adapter over PyHealth's global event dataframe."""

    _PATIENT = "patient_id"
    _TYPE = "event_type"

    def __init__(self, events: pl.DataFrame):
        """Initialize the adapter with a polars dataframe."""
        self._events = events

    def event_types(self) -> list[str]:
        """Return an alphabetically sorted list of event types.

        Returns:
            list[str]: an alphabetically sorted list of the unique values
            in the event_type column.

        """
        return self._events.select(self._TYPE).unique().to_series().sort().to_list()

    def n_events(self, event_type: str) -> int:
        """Return the number of rows where the event_type field matches event_type.

        Args:
            event_type(str): the name of the event type to filter for

        Returns:
            int: the number of rows after filtering for that event_type

        """
        return self._events.filter(pl.col(self._TYPE) == event_type).height

    def n_patients(self, event_type: str) -> int:
        """Return the number of distinct patients with a specific event type.

        Args:
            event_type(str): the name of the event type to filter for

        Returns:
            int: the number of distinct patients after filtering for that event_type

        """
        return (
            self._events.filter(pl.col(self._TYPE) == event_type)
            .select(self._PATIENT)
            .unique()
            .height
        )

    def fields(self, event_type: str) -> list[str]:
        """Return an alphabetically sorted list of dataframe field names.

        In PyHealth the fields belonging to an event_type are prefixed
        {event_type}/{field_name}.

        Args:
            event_type(str): the event_type whose attributes we filter for

        Returns:
            list[str]: an alphabetically sorted list of dataframe field names

        """
        prefix = f"{event_type}/"
        return sorted([c for c in self._events.columns if c.startswith(prefix)])

    def field_dtypes(self, event_type: str) -> pl.DataFrame:
        """Return a dataframe to display the field names and dtypes."""
        valid_cols = self.fields(event_type)
        col_dtypes = [str(self._events.schema[c]) for c in valid_cols]

        return pl.DataFrame({"field": valid_cols, "dtype": col_dtypes})

    def describe_field(self, field_name: str) -> pl.DataFrame:
        """Return a dataframe with summary measures for a given field.

        Args:
            field_name(str): the name of the field to filter for

        Returns:
            pd.DataFrame: a dataframe offering summary measures. If the field is
            numeric, it returns summary statistics as per the pl.Series.describe()
            method.Otherwise, it returns the proportion of each value in the field.

        """
        target_field = self._events.select(f"{field_name}").to_series()
        if self._events.schema[field_name].is_numeric():
            return target_field.describe()
        else:
            return target_field.value_counts(normalize=True).sort("proportion")

    def preview_table(self, table_name: str, n_rows: int = 10) -> pl.DataFrame:
        """Returns the head of the dataframe."""
        table_fields = self.fields(table_name)
        return (
            self._events.select(table_fields)
            .filter(pl.all_horizontal(pl.all().is_not_null()))
            .head(n_rows)
        )
