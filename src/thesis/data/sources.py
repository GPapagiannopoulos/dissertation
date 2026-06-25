"""Adapter for the MIMIC4Dataset class from PyHealth."""

import polars as pl


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

    def describe_field(self, event_type: str, field_name: str) -> pl.DataFrame:
        """Return a dataframe with summary measures for a given field.

        Args:
            event_type(str): the event_type whose attributes we filter for
            field_name(str): the name of the field to filter for

        Returns:
            pd.DataFrame: a dataframe offering summary measures. If the field is
            numeric, it returns summary statistics as per the pl.Series.describe()
            method.Otherwise, it returns the proportion of each value in the field.

        """
        target_field = self._events.filter(
            pl.col(f"{event_type}/{field_name}")
        ).to_series()
        if target_field.is_numeric():
            return target_field.describe()
        else:
            return target_field.value_counts(normalize=True)


def from_pyhealth(dataset) -> PolarsEDASource:
    """Construct a PolarsEDASource from a PyHealth-loaded dataset."""
    frame = dataset.global_event_df.collect()
    return PolarsEDASource(frame)
