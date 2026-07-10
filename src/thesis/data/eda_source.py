"""Class acting as port for Streamlit dashboard."""

from typing import NamedTuple, Protocol

import polars as pl


class MixedUnitsError(Exception):
    """Raised when a numeric slice spans more than one unit of measurement.

    Aggregating values recorded in different units yields a meaningless
    summary. Meant to prompt the dashboard user to narrow the selection.
    """

    def __init__(self, units: list[str]):
        """Store the offending units and build the error message."""
        self.units = units
        super().__init__(
            f"Multiple units found in slice: {units}. "
            "Additional filtering may be necessary."
        )


class NumericSummary(NamedTuple):
    """Summary of a numeric field: describe() stats plus the unit scalar."""

    stats: pl.DataFrame
    unit: str | None


class EDASource(Protocol):
    """Read-only aggregate view of the underlying mimic_data."""

    def event_types(self) -> list[str]:
        """Return a list of distinct event types."""
        pass

    def n_events(self, event_type: str) -> int:
        """Return the number of events of a specific type.

        Args:
            event_type (str): The type of event to return.

        Returns:
            int: The number of distinct events of that type aggregated
            across patients.

        """
        pass

    def n_patients(self, event_type: str) -> int:
        """Return the number of distinct patients for a specific event type.

        Args:
            event_type (str): The type of event to return.

        Returns:
            int: The number of distinct patients aggregated at event_type.
        """
        pass

    def fields(self, event_type: str) -> list[str]:
        """Return a list of field names for a specific event type."""
        pass

    def field_dtypes(self, event_type: str):
        """Return a mimic_data structure with field names and dtypes."""
        pass

    def describe_categorical_field(self, field_name: str):
        """Return a description of a categorical field belonging to an attribute."""
        pass

    def describe_numeric_field(
        self,
        field_name: str,
        filters: dict[str, str],
        uom_field: str | None = None,
    ) -> NumericSummary:
        """Return summary statistics for a numeric field scoped to one cohort.

        Args:
            field_name (str): the numeric field to summarize.
            filters (dict[str, str]): field:value equality filters pinning the
                cohort (e.g. {"labevents/label": "Red Blood Cells"}).
            uom_field (str | None): the field holding the unit of measurement,
                or None for fields with no unit.

        Returns:
            NumericSummary: the describe() statistics and the cohort's unit.

        Raises:
            MixedUnitsError: if the filtered slice spans multiple units.
        """
        pass

    def preview(self):
        """Return a preview of the underlying mimic_data."""
        pass
