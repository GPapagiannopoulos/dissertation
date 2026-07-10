"""Class acting as port for Streamlit dashboard."""

from typing import Protocol


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
    ):
        """Return a description of a numerical field belonging to an attribute."""
        pass

    def preview(self):
        """Return a preview of the underlying mimic_data."""
        pass
