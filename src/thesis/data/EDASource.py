"""Class acting as port for Streamlit dashboard."""

from typing import Protocol


class EDASource(Protocol):
    """Read-only aggregate view of the underlying data."""

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

    def describe_field(self, field_name: str):
        """Return a description of a specific field belonging to an attribute."""
        pass
