"""Parses raw MIMIC-IV data and converts it long format.

For MIMIC-IV data to be parsable by MOTOR it needs to first
be converted into long format by patient and its codes into OMOP
format.
"""

from pathlib import Path


def build_command(
    src: Path,
    dest: Path,
    *,
    num_shards: int,
    num_proc: int,
    backend: str,
    memory_max=None,
) -> list[str]:
    """Builds the command responsible for running meds."""
