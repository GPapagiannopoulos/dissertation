"""Generates the MEDS database return the subject ids for downstream consumption."""

from pathlib import Path


def build_command(
    source: Path, dest: Path, *, executable: Path, num_threads: int = 1
) -> list[str]:
    """Generates the command arguments for the MEDS db creation command.

    Args:
        source (Path): Path to the sharded MEDS output
        dest (Path): Path to the db root
        executable (Path): Path to the cli
        num_threads (int): Number of subprocesses to spawn

    Returns:
        list[str]: a list of the command arguments
    """
    pass


def run_meds_reader_convert(
    source: Path, dest: Path, *, executable: Path, num_threads: int = 1
) -> Path:
    """Runs the MEDS reader convert command."""
    pass
