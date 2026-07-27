"""Generates the MEDS database return the subject ids for downstream consumption."""

import subprocess
from pathlib import Path


def build_db_command(
    src: Path, dest: Path, *, executable: Path, num_threads: int = 1
) -> list[str]:
    """Generates the command arguments for the MEDS db creation command.

    Args:
        src (Path): Path to the sharded MEDS output
        dest (Path): Path to the db root
        executable (Path): Path to the cli
        num_threads (int): Number of subprocesses to spawn

    Returns:
        list[str]: a list of the command arguments

    Raises:
        ValueError: if the number of threads is not positive
    """
    if num_threads <= 0:
        raise ValueError(f"The minimum number of threads is 1. Received {num_threads}")
    return [str(executable), str(src), str(dest), "--num_threads", str(num_threads)]


def run_meds_reader_convert(
    src: Path, dest: Path, *, executable: Path, num_threads: int = 1
) -> Path:
    """Runs the MEDS reader convert command."""
    cmd = build_db_command(
        src,
        dest,
        executable=executable,
        num_threads=num_threads,
    )
    subprocess.run(cmd)
    return dest
