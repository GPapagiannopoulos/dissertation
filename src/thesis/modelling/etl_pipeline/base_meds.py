"""Parses raw MIMIC-IV data and converts it long format.

For MIMIC-IV data to be parsable by MOTOR it needs to first
be converted into long format by patient and its codes into OMOP
format.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Literal

_BACKENDS = frozenset({"cpp", "polars"})
_EXECUTABLE_NAME = "meds_etl_mimic"
_VERSION_SUBDIR = "2.2"


def build_command(
    src: Path,
    dest: Path,
    *,
    executable: Path,
    num_shards: int = 100,
    num_proc: int = 1,
    backend: Literal["cpp", "polars"] = "cpp",
) -> list[str]:
    """Builds the command responsible for running meds_mimic.

    MEDS conversion for MIMIC-IV is via a CLI, as it doesn't expose
    an API. The command is constructed via function for testability.

    Args:
        src: Path to the directory containing MIMIC-IV data
        dest: Path to the output root
        executable: Path to the meds_etl_mimic console script
        num_shards: The number of subject partitioned parquet files
        num_proc: The number of processes for sharding
        backend: The implementation used for the operation

    Returns:
         list[str]: the argv passed to the cli

    Raises:
        ValueError: If backend isn't recognized
        ValueError: If invalid number of processes and shards supplied
    """
    if backend not in _BACKENDS:
        raise ValueError(f"Backend must be one of {sorted(_BACKENDS)}, got {backend!r}")
    if num_shards < 1 or num_proc < 1:
        raise ValueError("num_shards and num_proc must be >= 1")
    return [
        str(executable),
        str(src),
        str(dest),
        "--num_shards",
        str(num_shards),
        "--num_proc",
        str(num_proc),
        "--backend",
        backend,
    ]


def run_base_meds(
    src: Path,
    dest: Path,
    *,
    executable: Path | None = None,
    num_shards: int = 100,
    num_proc: int = 1,
    backend: Literal["cpp", "polars"] = "cpp",
) -> Path:
    """Initiates a subprocess to run the MEDS conversion."""
    if not (src / _VERSION_SUBDIR).is_dir():
        raise FileNotFoundError(
            f"Expected a {_VERSION_SUBDIR!r} subfolder inside src {src}. "
            f"Point src at the directory containing {_VERSION_SUBDIR}/, "
            f"not at the version folder itself."
        )
    if dest.exists():
        raise FileExistsError(
            f"Destination {dest} already exists; refusing to overwrite. "
            f"Remove it or choose a new path."
        )
    if executable is None:
        found = shutil.which(_EXECUTABLE_NAME)
        if found is None:
            raise FileNotFoundError(
                f"Could not find {_EXECUTABLE_NAME!r} on PATH. Is the modelling "
                f"environment active? (e.g. .venv-modelling/bin/{_EXECUTABLE_NAME})"
            )
        executable = Path(found)
    cmd = build_command(
        src,
        dest,
        executable=executable,
        num_shards=num_shards,
        num_proc=num_proc,
        backend=backend,
    )
    subprocess.run(cmd, check=True)
    return dest
