"""Builds the meds_reader database from a MEDS dataset.

Stage 2 of the MOTOR ETL: the long-format shards produced by base_meds are
converted into the column store meds_reader queries. Reading subject ids back
out of the built database is a separate, read-only stage.
"""

import shutil
import subprocess
from pathlib import Path

_EXECUTABLE_NAME = "meds_reader_convert"
_DATA_SUBDIR = "data"


def build_db_command(
    src: Path, dest: Path, *, executable: Path, num_threads: int = 1
) -> list[str]:
    """Generates the command arguments for the MEDS db creation command.

    Args:
        src (Path): Path to the sharded MEDS output
        dest (Path): Path to the db root
        executable (Path): Path to the cli
        num_threads (int): Number of threads the cli uses to process shards

    Returns:
        list[str]: a list of the command arguments

    Raises:
        ValueError: if the number of threads is not positive
    """
    if num_threads <= 0:
        raise ValueError(f"The minimum number of threads is 1. Received {num_threads}")
    return [str(executable), str(src), str(dest), "--num_threads", str(num_threads)]


def run_meds_reader_convert(
    src: Path, dest: Path, *, executable: Path | None = None, num_threads: int = 1
) -> Path:
    """Initiates a subprocess to build the meds_reader database.

    The database is built by a CLI, which fans out across num_threads.

    Args:
        src: Path to the MEDS dataset root, the directory holding data/
        dest: Path to the database root, which must not already exist
        executable: Path to the meds_reader_convert console script. When
            omitted it is resolved from PATH
        num_threads: Number of threads the CLI uses to build the database

    Returns:
        Path: the database root, for downstream consumption

    Raises:
        FileNotFoundError: If src holds no data/ subfolder
        FileNotFoundError: If the executable is neither given nor on PATH
        FileExistsError: If dest already exists
        subprocess.CalledProcessError: If the conversion exits non-zero
    """
    if not (src / _DATA_SUBDIR).is_dir():
        raise FileNotFoundError(
            f"Expected a {_DATA_SUBDIR!r} subfolder inside src {src}. "
            f"Point src at the MEDS dataset root containing {_DATA_SUBDIR}/, "
            f"not at the shard folder itself."
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
    cmd = build_db_command(
        src,
        dest,
        executable=executable,
        num_threads=num_threads,
    )
    subprocess.run(cmd, check=True)
    return dest
