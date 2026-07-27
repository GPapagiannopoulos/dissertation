"""Testing suite for the function guards and subprocess-related test."""

import subprocess
from pathlib import Path

import pytest

from thesis.modelling.etl_pipeline import meds_reader_db
from thesis.modelling.etl_pipeline.meds_reader_db import (
    build_db_command,
    run_meds_reader_convert,
)


def test_launches_build_command(
    spy_run: list[tuple[list[str], dict]], reader_layout: dict[str, Path]
) -> None:
    """Asserts that the build command launches successfully."""
    expected = build_db_command(
        reader_layout["src"],
        reader_layout["dest"],
        executable=reader_layout["executable"],
    )

    result = run_meds_reader_convert(**reader_layout)

    assert len(spy_run) == 1
    cmd, _ = spy_run[0]
    assert cmd == expected
    assert result == reader_layout["dest"]


def test_checks_exit_status(
    spy_run: list[tuple[list[str], dict]], reader_layout: dict[str, Path]
) -> None:
    """Asserts a non-zero exit status is raised."""
    run_meds_reader_convert(**reader_layout)

    _, kwargs = spy_run[0]
    assert kwargs.get("check") is True


def test_does_not_capture_output(
    spy_run: list[tuple[list[str], dict]], reader_layout: dict[str, Path]
) -> None:
    """Asserts that the output is not captured to a buffer."""
    run_meds_reader_convert(**reader_layout)

    _, kwargs = spy_run[0]
    assert kwargs.get("capture_output") is not True
    assert kwargs.get("stdout") is None
    assert kwargs.get("stderr") is None


def test_refuses_existing_destination(
    spy_run: list[tuple[list[str], dict]], reader_layout: dict[str, Path]
) -> None:
    """If dest already exists, the function raises."""
    reader_layout["dest"].mkdir(parents=True)

    with pytest.raises(FileExistsError):
        run_meds_reader_convert(**reader_layout)

    assert spy_run == []


def test_requires_med_dataset_source(
    spy_run: list[tuple[list[str], dict]], reader_layout: dict[str, Path]
) -> None:
    """If the source is not a valid dataset, the function raises."""
    args = {**reader_layout, "src": reader_layout["src"] / "data"}

    with pytest.raises(FileNotFoundError):
        run_meds_reader_convert(**args)

    assert spy_run == []


def test_resolves_executable_from_path(
    spy_run: list[tuple[list[str], dict]],
    reader_layout: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert that paths land in the emitted command."""
    resolved = reader_layout["executable"]
    monkeypatch.setattr(meds_reader_db.shutil, "which", lambda name: str(resolved))
    args = {k: v for k, v in reader_layout.items() if k != "executable"}

    run_meds_reader_convert(**args)

    cmd, _ = spy_run[0]
    assert cmd[0] == str(resolved)


def test_raises_when_executable_is_absent(
    spy_run: list[tuple[list[str], dict]],
    reader_layout: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing executable should raise."""
    monkeypatch.setattr(meds_reader_db.shutil, "which", lambda name: None)
    args = {k: v for k, v in reader_layout.items() if k != "executable"}

    with pytest.raises(FileNotFoundError):
        run_meds_reader_convert(**args)

    assert spy_run == []


def test_propagates_nonzero_exit(
    reader_layout: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asserts that if the process fails, the error is propagated."""

    def _failing_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(meds_reader_db.subprocess, "run", _failing_run)

    with pytest.raises(subprocess.CalledProcessError):
        run_meds_reader_convert(**reader_layout)
