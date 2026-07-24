"""Testing suites for the run_base_meds helper function."""

import subprocess
from pathlib import Path

import pytest

from thesis.modelling.etl_pipeline import base_meds
from thesis.modelling.etl_pipeline.base_meds import build_command, run_base_meds


def test_run_base_meds_launches_the_build_command(
    spy_run: list[tuple[list[str], dict]], etl_layout: dict[str, Path]
) -> None:
    """Asserts the runner delegates argv to build_command and launches."""
    expected = build_command(
        etl_layout["src"], etl_layout["dest"], executable=etl_layout["executable"]
    )

    result = run_base_meds(**etl_layout)

    assert len(spy_run) == 1
    cmd, _ = spy_run[0]
    assert cmd == expected
    assert result == etl_layout["dest"]


def test_run_base_meds_checks_exit_status(
    spy_run: list[tuple[list[str], dict]], etl_layout: dict[str, Path]
) -> None:
    """Asserts a non-zero exit status is raised."""
    run_base_meds(**etl_layout)

    _, kwargs = spy_run[0]
    assert kwargs.get("check") is True


def test_run_base_meds_does_not_capture_output(
    spy_run: list[tuple[list[str], dict]], etl_layout: dict[str, Path]
) -> None:
    """Asserts progress streams to the terminal not into buffer."""
    run_base_meds(**etl_layout)

    _, kwargs = spy_run[0]
    assert kwargs.get("capture_output") is not True
    assert "stdout" not in kwargs
    assert "stderr" not in kwargs


def test_run_base_meds_refuses_existing_destination(
    spy_run: list[tuple[list[str], dict]], etl_layout: dict[str, Path]
) -> None:
    """Asserts an existing dest is rejected not overwritten."""
    etl_layout["dest"].mkdir(parents=True)

    with pytest.raises(FileExistsError):
        run_base_meds(**etl_layout)

    assert spy_run == []


def test_run_base_meds_requires_versioned_subdirectory(
    spy_run: list[tuple[list[str], dict]], etl_layout: dict[str, Path]
) -> None:
    """Asserts src must hold the 2.2/ folder, not be the version folder itself."""
    # Pointing src straight at the version folder is the classic mistake.
    args = {**etl_layout, "src": etl_layout["src"] / "2.2"}

    with pytest.raises(FileNotFoundError):
        run_base_meds(**args)

    assert spy_run == []


def test_run_base_meds_resolves_executable_from_path(
    spy_run: list[tuple[list[str], dict]],
    etl_layout: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asserts a None executable is looked up on PATH via shutil.which."""
    resolved = etl_layout["executable"]
    monkeypatch.setattr(base_meds.shutil, "which", lambda name: str(resolved))
    args = {k: v for k, v in etl_layout.items() if k != "executable"}

    run_base_meds(**args)

    cmd, _ = spy_run[0]
    assert cmd[0] == str(resolved)


def test_run_base_meds_errors_when_executable_absent(
    spy_run: list[tuple[list[str], dict]],
    etl_layout: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asserts a missing meds_etl_mimic fails here, not at subprocess time."""
    monkeypatch.setattr(base_meds.shutil, "which", lambda name: None)
    args = {k: v for k, v in etl_layout.items() if k != "executable"}

    with pytest.raises(FileNotFoundError):
        run_base_meds(**args)

    assert spy_run == []


def test_run_base_meds_propagates_nonzero_exit(
    etl_layout: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asserts a failed ETL surfaces as CalledProcessError, not a silent return."""

    def _failing_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(base_meds.subprocess, "run", _failing_run)

    with pytest.raises(subprocess.CalledProcessError):
        run_base_meds(**etl_layout)
