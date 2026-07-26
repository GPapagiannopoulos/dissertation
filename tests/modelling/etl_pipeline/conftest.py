"""Fixtures for ETL pipeline testing suite."""

import subprocess
from pathlib import Path

import pytest

from thesis.modelling.etl_pipeline import base_meds
from thesis.modelling.etl_pipeline.base_meds import build_command
from thesis.modelling.etl_pipeline.meds_reader_db import build_db_command


@pytest.fixture
def command_paths(tmp_path: Path) -> dict[str, Path]:
    """Returns the path arguments shared by the command factory and expectations."""
    return {
        "src": tmp_path / "mimic",
        "dest": tmp_path / "out",
        "executable": tmp_path / "bin" / "meds_etl_mimic",
    }


@pytest.fixture
def make_command(command_paths: dict[str, Path]):
    """Returns a factory for building meds_etl_mimic commands."""

    def _make(**overrides) -> list[str]:
        return build_command(**(command_paths | overrides))

    return _make


@pytest.fixture
def reader_paths(tmp_path: Path) -> dict[str, Path]:
    """Returns the path arguments shared by the command factory and expectations."""
    return {
        "src": tmp_path / "mimic",
        "dest": tmp_path / "out",
        "executable": tmp_path / "bin" / "meds_reader_convert",
    }


@pytest.fixture
def make_db_command(reader_paths: dict[str, Path]):
    """Returns a factory for building MEDS reader commands."""

    def _make(**overrides) -> list[str]:
        return build_db_command(**(reader_paths | overrides))

    return _make


@pytest.fixture
def etl_layout(tmp_path: Path) -> dict[str, Path]:
    """Returns a valid on-disk layout: src holds a 2.2/ folder, dest does not exist."""
    src = tmp_path / "mimic"
    (src / "2.2").mkdir(parents=True)
    return {
        "src": src,
        "dest": tmp_path / "out",
        "executable": tmp_path / "bin" / "meds_etl_mimic",
    }


@pytest.fixture
def spy_run(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], dict]]:
    """Replaces subprocess.run with a recorder, so no real ETL is ever launched."""
    calls: list[tuple[list[str], dict]] = []

    def _fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(base_meds.subprocess, "run", _fake_run)
    return calls
