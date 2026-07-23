"""Fixtures for ETL pipeline testing suite."""

from pathlib import Path

import pytest

from thesis.modelling.etl_pipeline.base_meds import build_command


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
