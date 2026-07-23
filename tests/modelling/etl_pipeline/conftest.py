"""Fixtures for ETL pipeline testing suite."""

from pathlib import Path

import pytest

from src.thesis.modelling.etl_pipeline.base_meds import build_command


@pytest.fixture
def make_command(tmp_path: Path):
    """Returns a factory for building meds_mimic_cli commands."""

    def _make(**overrides):
        kwargs = {
            "src": tmp_path / "mimic",
            "dest": tmp_path / "out",
            "executable": tmp_path / "bin" / "meds_etl_mimic",
        }
        return build_command(**(kwargs | overrides))

    return _make
