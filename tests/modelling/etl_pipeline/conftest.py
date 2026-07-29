"""Fixtures for ETL pipeline testing suite."""

import subprocess
from collections.abc import Callable
from pathlib import Path

import msgpack
import pytest

from thesis.modelling.etl_pipeline import base_meds
from thesis.modelling.etl_pipeline.base_meds import build_command
from thesis.modelling.etl_pipeline.coverage import MotorVocab
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
def reader_layout(tmp_path: Path) -> dict[str, Path]:
    """Returns a valid on-disk layout for the reader."""
    src = tmp_path / "meds"
    (src / "data").mkdir(parents=True)
    return {
        "src": src,
        "dest": tmp_path / "reader_db",
        "executable": tmp_path / "bin" / "meds_reader_convert",
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


@pytest.fixture
def rollup_entry() -> Callable:
    """Returns a factory for single dictionary entries."""

    def _make(code_string: str, entry_type: int, **overrides) -> dict:
        return {
            "code_string": code_string,
            "text_string": "",
            "type": entry_type,
            "val_start": 0.0,
            "val_end": 0.0,
            "weight": -0.1,
        } | overrides

    return _make


@pytest.fixture
def make_dictionary(tmp_path: Path) -> Callable:
    """Returns a factory writing a msgpack dictionary."""

    def _make(**overrides) -> Path:
        defaults = {
            "age_stats": {"mean": 0.0, "std": 1.0},
            "all_parents": {"SNOMED/1": ("SNOMED/1",)},
            "ontology_rollup": [],
            "regular": [],
        }
        defaults.update(**overrides)

        path = tmp_path / "dictionary"
        path.write_bytes(msgpack.dumps(defaults))
        return path

    return _make


@pytest.fixture
def make_vocab() -> Callable:
    """Returns a factory for MotorVocab objects, one code seeded per token set."""

    def _make(**overrides) -> MotorVocab:
        defaults = {
            "code_tokens": frozenset({"SNOMED/1"}),
            "numeric_codes": frozenset({"LOINC/2"}),
            "text_codes": frozenset({"SNOMED/3"}),
            "all_parents": {"SNOMED/1": ("SNOMED/1",)},
        }
        fields = defaults | overrides
        # weights covers exactly the tokens, so overriding a set alone stays coherent;
        # a case that cares about the tie-break passes its own spread of weights
        fields.setdefault(
            "weights",
            dict.fromkeys(
                fields["code_tokens"] | fields["numeric_codes"] | fields["text_codes"],
                -0.1,
            ),
        )

        return MotorVocab(**fields)

    return _make


@pytest.fixture
def meds_outputs() -> Callable:
    """Returns a factory for building MEDS base outputs."""

    def _make(drops: list, **overrides: dict[str, list]) -> dict[str, list]:
        defaults = {
            "subject_id": ["1", "2", "3"],
            "code": ["a", "b", "c"],
        }
        defaults.update(**overrides)
        for col in drops:
            defaults.pop(col, None)

        return defaults

    return _make
