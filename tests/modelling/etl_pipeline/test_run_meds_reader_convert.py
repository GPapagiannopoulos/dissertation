"""Testing suite for the function guards and subprocess-related test."""

from pathlib import Path

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


def test_refuses_existing_destination() -> None:
    """If dest already exists, the function raises."""
    pass


def test_requires_med_dataset_source() -> None:
    """If the source is not a valid dataset, the function raises."""
    pass


def test_resolves_executable_from_path() -> None:
    """Assert that paths land in the emitted command."""
    pass


def test_raises_when_executable_is_absent() -> None:
    """Missing executable should raise."""
    pass


def test_propagates_nonzero_exit() -> None:
    """Asserts that if the process fails, the error is propagated."""
    pass
