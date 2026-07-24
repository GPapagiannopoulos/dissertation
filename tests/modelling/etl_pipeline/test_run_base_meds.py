"""Testing suites for the run_base_meds helper function."""

from pathlib import Path

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
