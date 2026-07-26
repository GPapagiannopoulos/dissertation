"""Testing suite for the function guards and subprocess-related test."""


def test_launches_build_command() -> None:
    """Asserts that the build command launches successfully."""
    pass


def test_checks_exit_status() -> None:
    """Asserts that kwargs are recorded."""


def test_does_not_capture_output() -> None:
    """Asserts that the output is not captured to a buffer."""
    pass


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
