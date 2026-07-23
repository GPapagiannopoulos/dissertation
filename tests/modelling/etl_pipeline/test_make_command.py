"""Testing suite for the make_command helper function."""

import pytest


@pytest.mark.parametrize(
    "overrides, expected",
    [
        # 0. Defaults produce the correct argv
    ],
)
def test_make_command_happy_path() -> None:
    """Asserts normal behaviour for make_command."""
