"""Root pytest configuration: stack markers and cross-environment isolation.

The eda and modelling optional-dependency extras cannot coexist in one
environment (pyhealth pins numpy>=2.2, femr pins numpy<2), so CI installs one
extra per job and selects tests with -m eda / -m modelling. Driven by the
single directory->stack map below:

* every collected test is auto-tagged with its stack marker, so -m selects
  it without hand-tagging each module, and
* when exactly one stack is selected, the *other* stack's test tree is skipped
  before import. Markers only filter *after* collection, but collection
  imports the module, so a foreign import (e.g. pyhealth) would crash first
  without this guard.

Unclassified tests fail on collection.
"""

from pathlib import Path

import pytest

# Top-level directory under tests/ -> the stack (extra) it belongs to.
_STACK_BY_DIR = {
    "data": "eda",
    "feature_engineering": "eda",
    "modelling": "modelling",
}
_STACKS = frozenset(_STACK_BY_DIR.values())
_TESTS_ROOT = Path(__file__).parent.resolve()


def _stack_for(path: Path) -> str | None:
    """Returns the stack owning path, or 'None' if the map doesn't resolve."""
    try:
        rel = path.resolve().relative_to(_TESTS_ROOT)
    except ValueError:
        return None
    return _STACK_BY_DIR.get(rel.parts[0]) if rel.parts else None


def _selected_stack(config: pytest.Config) -> str | None:
    """Returns the stack named by -m, or None if not exactly one."""
    markexpr = config.getoption("markexpr") or ""
    named = [stack for stack in _STACKS if stack in markexpr]
    return named[0] if len(named) == 1 else None


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Skips the other stack's tree so its imports never run in a foreign env."""
    selected = _selected_stack(config)
    if selected is None:
        return None
    stack = _stack_for(collection_path)
    return stack is not None and stack != selected


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-tags each test with its stack marker; errors on unclassified tests."""
    unclassified: set[str] = set()
    for item in items:
        stack = _stack_for(Path(item.path))
        if stack is None:
            unclassified.add(str(Path(item.path).resolve().relative_to(_TESTS_ROOT)))
            continue
        item.add_marker(getattr(pytest.mark, stack))
    if unclassified:
        raise pytest.UsageError(
            "Unclassified test modules (add their top-level dir to _STACK_BY_DIR "
            f"in tests/conftest.py): {sorted(unclassified)}"
        )
