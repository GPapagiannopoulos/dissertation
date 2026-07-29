"""Testing suite for the transitive reduction of the ancestor closure."""

import pytest

from thesis.modelling.etl_pipeline.coverage import _immediate_parents

# A chain A <- B <- C, as the dictionary stores it: every entry is the flat set of
# ancestors, self included, so C lists B and A side by side with no depth to read off
CHAIN = {
    "A": ("A",),
    "B": ("B", "A"),
    "C": ("C", "B", "A"),
}

# A diamond: D has two parents, B and C, which share the grandparent A
DIAMOND = {
    "A": ("A",),
    "B": ("B", "A"),
    "C": ("C", "A"),
    "D": ("D", "B", "C", "A"),
}


@pytest.mark.parametrize(
    "all_parents, code, expected",
    [
        # 0. A root has no ancestor but itself
        (CHAIN, "A", ()),
        # 1. One step up the chain is the only parent there is
        (CHAIN, "B", ("A",)),
        # 2. The grandparent is dropped: A is reachable through B
        (CHAIN, "C", ("B",)),
        # 3. Both branches of a diamond are immediate, their shared root is not
        (DIAMOND, "D", ("B", "C")),
        # 4. A code whose ancestors are all mutually unreachable keeps every one
        ({"X": ("X", "P", "Q"), "P": ("P",), "Q": ("Q",)}, "X", ("P", "Q")),
        # 5. Depth is no defence: a two-hop grandparent is still redundant
        (
            {
                "W": ("W", "X", "Y", "Z"),
                "X": ("X", "Y", "Z"),
                "Y": ("Y", "Z"),
                "Z": ("Z",),
            },
            "W",
            ("X",),
        ),
    ],
)
def test_reduces_the_closure_to_its_nearest_layer(
    all_parents: dict[str, tuple[str, ...]], code: str, expected: tuple[str, ...]
) -> None:
    """An ancestor reachable through another ancestor is not an immediate parent."""
    assert _immediate_parents(code, all_parents) == expected


def test_excludes_the_code_itself() -> None:
    """Every entry carries a self-edge, which would make each code its own parent."""
    assert _immediate_parents("B", CHAIN) == ("A",)


def test_tolerates_a_mapping_without_self_edges() -> None:
    """The self filter is defensive: the reduction does not depend on that edge."""
    without_self = {"A": (), "B": ("A",), "C": ("B", "A")}

    assert _immediate_parents("C", without_self) == ("B",)


@pytest.mark.parametrize(
    "ancestors",
    [
        # 0. Already ordered
        ("D", "B", "C", "A"),
        # 1. Reversed
        ("A", "C", "B", "D"),
        # 2. Arbitrary
        ("C", "A", "D", "B"),
    ],
)
def test_orders_parents_deterministically(ancestors: tuple[str, ...]) -> None:
    """Ties on hops and weight are broken lexicographically, so order cannot drift."""
    all_parents = DIAMOND | {"D": ancestors}

    assert _immediate_parents("D", all_parents) == ("B", "C")
