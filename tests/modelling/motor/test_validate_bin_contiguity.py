"""Testing suite for the value-bin contiguity guard.

This is the one check standing between a malformed bin table and a silently wrong
embedding. Leaf assignment resolves a value with a backward `join_asof` on
`val_start`, which never reads `val_end`, so a gap between two bins is not a miss --
it is absorbed by the bin below and the event is embedded as a neighbouring concept.
The frame stays well formed, every index stays inside the table, and nothing
downstream can tell. These assert the guard fires on the shapes that would cause it
and stays quiet on the shapes the released dictionary actually holds.
"""

from collections.abc import Callable

import polars as pl
import pytest

from thesis.modelling.motor.tokenizer import _validate_bin_contiguity

# the open ends of a code's bins, C's DBL_MAX, as the dictionary stores them
LOWEST, HIGHEST = -1.7976931348623157e308, 1.7976931348623157e308

NUMERIC_SCHEMA = {
    "code": pl.String,
    "val_start": pl.Float64,
    "val_end": pl.Float64,
    "numeric_indices": pl.UInt32,
}


def _bins(*edges: tuple[float | None, float | None], code: str = "LOINC/lab"):
    """Builds a bin table from explicit (start, end) pairs.

    The `make_bins` fixture derives its edges from cut points and so can only express
    a partition; the malformed tables below have to be written out edge by edge.
    """
    return pl.DataFrame(
        {
            "code": [code] * len(edges),
            "val_start": [start for start, _ in edges],
            "val_end": [end for _, end in edges],
            "numeric_indices": range(len(edges)),
        },
        schema=NUMERIC_SCHEMA,
    )


def test_accepts_bins_that_partition_the_line(make_bins: Callable) -> None:
    """The shape every code in the released dictionary holds."""
    _validate_bin_contiguity(make_bins("LOINC/lab", [0.6, 1.03, 2.4]))


def test_accepts_each_code_on_its_own_terms(make_bins: Callable) -> None:
    """Bins are contiguous within a code, never across the vocabulary.

    Sorted globally, one code's closing bin sits beside the next code's opening one
    and the two share no edge. A check that forgot to group would read that seam as a
    gap and reject the real dictionary outright.
    """
    table = pl.concat(
        [make_bins("LOINC/a", [1.0]), make_bins("LOINC/b", [5.0], first_index=2)]
    )

    _validate_bin_contiguity(table)


def test_accepts_a_lone_bin(make_bins: Callable) -> None:
    """A code with no cut points owns one bin and so owns no seam to check."""
    _validate_bin_contiguity(make_bins("LOINC/lab", []))


def test_accepts_a_vocabulary_holding_no_bins() -> None:
    """Every entry may be a code or text token; the guard must not read row zero."""
    _validate_bin_contiguity(_bins())


def test_accepts_bins_truncated_by_the_vocabulary_size() -> None:
    """`vocab_size` cuts a code's upper bins, which is not a fault.

    Bins occupy consecutive, ascending rollup rows, so the survivors are a prefix:
    still contiguous, merely closing early. Values above the cut reach no bin and
    fall back to the code's own token, which is the right answer -- the bin they
    wanted owns no embedding row.
    """
    _validate_bin_contiguity(_bins((LOWEST, 0.6), (0.6, 1.03)))


def test_accepts_bins_stored_out_of_order() -> None:
    """The guard sorts before it reads, as the loader does before it joins.

    Rollup order is the checkpoint's, not ours, so order is corrected rather than
    demanded.
    """
    _validate_bin_contiguity(_bins((1.0, HIGHEST), (LOWEST, 0.5), (0.5, 1.0)))


@pytest.mark.parametrize(
    "edges",
    [
        # 0. A gap: values in [1.0, 2.0) are absorbed by the bin below
        [(LOWEST, 1.0), (2.0, HIGHEST)],
        # 1. An overlap: values in [1.0, 2.0) belong to two bins at once
        [(LOWEST, 2.0), (1.0, HIGHEST)],
        # 2. A repeated bin: the join picks one of the two arbitrarily
        [(LOWEST, 1.0), (LOWEST, 1.0), (1.0, HIGHEST)],
        # 3. A missing edge: the seam cannot be verified, so it is not trusted
        [(LOWEST, None), (1.0, HIGHEST)],
        # 4. A gap in the middle of an otherwise sound partition
        [(LOWEST, 0.5), (0.5, 1.0), (1.5, 2.0), (2.0, HIGHEST)],
    ],
)
def test_rejects_bins_that_do_not_meet(
    edges: list[tuple[float | None, float | None]],
) -> None:
    """Asserts every way a seam can fail is refused at load time."""
    with pytest.raises(ValueError, match="do not meet edge to edge"):
        _validate_bin_contiguity(_bins(*edges))


def test_rejects_one_bad_code_among_sound_ones(make_bins: Callable) -> None:
    """One broken code is the realistic failure; the sound ones must not hide it."""
    table = pl.concat(
        [
            make_bins("LOINC/sound", [1.0]),
            _bins((LOWEST, 1.0), (2.0, HIGHEST), code="LOINC/broken"),
            make_bins("LOINC/also-sound", [5.0]),
        ]
    )

    with pytest.raises(ValueError) as failure:
        _validate_bin_contiguity(table)

    assert "LOINC/broken" in str(failure.value)
    assert "LOINC/sound" not in str(failure.value)


def test_reports_the_edges_that_disagree() -> None:
    """The real table is 11,113 rows; a bare refusal leaves nowhere to start looking."""
    with pytest.raises(ValueError) as failure:
        _validate_bin_contiguity(_bins((LOWEST, 1.0), (2.0, HIGHEST)))

    assert "closes a bin at 1.0 and opens the next at 2.0" in str(failure.value)


def test_counts_every_offending_code(make_bins: Callable) -> None:
    """One code may break at several seams; the count is of codes, not of seams."""
    table = pl.concat(
        [
            _bins((LOWEST, 0.5), (1.0, 1.5), (2.0, HIGHEST), code="LOINC/twice"),
            make_bins("LOINC/sound", [1.0]),
        ]
    )

    with pytest.raises(ValueError, match="^1 code"):
        _validate_bin_contiguity(table)
