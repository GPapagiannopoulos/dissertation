"""Shared fixtures for the cohort splitting test suite."""

from collections.abc import Callable

import polars as pl
import pytest


@pytest.fixture
def make_strata() -> Callable:
    """Factory for a build_subject_strata-shaped frame.

    Takes ``{stratum: size}`` and emits that many subjects per stratum with
    consecutive ids, which keeps the expected fold counts arithmetic rather than
    something the test has to look up.
    """

    def _build(sizes: dict[str, int] | None = None, first_id: int = 1) -> pl.DataFrame:
        sizes = {"positive_1": 20} if sizes is None else sizes
        subject_ids: list[int] = []
        strata: list[str] = []
        next_id = first_id
        for stratum, size in sizes.items():
            subject_ids.extend(range(next_id, next_id + size))
            strata.extend([stratum] * size)
            next_id += size
        return pl.DataFrame(
            {
                "subject_id": pl.Series(subject_ids, dtype=pl.Int64),
                "stratum": pl.Series(strata, dtype=pl.String),
            }
        )

    return _build


@pytest.fixture
def fold_counts() -> Callable:
    """Reduces an assign_folds result to ``{(stratum, fold): n}``.

    Which subject lands in which fold is a property of the hash and is not worth
    asserting; how many land in each cell is the contract.
    """

    def _count(folds: pl.DataFrame) -> dict[tuple[str, str], int]:
        counted = folds.group_by("stratum", "fold").len()
        return {
            (row["stratum"], row["fold"]): row["len"]
            for row in counted.iter_rows(named=True)
        }

    return _count
