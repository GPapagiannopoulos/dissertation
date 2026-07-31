"""Test suite for assign_folds."""

from collections.abc import Callable

import polars as pl
import pytest

from thesis.modelling.cohort.split import assign_folds

SEVENTY_FIFTEEN_FIFTEEN = {"train": 0.70, "val": 0.15, "test": 0.15}


@pytest.mark.parametrize(
    "sizes, fractions, expected_counts",
    [
        # 0. A stratum of 20 divides cleanly into 14/3/3.
        (
            {"positive_1": 20},
            SEVENTY_FIFTEEN_FIFTEEN,
            {
                ("positive_1", "train"): 14,
                ("positive_1", "val"): 3,
                ("positive_1", "test"): 3,
            },
        ),
        # 1. Every stratum is cut on its own size, so a small one is not
        #    swamped by a large one sharing the frame.
        (
            {"positive_1": 20, "negative_4+": 100},
            SEVENTY_FIFTEEN_FIFTEEN,
            {
                ("positive_1", "train"): 14,
                ("positive_1", "val"): 3,
                ("positive_1", "test"): 3,
                ("negative_4+", "train"): 70,
                ("negative_4+", "val"): 15,
                ("negative_4+", "test"): 15,
            },
        ),
        # 2. Two folds rather than three.
        (
            {"positive_1": 10},
            {"train": 0.80, "test": 0.20},
            {("positive_1", "train"): 8, ("positive_1", "test"): 2},
        ),
        # 3. Four equal folds, for a cross-validation style split.
        (
            {"positive_1": 20},
            {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
            {
                ("positive_1", "a"): 5,
                ("positive_1", "b"): 5,
                ("positive_1", "c"): 5,
                ("positive_1", "d"): 5,
            },
        ),
        # 4. A stratum holding one subject gives it to the first fold.
        (
            {"positive_1": 1},
            SEVENTY_FIFTEEN_FIFTEEN,
            {("positive_1", "train"): 1},
        ),
    ],
)
def test_assign_folds_cuts_each_stratum_to_the_fractions(
    make_strata: Callable,
    fold_counts: Callable,
    sizes: dict[str, int],
    fractions: dict[str, float],
    expected_counts: dict[tuple[str, str], int],
) -> None:
    """Asserts each stratum is divided in the requested proportions."""
    folds = assign_folds(make_strata(sizes), fractions, seed=42)
    assert fold_counts(folds) == expected_counts


def test_assign_folds_boundary_is_left_closed(
    make_strata: Callable, fold_counts: Callable
) -> None:
    """Asserts a subject landing exactly on a break goes to the later fold.

    Ten subjects rank at 0.0, 0.1 ... 0.9 and the breaks fall at 0.70 and 0.85.
    The subject at 0.70 belongs to val rather than train and the one at 0.85 to
    test rather than val, which is 7/2/1. Right-closed intervals would give
    8/2/0 and quietly starve the last fold.
    """
    folds = assign_folds(make_strata({"positive_1": 10}), SEVENTY_FIFTEEN_FIFTEEN, 42)
    assert fold_counts(folds) == {
        ("positive_1", "train"): 7,
        ("positive_1", "val"): 2,
        ("positive_1", "test"): 1,
    }


def test_assign_folds_is_deterministic(make_strata: Callable) -> None:
    """Asserts the same strata and seed always give the same folds."""
    strata = make_strata({"positive_1": 200, "negative_1": 200})
    first = assign_folds(strata, SEVENTY_FIFTEEN_FIFTEEN, seed=42)
    second = assign_folds(strata, SEVENTY_FIFTEEN_FIFTEEN, seed=42)
    assert first.equals(second)


def test_assign_folds_ignores_input_row_order(make_strata: Callable) -> None:
    """Asserts the assignment depends on the subject, not on where its row sits.

    This is the property hashing is chosen for. A shuffle-based split would
    reassign everyone the moment the database was rebuilt in a different order.
    """
    strata = make_strata({"positive_1": 200, "negative_1": 200})
    shuffled = strata.sample(fraction=1.0, shuffle=True, seed=9)
    assert assign_folds(strata, SEVENTY_FIFTEEN_FIFTEEN, seed=42).equals(
        assign_folds(shuffled, SEVENTY_FIFTEEN_FIFTEEN, seed=42)
    )


def test_assign_folds_seed_changes_the_assignment(make_strata: Callable) -> None:
    """Asserts a different seed reshuffles subjects between folds.

    The stability-across-seeds axis needs genuinely independent splits, so two
    seeds agreeing everywhere would be a defect rather than a coincidence.
    """
    strata = make_strata({"positive_1": 200, "negative_1": 200})
    first = assign_folds(strata, SEVENTY_FIFTEEN_FIFTEEN, seed=42)
    second = assign_folds(strata, SEVENTY_FIFTEEN_FIFTEEN, seed=43)
    assert not first.equals(second)
    assert first["subject_id"].equals(second["subject_id"])


def test_assign_folds_returns_one_row_per_subject(make_strata: Callable) -> None:
    """Asserts the schema, the row count and the ordering of the result."""
    strata = make_strata({"positive_1": 20, "negative_4+": 100})
    folds = assign_folds(strata, SEVENTY_FIFTEEN_FIFTEEN, seed=42)

    assert folds.columns == ["subject_id", "stratum", "fold"]
    assert dict(folds.schema) == {
        "subject_id": pl.Int64,
        "stratum": pl.String,
        "fold": pl.String,
    }
    assert folds.height == strata.height
    assert folds["subject_id"].n_unique() == strata.height
    assert folds["subject_id"].equals(strata["subject_id"].sort())
    assert folds["fold"].null_count() == 0


def test_assign_folds_takes_its_fold_names_from_the_keys(
    make_strata: Callable,
) -> None:
    """Asserts fold names are data, not a convention baked into the function."""
    folds = assign_folds(
        make_strata({"positive_1": 20}),
        {"development": 0.70, "tuning": 0.15, "holdout": 0.15},
        seed=42,
    )
    assert sorted(folds["fold"].unique().to_list()) == [
        "development",
        "holdout",
        "tuning",
    ]


@pytest.mark.parametrize(
    "fractions",
    [
        pytest.param({"train": 0.70, "val": 0.15}, id="sums_below_one"),
        pytest.param({"train": 0.70, "val": 0.15, "test": 0.25}, id="sums_above_one"),
        pytest.param({"train": 1.5, "val": -0.5}, id="negative_but_sums_to_one"),
    ],
)
def test_assign_folds_rejects_invalid_fractions(
    make_strata: Callable, fractions: dict[str, float]
) -> None:
    """Asserts fractions are validated before any subject is assigned.

    The negative case sums to 1.0 and so passes the total check, but produces
    non-monotonic breaks; without its own guard it would cut silently and wrong.
    """
    with pytest.raises(ValueError):
        assign_folds(make_strata(), fractions, seed=42)
