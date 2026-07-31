"""Test suite for run_build_subject_split."""

from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest
from conftest import SENTINELS, FakeSubjectDatabase

from thesis.modelling.cohort.split import run_build_subject_split

ONE_FOLD = {"train": 1.0}
SEVENTY_FIFTEEN_FIFTEEN = {"train": 0.70, "val": 0.15, "test": 0.15}
COHORT = {1: [10, 11], 2: [12], 3: [13], 99: [14]}


def _strata_by_subject(folds: pl.DataFrame) -> dict[int, str]:
    """Reduces a result to ``{subject_id: stratum}``."""
    return dict(zip(folds["subject_id"], folds["stratum"], strict=True))


def test_writes_one_row_per_database_subject(
    make_database: Callable, make_events: Callable, make_labels: Callable, dest: Path
) -> None:
    """Asserts the schema, the row count and the returned path."""
    written = run_build_subject_split(
        make_database(),
        dest,
        make_events(COHORT),
        make_labels([10]),
        fractions=ONE_FOLD,
        db_opener=FakeSubjectDatabase([1, 2, 3, 4]),
    )

    folds = pl.read_parquet(written)
    assert written == dest
    assert folds.columns == ["subject_id", "stratum", "fold"]
    assert dict(folds.schema) == {
        "subject_id": pl.Int64,
        "stratum": pl.String,
        "fold": pl.String,
    }
    assert folds["subject_id"].to_list() == [1, 2, 3, 4]
    assert folds["fold"].null_count() == 0


def test_universe_is_the_database_not_the_events(
    make_database: Callable, make_events: Callable, make_labels: Callable, dest: Path
) -> None:
    """Asserts subjects are taken from the database, in both directions.

    Subject 4 holds no admission and so appears in no shard, but still needs a
    fold; subject 99 holds admissions but is absent from the database, so it
    cannot be modelled and must not acquire one.
    """
    folds = pl.read_parquet(
        run_build_subject_split(
            make_database(),
            dest,
            make_events(COHORT),
            make_labels([10]),
            fractions=ONE_FOLD,
            db_opener=FakeSubjectDatabase([1, 2, 3, 4]),
        )
    )
    assert folds["subject_id"].to_list() == [1, 2, 3, 4]


def test_strata_cross_the_label_with_the_admission_count(
    make_database: Callable, make_events: Callable, make_labels: Callable, dest: Path
) -> None:
    """Asserts the frame the folds are cut on carries through to the output."""
    folds = pl.read_parquet(
        run_build_subject_split(
            make_database(),
            dest,
            make_events(COHORT),
            make_labels([10]),
            fractions=ONE_FOLD,
            db_opener=FakeSubjectDatabase([1, 2, 3, 4]),
        )
    )
    assert _strata_by_subject(folds) == {
        1: "positive_2-3",
        2: "negative_1",
        3: "negative_1",
        4: "negative_0",
    }


def test_reads_every_shard(
    make_database: Callable, make_events: Callable, make_labels: Callable, dest: Path
) -> None:
    """Asserts a subject's admissions are counted across the whole folder.

    Split over two shards, subject 1's four admissions land in different files;
    reading one shard would band it as 2-3 rather than 4+.
    """
    folds = pl.read_parquet(
        run_build_subject_split(
            make_database(),
            dest,
            make_events({1: [10, 11, 12, 13]}, n_shards=2),
            make_labels([10]),
            fractions=ONE_FOLD,
            db_opener=FakeSubjectDatabase([1]),
        )
    )
    assert _strata_by_subject(folds) == {1: "positive_4+"}


def test_opens_the_database_at_db_path_and_closes_it(
    make_database: Callable, make_events: Callable, make_labels: Callable, dest: Path
) -> None:
    """Asserts the database is opened by path string and released again.

    meds_reader takes a str, not a Path, and holds the store open until exit.
    """
    database = FakeSubjectDatabase([1])
    db_path = make_database()

    run_build_subject_split(
        db_path,
        dest,
        make_events(),
        make_labels(),
        fractions=ONE_FOLD,
        db_opener=database,
    )

    assert database.paths == [str(db_path)]
    assert database.closed


def test_creates_the_destination_parent(
    make_database: Callable, make_events: Callable, make_labels: Callable, dest: Path
) -> None:
    """Asserts a dest below a new folder is written rather than failing late."""
    assert not dest.parent.exists()

    run_build_subject_split(
        make_database(),
        dest,
        make_events(),
        make_labels(),
        fractions=ONE_FOLD,
        db_opener=FakeSubjectDatabase([1]),
    )

    assert dest.is_file()


def test_passes_the_fractions_through(
    make_database: Callable, make_events: Callable, make_labels: Callable, dest: Path
) -> None:
    """Asserts the fold names and sizes are the caller's, not the defaults."""
    folds = pl.read_parquet(
        run_build_subject_split(
            make_database(),
            dest,
            make_events({subject: [subject] for subject in range(1, 21)}),
            make_labels([]),
            fractions=SEVENTY_FIFTEEN_FIFTEEN,
            db_opener=FakeSubjectDatabase(list(range(1, 21))),
        )
    )
    assert dict(folds["fold"].value_counts().iter_rows()) == {
        "train": 14,
        "val": 3,
        "test": 3,
    }


def test_passes_the_seed_through(
    make_database: Callable, make_events: Callable, make_labels: Callable, dest: Path
) -> None:
    """Asserts the seed reaches the hash, so a rerun can be reseeded."""
    subjects = list(range(1, 41))
    admissions = {subject: [subject] for subject in subjects}
    written = []
    for seed in (42, 43):
        run_build_subject_split(
            make_database(name=f"reader_db_{seed}"),
            dest.with_name(f"folds_{seed}.parquet"),
            make_events(admissions, name=f"data_{seed}"),
            make_labels([], name=f"labels_{seed}"),
            fractions=SEVENTY_FIFTEEN_FIFTEEN,
            seed=seed,
            db_opener=FakeSubjectDatabase(subjects),
        )
        written.append(pl.read_parquet(dest.with_name(f"folds_{seed}.parquet")))

    assert not written[0].equals(written[1])
    assert written[0]["subject_id"].equals(written[1]["subject_id"])


def test_refuses_an_existing_dest(
    make_database: Callable, make_events: Callable, make_labels: Callable, dest: Path
) -> None:
    """Asserts a split is never silently overwritten."""
    dest.parent.mkdir()
    dest.touch()

    with pytest.raises(FileExistsError):
        run_build_subject_split(
            make_database(),
            dest,
            make_events(),
            make_labels(),
            fractions=ONE_FOLD,
            db_opener=FakeSubjectDatabase([1]),
        )


@pytest.mark.parametrize("missing", ["absent", "a_file"])
def test_rejects_a_db_path_that_is_not_a_directory(
    make_events: Callable,
    make_labels: Callable,
    dest: Path,
    tmp_path: Path,
    missing: str,
) -> None:
    """Asserts a path that cannot be a database is refused before the open."""
    db_path = tmp_path / missing
    if missing == "a_file":
        db_path.touch()

    with pytest.raises(FileNotFoundError):
        run_build_subject_split(
            db_path,
            dest,
            make_events(),
            make_labels(),
            fractions=ONE_FOLD,
            db_opener=FakeSubjectDatabase([1]),
        )


@pytest.mark.parametrize("sentinel", SENTINELS)
def test_rejects_a_directory_missing_a_sentinel(
    make_database: Callable,
    make_events: Callable,
    make_labels: Callable,
    dest: Path,
    sentinel: str,
) -> None:
    """Asserts a directory that is not a meds_reader database is named as such.

    meds_reader itself raises a bare RuntimeError about a missing file size,
    after mapping the store, which does not say what the caller got wrong.
    """
    with pytest.raises(FileNotFoundError, match=sentinel):
        run_build_subject_split(
            make_database(omit=sentinel),
            dest,
            make_events(),
            make_labels(),
            fractions=ONE_FOLD,
            db_opener=FakeSubjectDatabase([1]),
        )


def test_rejects_missing_labels(
    make_database: Callable, make_events: Callable, dest: Path, tmp_path: Path
) -> None:
    """Asserts a missing labels file is caught by the guard.

    scan_parquet is lazy and accepts a path that does not exist, so without the
    guard this surfaces only at collect, after the whole shard scan.
    """
    with pytest.raises(FileNotFoundError):
        run_build_subject_split(
            make_database(),
            dest,
            make_events(),
            tmp_path / "no_such_labels.parquet",
            fractions=ONE_FOLD,
            db_opener=FakeSubjectDatabase([1]),
        )


@pytest.mark.parametrize("populate", [False, True])
def test_rejects_a_shard_folder_holding_no_parquet(
    make_database: Callable,
    make_labels: Callable,
    dest: Path,
    tmp_path: Path,
    populate: bool,
) -> None:
    """Asserts an empty folder, and one holding no parquet, are both refused."""
    admissions = tmp_path / "data"
    admissions.mkdir()
    if populate:
        (admissions / "shard.csv").touch()

    with pytest.raises(FileNotFoundError):
        run_build_subject_split(
            make_database(),
            dest,
            admissions,
            make_labels(),
            fractions=ONE_FOLD,
            db_opener=FakeSubjectDatabase([1]),
        )


def test_writes_nothing_when_a_guard_trips(
    make_database: Callable, make_events: Callable, dest: Path, tmp_path: Path
) -> None:
    """Asserts a refused run leaves no half-built artefact behind."""
    with pytest.raises(FileNotFoundError):
        run_build_subject_split(
            make_database(),
            dest,
            make_events(),
            tmp_path / "no_such_labels.parquet",
            fractions=ONE_FOLD,
            db_opener=FakeSubjectDatabase([1]),
        )

    assert not dest.exists()
    assert not dest.parent.exists()


def test_rejects_invalid_fractions(
    make_database: Callable, make_events: Callable, make_labels: Callable, dest: Path
) -> None:
    """Asserts assign_folds' validation is reached before anything is written."""
    with pytest.raises(ValueError):
        run_build_subject_split(
            make_database(),
            dest,
            make_events(),
            make_labels(),
            fractions={"train": 0.70, "val": 0.15},
            db_opener=FakeSubjectDatabase([1]),
        )

    assert not dest.exists()
