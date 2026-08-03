"""Shared fixtures for the cohort splitting test suite."""

from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Self

import polars as pl
import pytest

SENTINELS = (
    "meds_reader.version",
    "meds_reader.properties",
    "meds_reader.length",
    "subject_id",
)


class FakeSubjectDatabase:
    """Stands in for meds_reader.SubjectDatabase, which needs a real 1.4 GB store.

    It is its own opener: calling it records the path and returns itself, so a
    test can assert both what was opened and that it was closed again.
    """

    def __init__(self, subject_ids: list[int]) -> None:
        """Holds the ids this database will hand out."""
        self.subject_ids = subject_ids
        self.paths: list[str] = []
        self.closed = False

    def __call__(self, path: str) -> Self:
        """Stands in for the constructor, recording what was opened."""
        self.paths.append(path)
        return self

    def __enter__(self) -> Self:
        """Enters the context the real database is used through."""
        return self

    def __exit__(self, *exc_info: object) -> bool:
        """Records the close and never suppresses an exception."""
        self.closed = True
        return False

    def __iter__(self) -> Iterator[int]:
        """Yields the subject ids, as SubjectDatabase does over its keys."""
        return iter(self.subject_ids)


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


@pytest.fixture
def make_database(tmp_path: Path) -> Callable:
    """Factory for a directory shaped like a meds_reader_convert output.

    Only the sentinel files the guard looks for are written; the events live in
    the parquet shards, never in here, so empty files are enough. ``omit`` drops
    one sentinel, which is how the rejection cases are driven.
    """

    def _build(omit: str | None = None, name: str = "reader_db") -> Path:
        db_path = tmp_path / name
        db_path.mkdir()
        for sentinel in SENTINELS:
            if sentinel != omit:
                (db_path / sentinel).touch()
        return db_path

    return _build


@pytest.fixture
def make_events(tmp_path: Path) -> Callable:
    """Factory writing normalised-shard-shaped parquet, returning the folder.

    Takes ``{subject_id: [visit_id, ...]}``, where a None visit stands for the
    events attributable to no admission. The code column is filler: it is one of
    the twenty columns the function must not read.
    """

    def _build(
        admissions: dict[int, list[int | None]] | None = None,
        n_shards: int = 1,
        name: str = "data",
    ) -> Path:
        admissions = {1: [10]} if admissions is None else admissions
        rows = [
            (subject_id, visit_id)
            for subject_id, visits in admissions.items()
            for visit_id in visits
        ]
        events = pl.DataFrame(
            {
                "subject_id": pl.Series([row[0] for row in rows], dtype=pl.Int64),
                "visit_id": pl.Series([row[1] for row in rows], dtype=pl.Int64),
                "code": pl.Series(["SNOMED/1"] * len(rows), dtype=pl.String),
            }
        )
        data_dir = tmp_path / name
        data_dir.mkdir()
        for shard, frame in enumerate(_chunk(events, n_shards)):
            frame.write_parquet(data_dir / f"{shard}.parquet")
        return data_dir

    return _build


def _chunk(frame: pl.DataFrame, n_shards: int) -> list[pl.DataFrame]:
    """Splits a frame into n_shards near-equal parts, dropping empty ones."""
    size = -(-frame.height // n_shards)
    return [part for part in frame.iter_slices(size) if part.height]


@pytest.fixture
def make_labels(tmp_path: Path) -> Callable:
    """Factory writing a positive-diagnosis-labels parquet, returning the path."""

    def _build(visit_ids: list[int] | None = None, name: str = "labels") -> Path:
        visit_ids = [10] if visit_ids is None else visit_ids
        labels = pl.DataFrame(
            {
                "subject_id": pl.Series([0] * len(visit_ids), dtype=pl.Int64),
                "visit_id": pl.Series(visit_ids, dtype=pl.Int64),
                "diagnosis_made/diagnosis": pl.Series(
                    ["aki"] * len(visit_ids), dtype=pl.String
                ),
            }
        )
        path = tmp_path / f"{name}.parquet"
        labels.write_parquet(path)
        return path

    return _build


@pytest.fixture
def dest(tmp_path: Path) -> Path:
    """The path to write to, whose parent deliberately does not exist yet."""
    return tmp_path / "splits" / "subject_folds.parquet"


@pytest.fixture
def make_normalised_events() -> Callable:
    """Returns a factory for a normalised-shard-shaped frame, one row per event.

    The default holds one event of each of the four visit types MIMIC-IV
    produces, plus one non-visit event that carries a non-null ``end``. That
    last row is the point of the fixture: ``end`` is populated on 2.1M
    inputevents and procedureevents rows, so a window filter written against
    ``end`` rather than ``code`` would pick it up. It shares visit 10 with the
    Visit/IP row, so such a filter also trips the duplicate guard.

    ``numeric_value`` and ``source_code`` stand in for the sixteen columns the
    function must project away.
    """

    def _make(**columns: list) -> pl.LazyFrame:
        defaults = {
            "subject_id": [1, 1, 2, 3, 4],
            "time": [
                datetime(2020, 1, 1, 8),
                datetime(2020, 1, 2, 9),
                datetime(2020, 2, 1, 7),
                datetime(2020, 3, 1, 6),
                datetime(2020, 4, 1, 5),
            ],
            "code": [
                "Visit/IP",
                "MIMIC_IV_INPUT/1",
                "Visit/ERIP",
                "Visit/ER",
                "Visit/OP",
            ],
            "end": [
                datetime(2020, 1, 6, 12),
                datetime(2020, 1, 2, 11),
                datetime(2020, 2, 4, 10),
                datetime(2020, 3, 1, 20),
                datetime(2020, 4, 2, 5),
            ],
            "visit_id": [10, 10, 20, 30, 40],
            "numeric_value": [None, 2.5, None, None, None],
            "source_code": ["EW EMER.", "221749", "EW EMER.", "URGENT", "AMB OBS"],
        }
        return pl.LazyFrame(
            defaults | columns,
            schema_overrides={
                "subject_id": pl.Int64,
                "time": pl.Datetime("us"),
                "code": pl.String,
                "end": pl.Datetime("us"),
                "visit_id": pl.Int64,
                "numeric_value": pl.Float32,
                "source_code": pl.String,
            },
        )

    return _make


@pytest.fixture
def window_schema() -> dict[str, pl.DataType]:
    """The schema build_admission_windows emits and both filters must preserve."""
    return {
        "subject_id": pl.Int64,
        "visit_id": pl.Int64,
        "visit_code": pl.String,
        "admittime": pl.Datetime("us"),
        "dischtime": pl.Datetime("us"),
    }


@pytest.fixture
def make_admission_windows(window_schema: dict[str, pl.DataType]) -> Callable:
    """Returns a factory for a build_admission_windows-shaped frame.

    The default is what build_admission_windows emits from the default
    make_normalised_events frame: one admission of each visit type, all four
    with a positive length of stay, so a case overrides only the one column it
    is about.
    """

    def _make(**columns: list) -> pl.LazyFrame:
        defaults = {
            "subject_id": [1, 2, 3, 4],
            "visit_id": [10, 20, 30, 40],
            "visit_code": ["Visit/IP", "Visit/ERIP", "Visit/ER", "Visit/OP"],
            "admittime": [
                datetime(2020, 1, 1, 8),
                datetime(2020, 2, 1, 7),
                datetime(2020, 3, 1, 6),
                datetime(2020, 4, 1, 5),
            ],
            "dischtime": [
                datetime(2020, 1, 6, 12),
                datetime(2020, 2, 4, 10),
                datetime(2020, 3, 1, 20),
                datetime(2020, 4, 2, 5),
            ],
        }
        return pl.LazyFrame(defaults | columns, schema_overrides=window_schema)

    return _make


@pytest.fixture
def make_diagnosis_labels() -> Callable:
    """Returns a factory shaped like surviving_aki_admissions.parquet.

    The timestamp is Datetime('ns') because that is what the label pipeline
    writes, against the Datetime('us') the MEDS events carry. event_type,
    diagnosis_made/diagnosis and n_surviving_events are the three columns the
    join must leave behind rather than carry into the grid explosion.
    """

    def _make(**columns: list) -> pl.LazyFrame:
        defaults = {
            "event_type": ["diagnosis_made"],
            "subject_id": [1],
            "visit_id": [10],
            "timestamp": [datetime(2020, 1, 4)],
            "diagnosis_made/diagnosis": ["Acute Kidney Injury"],
            "n_surviving_events": [615.0],
        }
        return pl.LazyFrame(
            defaults | columns,
            schema_overrides={
                "event_type": pl.String,
                "subject_id": pl.Int64,
                "visit_id": pl.Int64,
                "timestamp": pl.Datetime("ns"),
                "diagnosis_made/diagnosis": pl.String,
                "n_surviving_events": pl.Float64,
            },
        )

    return _make


@pytest.fixture
def onset_window_schema(
    window_schema: dict[str, pl.DataType],
) -> dict[str, pl.DataType]:
    """A window frame once the AKI onset has been joined onto it."""
    return window_schema | {"diagtime": pl.Datetime("us")}


@pytest.fixture
def make_windows_with_onset(
    onset_window_schema: dict[str, pl.DataType],
) -> Callable:
    """Returns a factory for one admission carrying an onset column.

    The default is a single negative admission on round hours: admitted
    2020-01-01 00:00, discharged five days later, no diagnosis. The grid then
    starts at 2020-01-03 00:00 and every landmark falls on a whole or half day,
    so a case can state its expectation as a literal list rather than compute
    one. ``diagtime`` defaults to null, which is what every negative admission
    carries after the label join.
    """

    def _make(**columns: list) -> pl.LazyFrame:
        defaults = {
            "subject_id": [1],
            "visit_id": [10],
            "visit_code": ["Visit/IP"],
            "admittime": [datetime(2020, 1, 1)],
            "dischtime": [datetime(2020, 1, 6)],
            "diagtime": [None],
        }
        return pl.LazyFrame(defaults | columns, schema_overrides=onset_window_schema)

    return _make
