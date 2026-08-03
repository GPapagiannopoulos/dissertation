"""Tests for the stage 4 wrapper, which composes the six helpers over the shards."""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from thesis.modelling.cohort.labeller import run_build_labels


def test_run_build_labels_writes_one_row_per_landmark(
    make_event_shards: Callable, make_surviving_labels: Callable, dest: Path
) -> None:
    """The grid is laid inside each admission and exploded to one row a landmark.

    Visit 10 is censored at its 2020-01-04 diagnosis and yields two landmarks;
    visit 20 is an unlabelled negative running to discharge and yields three.
    The order is asserted because the left join returns hash-probe order and
    the wrapper sorts it back; without that sort this artifact is not
    reproducible between runs.
    """
    run_build_labels(make_event_shards(), make_surviving_labels(), dest)

    assert pl.read_parquet(dest)["prediction_time"].to_list() == [
        datetime(2020, 1, 3, 8),
        datetime(2020, 1, 3, 20),
        datetime(2020, 2, 3, 7),
        datetime(2020, 2, 3, 19),
        datetime(2020, 2, 4, 7),
    ]


def test_run_build_labels_labels_each_landmark_against_the_diagnosis(
    make_event_shards: Callable, make_surviving_labels: Callable, dest: Path
) -> None:
    """Both of visit 10's landmarks reach the onset; the negative's never do."""
    run_build_labels(make_event_shards(), make_surviving_labels(), dest)

    written = pl.read_parquet(dest)

    assert written["boolean_value"].to_list() == [True, True, False, False, False]
    assert written["horizon_hours"].to_list() == [48.0, 48.0, 27.0, 15.0, 3.0]


def test_run_build_labels_keeps_only_inpatient_admissions(
    make_event_shards: Callable, make_surviving_labels: Callable, dest: Path
) -> None:
    """The ED-only and outpatient encounters never reach the grid.

    Their AKI rate is 0.2% against the inpatient 8.5%, so a hospital-acquired
    diagnosis on one is a coding artefact. Visits 30 and 40 are stretched past
    48h here on purpose: at the fixture's default length they are too short to
    produce a landmark at all, so the assertion would hold even with the cohort
    filter removed.
    """
    events = make_event_shards(
        end=[
            datetime(2020, 1, 6, 12),
            datetime(2020, 1, 2, 11),
            datetime(2020, 2, 4, 10),
            datetime(2020, 3, 6, 6),
            datetime(2020, 4, 6, 5),
        ]
    )

    run_build_labels(events, make_surviving_labels(), dest)

    assert pl.read_parquet(dest)["visit_id"].unique().to_list() == [10, 20]


def test_run_build_labels_drops_a_backwards_admission(
    make_event_shards: Callable, make_surviving_labels: Callable, dest: Path
) -> None:
    """An admission discharged before it began carries no window to lay a grid in.

    Defence in depth, and deliberately so: filter_false_admissions removes it,
    but a backwards window also makes datetime_ranges return an empty list,
    which the grid's own list.len() filter drops. Removing either mechanism
    leaves this passing, so the isolated test on filter_false_admissions is
    what pins that function.
    """
    events = make_event_shards(
        end=[
            datetime(2020, 1, 6, 12),
            datetime(2020, 1, 2, 11),
            datetime(2020, 1, 1, 7),
            datetime(2020, 3, 1, 20),
            datetime(2020, 4, 2, 5),
        ]
    )

    run_build_labels(events, make_surviving_labels(), dest)

    assert pl.read_parquet(dest)["visit_id"].unique().to_list() == [10]


def test_run_build_labels_reads_every_shard(
    make_event_shards: Callable, make_surviving_labels: Callable, dest: Path
) -> None:
    """The events arrive as 200 shards, so a single-shard read would lose most."""
    run_build_labels(make_event_shards(n_shards=3), make_surviving_labels(), dest)

    assert pl.read_parquet(dest).height == 5


def test_run_build_labels_passes_the_delta_through(
    make_event_shards: Callable, make_surviving_labels: Callable, dest: Path
) -> None:
    """Halving the spacing lays more landmarks in the same windows."""
    run_build_labels(
        make_event_shards(), make_surviving_labels(), dest, delta_hours="6h"
    )

    assert pl.read_parquet(dest).height == 8


def test_run_build_labels_passes_the_horizon_through(
    make_event_shards: Callable, make_surviving_labels: Callable, dest: Path
) -> None:
    """A horizon too short to reach the onset turns its landmark negative."""
    run_build_labels(
        make_event_shards(), make_surviving_labels(), dest, prediction_horizon="6h"
    )

    assert pl.read_parquet(dest)["boolean_value"].to_list() == [
        False,
        True,
        False,
        False,
        False,
    ]


def test_run_build_labels_creates_the_destination_parent(
    make_event_shards: Callable, make_surviving_labels: Callable, dest: Path
) -> None:
    """The driver names a folder that need not exist yet."""
    assert not dest.parent.exists()

    written = run_build_labels(make_event_shards(), make_surviving_labels(), dest)

    assert written == dest
    assert dest.is_file()


def test_run_build_labels_refuses_an_existing_dest(
    make_event_shards: Callable, make_surviving_labels: Callable, tmp_path: Path
) -> None:
    """A grid is expensive to rebuild and silently overwriting one loses it."""
    occupied = tmp_path / "already.parquet"
    occupied.touch()

    with pytest.raises(FileExistsError, match="already exists"):
        run_build_labels(make_event_shards(), make_surviving_labels(), occupied)


def test_run_build_labels_rejects_missing_labels(
    make_event_shards: Callable, tmp_path: Path, dest: Path
) -> None:
    """Without the positives every landmark would come out negative."""
    with pytest.raises(FileNotFoundError, match="Expected file"):
        run_build_labels(make_event_shards(), tmp_path / "absent.parquet", dest)


def test_run_build_labels_rejects_a_shard_folder_holding_no_parquet(
    make_surviving_labels: Callable, tmp_path: Path, dest: Path
) -> None:
    """Pointing at the dataset root rather than its data/ folder finds no shards."""
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="no parquet shards"):
        run_build_labels(empty, make_surviving_labels(), dest)


def test_run_build_labels_refuses_a_stranded_admission(
    make_event_shards: Callable, make_surviving_labels: Callable, dest: Path
) -> None:
    """An admission that kept no events through stage 2.6 cannot be featurized.

    Every one of the 34,625 positives survives today, so this guard is what
    would catch a concept map rebuilt with worse coverage.
    """
    labels = make_surviving_labels(n_surviving_events=[0.0])

    with pytest.raises(ValueError, match="kept no events"):
        run_build_labels(make_event_shards(), labels, dest)


def test_run_build_labels_writes_nothing_when_a_guard_trips(
    make_event_shards: Callable, make_surviving_labels: Callable, dest: Path
) -> None:
    """A refused run leaves no half-built grid for the next stage to pick up."""
    labels = make_surviving_labels(n_surviving_events=[0.0])

    with pytest.raises(ValueError):
        run_build_labels(make_event_shards(), labels, dest)

    assert not dest.exists()
    assert not dest.parent.exists()


def test_run_build_labels_reports_what_it_wrote(
    make_event_shards: Callable,
    make_surviving_labels: Callable,
    dest: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """The run is hand-driven, so the counts are the only sanity check it gets."""
    run_build_labels(make_event_shards(), make_surviving_labels(), dest)

    printed = capsys.readouterr().out

    assert "5 landmarks" in printed
    assert "2 admissions" in printed
    assert "2 subjects" in printed
    assert "2 positive" in printed
