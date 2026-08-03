"""Tests for attaching the AKI onset to the admission it was diagnosed in."""

from collections.abc import Callable
from datetime import datetime

import polars as pl

from thesis.modelling.cohort.labeller import join_diagnosis_time


def test_join_diagnosis_time_attaches_the_onset_to_its_admission(
    make_admission_windows: Callable, make_diagnosis_labels: Callable
) -> None:
    """The label's timestamp becomes diagtime, in the unit the windows use.

    The label pipeline writes Datetime('ns') and the MEDS events carry
    Datetime('us'), so the cast is what makes the two comparable on the terms
    this module declares rather than on Polars' supertype rules.
    """
    joined = join_diagnosis_time(
        make_admission_windows(), make_diagnosis_labels()
    ).collect()

    assert joined.filter(pl.col("visit_id") == 10)["diagtime"].to_list() == [
        datetime(2020, 1, 4)
    ]
    assert joined.schema["diagtime"] == pl.Datetime("us")


def test_join_diagnosis_time_leaves_unlabelled_admissions_null(
    make_admission_windows: Callable, make_diagnosis_labels: Callable
) -> None:
    """A negative admission carries a null onset, never a dropped row.

    build_landmark_grid leans on that null: pl.min_horizontal ignores it and
    censors the admission at its discharge instead.
    """
    joined = join_diagnosis_time(
        make_admission_windows(), make_diagnosis_labels()
    ).collect()

    assert joined.filter(pl.col("visit_id") != 10)["diagtime"].to_list() == [None] * 3


def test_join_diagnosis_time_keeps_every_admission(
    make_admission_windows: Callable, make_diagnosis_labels: Callable
) -> None:
    """The join annotates the cohort; it must never narrow it."""
    windows = make_admission_windows()

    joined = join_diagnosis_time(windows, make_diagnosis_labels()).collect()

    assert joined.height == windows.collect().height
    assert joined["visit_id"].to_list() == [10, 20, 30, 40]


def test_join_diagnosis_time_emits_only_the_window_columns_and_diagtime(
    make_admission_windows: Callable,
    make_diagnosis_labels: Callable,
    onset_window_schema: dict[str, pl.DataType],
) -> None:
    """event_type, the diagnosis string and n_surviving_events stay behind.

    build_landmark_grid explodes this frame to roughly 2.97M rows, so every
    column that rides along is paid for three million times over, and all three
    are null on the negatives anyway.
    """
    joined = join_diagnosis_time(make_admission_windows(), make_diagnosis_labels())

    assert joined.collect_schema() == pl.Schema(onset_window_schema)


def test_join_diagnosis_time_matches_on_subject_as_well_as_visit(
    make_admission_windows: Callable, make_diagnosis_labels: Callable
) -> None:
    """A label whose subject disagrees with the window does not attach.

    MIMIC-IV reuses hadm_id 25029676 across two subjects, so visit_id alone is
    not a key. On both columns a mismatch loses the label; on visit_id alone it
    would pin a diagnosis to the wrong patient's admission.
    """
    labels = make_diagnosis_labels(subject_id=[999])

    joined = join_diagnosis_time(make_admission_windows(), labels).collect()

    assert joined["diagtime"].to_list() == [None] * 4


def test_join_diagnosis_time_stays_lazy(
    make_admission_windows: Callable, make_diagnosis_labels: Callable
) -> None:
    """The whole of stage 4 is one plan the run_* wrapper collects once."""
    assert isinstance(
        join_diagnosis_time(make_admission_windows(), make_diagnosis_labels()),
        pl.LazyFrame,
    )
