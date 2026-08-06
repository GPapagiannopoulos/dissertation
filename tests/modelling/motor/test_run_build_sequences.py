"""Testing suite for the driver that materialises the sequences.

The guards are the point. This writes a dataset that later stages read blindly, and
its one genuinely global property -- that a sequence id means the same thing in every
shard -- is established by a running offset rather than by anything in the data. If
that offset were dropped, every shard would restart at zero, two subjects would share
an id, and the collate would pack their positions into one row of the batch.
"""

from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from thesis.modelling.motor.sequences import LABEL_SCHEMA, run_build_sequences

BIRTH = datetime(2100, 1, 1)
CODE = 0


@pytest.fixture
def layout(tmp_path: Path, make_dictionary: Callable, rollup_entry: Callable) -> dict:
    """A valid on-disk layout: two shards, labels, and a dictionary holding one code.

    Each shard carries two subjects of four events, so both shards produce sequences
    and the ids across them have to be reconciled.
    """
    events = tmp_path / "data"
    events.mkdir()
    for shard, subjects in enumerate(((1, 2), (3, 4))):
        rows = [
            {
                "subject_id": subject,
                "time": BIRTH + timedelta(days=day),
                "code": "MEDS_BIRTH" if day == 0 else "SNOMED/1",
                "numeric_value": None,
                "text_value": None,
            }
            for subject in subjects
            for day in range(4)
        ]
        pl.DataFrame(
            rows,
            schema_overrides={
                "subject_id": pl.Int64,
                "time": pl.Datetime("us"),
                "numeric_value": pl.Float32,
                "text_value": pl.String,
            },
        ).write_parquet(events / f"{shard}.parquet")

    labels = tmp_path / "labels.parquet"
    pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "prediction_time": [BIRTH + timedelta(days=2)] * 4,
            "boolean_value": [True, False, True, False],
        },
        schema_overrides={"subject_id": pl.Int64, "prediction_time": pl.Datetime("us")},
    ).write_parquet(labels)

    dictionary = make_dictionary(
        ontology_rollup=[rollup_entry("SNOMED/1", CODE)],
        age_stats={"mean": 1.0, "std": 2.0},
    )
    return {
        "events": events,
        "labels": labels,
        "dest": tmp_path / "out",
        "dictionary": dictionary,
        "vocab_size": 1,
        "length": 8,
        "stride": 4,
    }


def test_writes_a_sequence_and_label_file_per_shard(layout: dict) -> None:
    """Shard structure is preserved, so a crash leaves the earlier shards readable."""
    dest = run_build_sequences(**layout)

    assert sorted(p.name for p in (dest / "sequences").glob("*.parquet")) == [
        "0.parquet",
        "1.parquet",
    ]
    assert sorted(p.name for p in (dest / "labels").glob("*.parquet")) == [
        "0.parquet",
        "1.parquet",
    ]


def test_sequence_ids_do_not_collide_across_shards(layout: dict) -> None:
    """The one property no single shard can establish on its own.

    `build_sequences` numbers densely from zero per call, so without the running
    offset shard 1's ids would repeat shard 0's -- and the collate, which groups by
    id alone, would fold two subjects into one row of the batch.
    """
    dest = run_build_sequences(**layout)

    first = pl.read_parquet(dest / "sequences" / "0.parquet")["sequence_id"]
    second = pl.read_parquet(dest / "sequences" / "1.parquet")["sequence_id"]

    assert set(first.unique()).isdisjoint(set(second.unique()))
    assert int(second.min()) == int(first.max()) + 1


def test_labels_carry_the_offset_ids_too(layout: dict) -> None:
    """A label offset differently from its sequence would point into another subject."""
    dest = run_build_sequences(**layout)

    sequences = pl.read_parquet(dest / "sequences" / "*.parquet")
    labels = pl.read_parquet(dest / "labels" / "*.parquet")

    landed = labels.join(sequences, on=["sequence_id", "position"], how="inner")
    assert landed.height == labels.height


def test_places_every_label_it_can(layout: dict) -> None:
    """Four subjects, one landmark each, all after their subject's first event."""
    dest = run_build_sequences(**layout)

    labels = pl.read_parquet(dest / "labels" / "*.parquet")

    assert labels.height == 4
    assert dict(labels.schema) == LABEL_SCHEMA


def test_ages_are_measured_from_the_birth_row(layout: dict) -> None:
    """The origin comes off the raw shard, before tokenisation drops MEDS_BIRTH."""
    dest = run_build_sequences(**layout)

    sequences = pl.read_parquet(dest / "sequences" / "*.parquet")

    # MEDS_BIRTH itself owns no token, so day 1 is the first surviving event
    assert sequences["age"].min() == 1.0
    assert sequences["age"].null_count() == 0


def test_refuses_an_existing_destination(layout: dict) -> None:
    """Overwriting would mix two runs' sequence ids in one folder."""
    layout["dest"].mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        run_build_sequences(**layout)


def test_refuses_a_source_holding_no_shards(layout: dict, tmp_path: Path) -> None:
    """Pointed at the dataset root rather than its data/ folder, this finds nothing."""
    empty = tmp_path / "empty"
    empty.mkdir()
    layout["events"] = empty

    with pytest.raises(FileNotFoundError, match="no parquet shards"):
        run_build_sequences(**layout)


@pytest.mark.parametrize("missing", ["labels", "dictionary"])
def test_refuses_a_missing_input(layout: dict, tmp_path: Path, missing: str) -> None:
    """Checked before the loop: a mistyped path found after it costs the whole run."""
    layout[missing] = tmp_path / "absent"

    with pytest.raises(FileNotFoundError, match="Expected a file"):
        run_build_sequences(**layout)


def test_a_shard_with_no_labelled_subject_is_skipped(
    layout: dict, tmp_path: Path
) -> None:
    """Most shards hold some; one that holds none must not raise or shift the ids."""
    pl.DataFrame(
        {
            "subject_id": [1],
            "prediction_time": [BIRTH + timedelta(days=2)],
            "boolean_value": [True],
        },
        schema_overrides={"subject_id": pl.Int64, "prediction_time": pl.Datetime("us")},
    ).write_parquet(layout["labels"])

    dest = run_build_sequences(**layout)

    assert [p.name for p in (dest / "sequences").glob("*.parquet")] == ["0.parquet"]
    written = pl.read_parquet(dest / "sequences" / "0.parquet")
    assert written["subject_id"].unique().to_list() == [1]
