"""Tests for stage 2.6, which rewrites the MEDS shards onto MOTOR's codes."""

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.modelling.etl_pipeline.normalise import (
    build_code_metadata,
    build_dataset_metadata,
    build_lookup,
    rewrite_codes,
    run_apply_concept_map,
)


def test_build_lookup_adds_identity_rows(make_concept_map: Callable) -> None:
    """The structural codes map to themselves, so the join cannot drop them."""
    lookup = build_lookup(make_concept_map(), structural=("MEDS_BIRTH",))

    assert_frame_equal(
        lookup.filter(pl.col("code") == "MEDS_BIRTH"),
        pl.DataFrame({"code": ["MEDS_BIRTH"], "vocab_code": ["MEDS_BIRTH"]}),
    )


def test_build_lookup_keeps_only_the_join_columns(make_concept_map: Callable) -> None:
    """Method and the rest stay in the map, never riding along into 226M events."""
    lookup = build_lookup(make_concept_map(method=["maps_to"] * 3))

    assert lookup.columns == ["code", "vocab_code"]


@pytest.mark.parametrize(
    ("codes", "structural"),
    [
        pytest.param(["SNOMED/1", "SNOMED/1"], (), id="duplicate_within_map"),
        pytest.param(
            ["MEDS_BIRTH", "SNOMED/1"], ("MEDS_BIRTH",), id="structural_now_mapped"
        ),
    ],
)
def test_build_lookup_refuses_duplicates(
    make_concept_map: Callable, codes: list[str], structural: tuple[str, ...]
) -> None:
    """A duplicated code would fan out the join, silently inflating the event count."""
    concept_map = make_concept_map(code=codes, vocab_code=["SNOMED/9"] * len(codes))

    with pytest.raises(ValueError, match="duplicated codes"):
        build_lookup(concept_map, structural=structural)


def test_rewrite_codes_swaps_the_code_and_keeps_the_original(
    make_concept_map: Callable,
) -> None:
    """The token replaces code in place; the MIMIC code survives as source_code."""
    lookup = build_lookup(make_concept_map(), structural=())
    events = pl.LazyFrame(
        {"subject_id": [1], "time": [datetime(2020, 1, 1)], "code": ["ICD10CM/A"]}
    )

    assert_frame_equal(
        rewrite_codes(events, lookup.lazy()).collect(),
        pl.DataFrame(
            {
                "subject_id": [1],
                "time": [datetime(2020, 1, 1)],
                "code": ["SNOMED/1"],
                "source_code": ["ICD10CM/A"],
            }
        ),
    )


@pytest.mark.parametrize(
    ("code", "kept"),
    [
        pytest.param("ICD10CM/A", True, id="mapped_code_kept"),
        pytest.param("MEDS_BIRTH", True, id="structural_code_kept"),
        pytest.param("MIMIC_IV_ITEM/9", False, id="unmapped_code_dropped"),
        pytest.param(None, False, id="null_code_dropped"),
    ],
)
def test_rewrite_codes_applies_the_drop_policy(
    make_concept_map: Callable, code: str | None, kept: bool
) -> None:
    """Only codes reaching a token, plus the structural exemptions, survive."""
    lookup = build_lookup(make_concept_map())
    events = pl.LazyFrame(
        {"subject_id": [1], "time": [datetime(2020, 1, 1)], "code": [code]},
        schema={"subject_id": pl.Int64, "time": pl.Datetime, "code": pl.String},
    )

    assert (rewrite_codes(events, lookup.lazy()).collect().height == 1) is kept


def test_rewrite_codes_keeps_duplicates(make_concept_map: Callable) -> None:
    """Two source codes climbing to one token stay two events: recording is signal."""
    lookup = build_lookup(make_concept_map())
    events = pl.LazyFrame(
        {
            "subject_id": [1, 1],
            "time": [datetime(2020, 1, 1)] * 2,
            "code": ["ICD10CM/A", "ICD10CM/B"],
        }
    )

    result = rewrite_codes(events, lookup.lazy()).collect()

    assert result.height == 2
    assert result["code"].to_list() == ["SNOMED/1", "SNOMED/1"]


def test_rewrite_codes_restores_the_meds_ordering(make_concept_map: Callable) -> None:
    """The join returns hash order; convert aborts on "Times are not in order"."""
    lookup = build_lookup(make_concept_map())
    events = pl.LazyFrame(
        {
            "subject_id": [2, 1, 2, 1],
            "time": [
                datetime(2020, 1, 2),
                datetime(2020, 1, 2),
                datetime(2020, 1, 1),
                datetime(2020, 1, 1),
            ],
            "code": ["ICD10CM/A", "ICD10CM/B", "ICD10CM/B", "ICD10CM/A"],
        }
    )

    result = rewrite_codes(events, lookup.lazy()).collect()

    assert result["subject_id"].to_list() == [1, 1, 2, 2]
    assert result["time"].to_list() == [
        datetime(2020, 1, 1),
        datetime(2020, 1, 2),
        datetime(2020, 1, 1),
        datetime(2020, 1, 2),
    ]


def test_rewrite_codes_keeps_the_birth_event_first(make_concept_map: Callable) -> None:
    """MEDS_BIRTH shares its timestamp with demographics and is the subject's origin."""
    lookup = build_lookup(make_concept_map())
    events = pl.LazyFrame(
        {
            "subject_id": [1, 1],
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 1)],
            "code": ["ICD10CM/A", "MEDS_BIRTH"],
        }
    )

    result = rewrite_codes(events, lookup.lazy()).collect()

    assert result["code"].to_list() == ["MEDS_BIRTH", "SNOMED/1"]


def test_build_code_metadata_describes_the_surviving_codes(
    make_concept_map: Callable, make_athena_concept: Callable
) -> None:
    """One row per token, described by Athena, with an always-empty ontology."""
    lookup = build_lookup(make_concept_map(), structural=("MEDS_BIRTH",))
    concept = pl.read_csv(make_athena_concept(), separator="\t").lazy()

    assert_frame_equal(
        build_code_metadata(lookup.lazy(), concept).collect(),
        pl.DataFrame(
            {
                "code": ["LOINC/2", "MEDS_BIRTH", "SNOMED/1"],
                "description": ["Creatinine", "MEDS_BIRTH", "Sepsis"],
                "parent_codes": [[], [], []],
            },
            schema={
                "code": pl.String,
                "description": pl.String,
                "parent_codes": pl.List(pl.String),
            },
        ),
    )


def test_build_dataset_metadata_corrects_the_version() -> None:
    """Stage 1 reads 2.2 off a symlink; the data has always been 3.1."""
    metadata = build_dataset_metadata()

    assert metadata["dataset_version"] == "3.1"
    assert set(metadata) == {
        "dataset_name",
        "dataset_version",
        "etl_name",
        "etl_version",
        "meds_version",
    }


def test_run_apply_concept_map_writes_the_meds_layout(
    tmp_path: Path,
    make_shard: Callable,
    make_concept_map: Callable,
    make_athena_concept: Callable,
) -> None:
    """The convert wants data/ and metadata/ under one root, and nothing else."""
    shard = make_shard()
    concept_map = tmp_path / "concept_map.parquet"
    make_concept_map().sink_parquet(concept_map)
    dest = tmp_path / "normalized"

    run_apply_concept_map(
        shard.parent,
        dest,
        concept_map=concept_map,
        concept=make_athena_concept(),
    )

    assert sorted(p.name for p in (dest / "data").iterdir()) == ["0.parquet"]
    assert sorted(p.name for p in (dest / "metadata").iterdir()) == [
        "codes.parquet",
        "dataset.json",
    ]
    assert (dest / "concept_map.parquet").is_file()

    metadata = json.loads((dest / "metadata" / "dataset.json").read_text())
    assert metadata["dataset_version"] == "3.1"


def test_run_apply_concept_map_keeps_the_provenance_copy_out_of_data(
    tmp_path: Path,
    make_shard: Callable,
    make_concept_map: Callable,
    make_athena_concept: Callable,
) -> None:
    """Convert globs data/*.parquet as events, so a map left there would poison it."""
    shard = make_shard()
    concept_map = tmp_path / "concept_map.parquet"
    make_concept_map().sink_parquet(concept_map)
    dest = tmp_path / "normalized"

    run_apply_concept_map(
        shard.parent, dest, concept_map=concept_map, concept=make_athena_concept()
    )

    assert not (dest / "data" / "concept_map.parquet").exists()


def test_run_apply_concept_map_preserves_the_shard_structure(
    tmp_path: Path,
    make_shard: Callable,
    make_concept_map: Callable,
    make_athena_concept: Callable,
) -> None:
    """One output per input, same name: it is what convert parallelises over."""
    shard = make_shard("0.parquet")
    make_shard("1.parquet")
    concept_map = tmp_path / "concept_map.parquet"
    make_concept_map().sink_parquet(concept_map)
    dest = tmp_path / "normalized"

    run_apply_concept_map(
        shard.parent, dest, concept_map=concept_map, concept=make_athena_concept()
    )

    assert sorted(p.name for p in (dest / "data").iterdir()) == [
        "0.parquet",
        "1.parquet",
    ]


def test_run_apply_concept_map_refuses_an_existing_dest(
    tmp_path: Path,
    make_shard: Callable,
    make_concept_map: Callable,
    make_athena_concept: Callable,
) -> None:
    """Writing into a populated root would mix stale shards with new ones unnoticed."""
    shard = make_shard()
    concept_map = tmp_path / "concept_map.parquet"
    make_concept_map().sink_parquet(concept_map)
    dest = tmp_path / "normalized"
    dest.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        run_apply_concept_map(
            shard.parent, dest, concept_map=concept_map, concept=make_athena_concept()
        )


@pytest.mark.parametrize(
    ("break_", "match"),
    [
        pytest.param("events", "no parquet shards", id="empty_events_folder"),
        pytest.param("concept_map", "Expected a file", id="missing_concept_map"),
        pytest.param("concept", "Expected a file", id="missing_athena_concept"),
    ],
)
def test_run_apply_concept_map_validates_inputs_before_the_loop(
    tmp_path: Path,
    make_shard: Callable,
    make_concept_map: Callable,
    make_athena_concept: Callable,
    break_: str,
    match: str,
) -> None:
    """A mistyped path must cost seconds, not the whole 200-shard run."""
    shard = make_shard()
    concept_map = tmp_path / "concept_map.parquet"
    make_concept_map().sink_parquet(concept_map)
    paths = {
        "events": shard.parent,
        "concept_map": concept_map,
        "concept": make_athena_concept(),
    }
    if break_ == "events":
        shard.unlink()
    else:
        paths[break_] = tmp_path / "absent"

    with pytest.raises(FileNotFoundError, match=match):
        run_apply_concept_map(
            paths["events"],
            tmp_path / "normalized",
            concept_map=paths["concept_map"],
            concept=paths["concept"],
        )
    assert not (tmp_path / "normalized").exists()
