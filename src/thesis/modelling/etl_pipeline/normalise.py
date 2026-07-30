"""Stage 2.6: rewrites the MEDS shards onto the codes MOTOR holds.

This module applies the mapping created in coverage.py to the events.
The concept map is a lookup of 44k rows, one per distinct code and the
events are ~680M rows. It is implemented as a join of the two.

Design choices:

1. An event whose code reaches no token is dropped. MOTOR cannot tokenise it, so
   carrying it forward only inflates the sequences.
2. The codes in STRUCTURAL_CODES are never dropped. They carry no concept and so
   appear in no concept map, but MEDS_BIRTH is every subject's time origin and
   MEDS_DEATH is a label source. They enter the lookup as identity rows, which keeps
   the exemption inside the data the join reads instead of in a second branch.
3. Duplicates are deliberately kept. Several source codes climb to one token, so a
   subject can carry the same token twice at one timestamp. The repetition is
   a useful signal
"""

import json
import shutil
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from thesis.modelling.etl_pipeline.coverage import scan_athena

STRUCTURAL_CODES: tuple[str, ...] = ("MEDS_BIRTH", "MEDS_DEATH")

_BIRTH_CODE = "MEDS_BIRTH"

_DATA_SUBDIR = "data"
_METADATA_SUBDIR = "metadata"
_DATASET_NAME = "MIMIC-IV"
_DATASET_VERSION = "3.1"
_ETL_NAME = "meds_etl.mimic + thesis.normalise"
_ETL_VERSION = "0.3.11"
_MEDS_VERSION = "0.3.3"


def build_lookup(
    concept_map: pl.LazyFrame, *, structural: Sequence[str] = STRUCTURAL_CODES
) -> pl.DataFrame:
    """Assembles the table the rewrite joins against.

    Args:
        concept_map: build_concept_map's output
        structural: codes exempt from the drop, mapped to themselves

    Returns:
        pl.DataFrame: code (String) and vocab_code (String), one row per code

    Raises:
        ValueError: if any code appears twice
    """
    identity = pl.LazyFrame(
        {"code": list(structural), "vocab_code": list(structural)},
        schema={"code": pl.String, "vocab_code": pl.String},
    )
    lookup = pl.concat([concept_map.select("code", "vocab_code"), identity]).collect(
        engine="streaming"
    )

    if lookup["code"].n_unique() != lookup.height:
        duplicated = (
            lookup.group_by("code")
            .len()
            .filter(pl.col("len") > 1)
            .get_column("code")
            .to_list()
        )
        raise ValueError(
            f"The lookup holds {len(duplicated)} duplicated codes, which would fan "
            f"out the rewrite. First few: {duplicated[:5]}"
        )
    return lookup


def rewrite_codes(events: pl.LazyFrame, lookup: pl.LazyFrame) -> pl.LazyFrame:
    """Swaps each event's code for its token, dropping the events that have none.

    code is overwritten in place so the column order stage 1 produced survives, and
    the original is kept alongside as source_code, which makes the shards
    self-auditing: any token can be traced back to the MIMIC code it came from.
    The join is responsible for dropping codes that do not map to tokens.

    meds_reader_convert expects the timestamps in each shard to be in order. Polars
    joins on a hash table and returns rows in which order the probe side yields them.
    The sort at the end enforces the necessary ordering.

    Args:
        events: one stage 1 shard, needing subject_id, time and code columns
        lookup: build_lookup's output, as a LazyFrame

    Returns:
        pl.LazyFrame: the shard's schema plus source_code (String), sorted by
            subject_id then time, with MEDS_BIRTH first within its timestamp
    """
    return (
        events.join(lookup, on="code", how="inner")
        .with_columns(
            pl.col("code").alias("source_code"),
            pl.col("vocab_code").alias("code"),
        )
        .drop("vocab_code")
        .sort("subject_id", "time", pl.col("code") != _BIRTH_CODE)
    )


def build_code_metadata(lookup: pl.LazyFrame, concept: pl.LazyFrame) -> pl.LazyFrame:
    """Rebuilds metadata/codes.parquet on the codes the rewrite leaves behind.

    Args:
        lookup: build_lookup's output, whose vocab_code column is the new vocabulary
        concept: Athena CONCEPT.csv, opened with scan_athena

    Returns:
        pl.LazyFrame: sorted by code, matching stage 1's sidecar schema::

            code         (String)       the token as it appears in the events
            description  (String)       Athena's concept_name, else the code itself
            parent_codes (List(String)) always empty
    """
    descriptions = concept.select(
        code=pl.concat_str(
            [pl.col("vocabulary_id"), pl.col("concept_code")], separator="/"
        ),
        description=pl.col("concept_name"),
    ).unique(subset="code", keep="first")

    return (
        lookup.select(code=pl.col("vocab_code"))
        .unique()
        .join(descriptions, on="code", how="left")
        .with_columns(
            description=pl.col("description").fill_null(pl.col("code")),
            parent_codes=pl.lit([], dtype=pl.List(pl.String)),
        )
        .sort("code")
    )


def build_dataset_metadata(
    *,
    dataset_name: str = _DATASET_NAME,
    dataset_version: str = _DATASET_VERSION,
    etl_name: str = _ETL_NAME,
    etl_version: str = _ETL_VERSION,
    meds_version: str = _MEDS_VERSION,
) -> dict[str, str]:
    """Builds the contents of metadata/dataset.json as per meds.DatasetMetadata.

    Args:
        dataset_name: the source dataset
        dataset_version: the source dataset's release
        etl_name: the chain of ETLs that produced these shards
        etl_version: the meds_etl release that ran stage 1
        meds_version: the MEDS schema release the shards conform to

    Returns:
        dict[str, str]: the five keys, ready for json.dumps
    """
    return {
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "etl_name": etl_name,
        "etl_version": etl_version,
        "meds_version": meds_version,
    }


def run_apply_concept_map(
    events: Path,
    dest: Path,
    *,
    concept_map: Path,
    concept: Path,
    structural: Sequence[str] = STRUCTURAL_CODES,
    dataset_version: str = _DATASET_VERSION,
) -> Path:
    """Rewrites every shard and writes the MEDS dataset root meds_reader expects.

    Shards are processed one at a time. That keeps peak memory at one shard, preserves
    the shard structure meds_reader_convert parallelises over, and avoids total output
    corruption if any of the conversions fail.

    Args:
        events: Path to stage 1's shard folder, i.e. <base>/data
        dest: Path to the dataset root to create, which must not already exist
        concept_map: Path to concept_map.parquet
        concept: Path to Athena's CONCEPT.csv
        structural: codes exempt from the drop
        dataset_version: the source dataset's release, for dataset.json

    Returns:
        Path: the dataset root, ready to hand to run_meds_reader_convert

    Raises:
        FileExistsError: if dest already exists
        FileNotFoundError: if events holds no parquet, or an input file is missing
        ValueError: if the lookup holds duplicated codes
    """
    if dest.exists():
        raise FileExistsError(
            f"Destination {dest} already exists; refusing to overwrite. "
            f"Remove it or choose a new path."
        )
    shards = sorted(events.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"Found no parquet shards in {events}. Point events at stage 1's "
            f"data/ folder."
        )
    for path in (concept_map, concept):
        if not path.is_file():
            raise FileNotFoundError(f"Expected a file at {path}.")

    lookup = build_lookup(pl.scan_parquet(concept_map), structural=structural)
    print(f"lookup holds {lookup.height} codes")

    data_dir = dest / _DATA_SUBDIR
    metadata_dir = dest / _METADATA_SUBDIR
    data_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    (metadata_dir / "dataset.json").write_text(
        json.dumps(build_dataset_metadata(dataset_version=dataset_version), indent=2)
    )
    build_code_metadata(lookup.lazy(), scan_athena(concept)).sink_parquet(
        metadata_dir / "codes.parquet"
    )
    shutil.copy2(concept_map, dest / concept_map.name)
    print(f"metadata written to {metadata_dir}")

    lookup_lf = lookup.lazy()
    for position, shard in enumerate(shards, start=1):
        rewrite_codes(pl.scan_parquet(shard), lookup_lf).sink_parquet(
            data_dir / shard.name
        )
        print(f"  [{position}/{len(shards)}] {shard.name}", flush=True)

    return dest
