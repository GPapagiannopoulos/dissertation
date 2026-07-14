"""Caching module for the dashboard data."""

import json
from hashlib import sha256
from importlib import resources
from pathlib import Path
from typing import Final

import polars as pl
from pyhealth.datasets import MIMIC4Dataset

from thesis.config import settings
from thesis.data.sources import (
    cast_frame,
    cleanse_float_values,
    mimic4_add_descriptions_to_icd_codes,
    replace_mimic4_non_icd_codes,
)

# Bump when the transform pipeline changes
CACHE_VERSION: Final[int] = 1


def _fingerprint(pyhealth_cache_dir: Path) -> dict:
    """Fingerprints a cached version of the data.

    This function serves to "version" the cached data so that staleness
    is detectable. It uses hashing for configuration parameters and metadata
    for the dataset.

    Args:
        pyhealth_cache_dir (Path): Path to the cache directory.

    Returns:
        dict: a fingerprint defining the current version of the data.
    """
    source = pyhealth_cache_dir / "global_event_df.parquet"
    return {
        "version": CACHE_VERSION,
        "source_parts": sorted(
            [p.name, p.stat().st_size, p.stat().st_mtime_ns]
            for p in source.glob("*.parquet")
        ),
        "manifest": sha256(
            resources.files("thesis").joinpath("mimic4_ehr.yaml").read_bytes()
        ).hexdigest(),
        "mappings": {
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in [
                settings.mimic4_ehr_d_icd_diagnoses,
                settings.mimic4_ehr_d_icd_procedures,
                settings.mimic4_ehr_d_labitems,
            ]
        },
    }


def build_event_pipeline(ds: MIMIC4Dataset) -> pl.LazyFrame:
    """Lazily generates a transformation pipeline schema.

    Args:
        ds (MIMIC4Dataset): PyHealth's native MIMIC4Dataset loader object.

    Returns:
        LazyFrame: a lf containing the transformed data
    """
    float_fields = [
        col
        for col, dtype in settings.mimic4_ehr_dtype_mapping.items()
        if dtype == "Float"
    ]
    cleansed_df = cleanse_float_values(ds.global_event_df, float_fields)
    lf = cast_frame(cleansed_df, settings.mimic4_ehr_dtype_mapping)
    event_type_icd_maps: list[tuple[str, Path]] = [
        ("procedures_icd", settings.mimic4_ehr_d_icd_procedures),
        ("diagnoses_icd", settings.mimic4_ehr_d_icd_diagnoses),
    ]

    for event_type, mapping in event_type_icd_maps:
        lf = mimic4_add_descriptions_to_icd_codes(lf, mapping, event_type)

    lf = replace_mimic4_non_icd_codes(lf, settings.mimic4_ehr_d_labitems, "labevents")
    return lf


def sink_global_event_frame(lf: pl.LazyFrame, out: Path) -> None:
    """Sink the transformed MIMIC4Dataset into a parquet file.

    Uses the streaming engine to sink a parquet file. The file is marked
    .tmp until the sinking completes. If the process crashes then the file
    is clearly marked as .tmp and won't be used downstream.

    Args:
        lf (pl.LazyFrame): the lf to be sunk into a parquet file
        out (Path): path at which to sink the file.
    """
    tmp = out.with_name(out.name + ".tmp")
    lf.sink_parquet(tmp)
    tmp.rename(out)


def ensure_event_cache() -> Path:
    """Confirms whether a cached ds already exists."""
    ds = MIMIC4Dataset(
        ehr_root=settings.mimic4_ehr_data_path,
        dev=settings.mimic4_ehr_dev_mode,
        ehr_tables=settings.mimic4_ehr_tables,
    )
    cache_root = ds.cache_dir / "thesis_eda"
    parquet = cache_root / "events.parquet"
    sidecar = cache_root / "events.fingerprint.json"

    if parquet.exists() and sidecar.exists():
        if json.loads(sidecar.read_text()) == _fingerprint(ds.cache_dir):
            return parquet

    cache_root.mkdir(parents=True, exist_ok=True)
    sidecar.unlink(missing_ok=True)
    sink_global_event_frame(build_event_pipeline(ds), parquet)
    sidecar.write_text(json.dumps(_fingerprint(ds.cache_dir)))
    return parquet
