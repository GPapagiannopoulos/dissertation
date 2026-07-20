"""Caching module for the dashboard data."""

import json
import shutil
from hashlib import sha256
from importlib import resources
from pathlib import Path
from typing import Final

import polars as pl
from pyhealth.datasets import MIMIC4Dataset

from thesis.config import settings
from thesis.data.normalize import build_event_pipeline

# Bump when the transform pipeline in thesis.data.normalize changes
# Parsing the bytes of the repository was thought excessive
CACHE_VERSION: Final[int] = 2


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


def sink_atomic(lf: pl.LazyFrame, out: Path) -> None:
    """Sinks a single lazyframe into a parquet file.

    Uses the streaming engine to sink a parquet file. The file is marked
    .tmp until the sinking completes. If the process crashes then the file
    is clearly marked as .tmp and won't be used downstream.
    """
    tmp = out.with_name(out.name + ".tmp")
    lf.sink_parquet(tmp)
    tmp.rename(out)


def sink_global_event_frame(ds: MIMIC4Dataset, out: Path) -> None:
    """Sink the transformed MIMIC4Dataset into a parquet file."""
    parts_dir = out.parent / "parts"
    # removing tree if prev run crashed
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)

    part_paths: list[Path] = []
    for table in settings.mimic4_ehr_tables:
        part = parts_dir / f"{table}.parquet"
        sink_atomic(build_event_pipeline(ds, table), part)
        part_paths.append(part)

    sink_atomic(
        pl.concat([pl.scan_parquet(p) for p in part_paths], how="diagonal"), out
    )
    shutil.rmtree(parts_dir)


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
    sink_global_event_frame(ds, parquet)
    sidecar.write_text(json.dumps(_fingerprint(ds.cache_dir)))
    return parquet
