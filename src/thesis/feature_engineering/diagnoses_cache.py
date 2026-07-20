"""Module for caching the diagnoses_made LazyFrame."""

import json
from pathlib import Path
from typing import Final

import polars as pl

from thesis.config import settings
from thesis.eda.cache import ensure_event_cache, sink_atomic
from thesis.feature_engineering.driver import diagnose_all

# Bump when diagnosis logic changes
ENRICH_VERSION: Final[int] = 1


def _fingerprint(base_sidecar_path: Path, uo_sources: list[Path]) -> dict:
    """Fingerprints the cached version of the diagnoses_made LazyFrame."""
    return {
        "enrich_version": ENRICH_VERSION,
        "base": json.loads(base_sidecar_path.read_text()),
        "uo_sources": sorted(
            [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in uo_sources]
        ),
    }


def ensure_diagnosis_cache() -> Path:
    """Confirms whether the diagnoses LF is already cached."""
    uo_sources: list[Path] = [
        settings.mimic4_ehr_urine_output_parquet,
        settings.mimic4_ehr_weight_parquet,
    ]

    base_path = ensure_event_cache()
    sidecar = base_path.parent / "events.fingerprint.json"
    diagnoses_path = base_path.parent / "diagnoses.parquet"
    diagnoses_sidecar = base_path.parent / "diagnoses.fingerprint.json"
    fingerprint = _fingerprint(sidecar, uo_sources)

    if base_path.exists() and sidecar.exists():
        if json.loads(diagnoses_sidecar.read_text()) == fingerprint:
            return diagnoses_path

    diagnoses_sidecar.unlink(missing_ok=True)
    sink_atomic(diagnose_all(pl.scan_parquet(base_path)), diagnoses_path)
    diagnoses_sidecar.write_text(json.dumps(fingerprint))

    return diagnoses_path


def compose_enriched(base_parquet: Path, diagnoses_parquet: Path) -> pl.LazyFrame:
    """Concatenates the diagnoses_made LazyFrame to the base dataset LazyFrame."""
    return pl.concat(
        [pl.scan_parquet(base_parquet), pl.scan_parquet(diagnoses_parquet)],
        how="diagonal",
    )


def load_enriched() -> pl.LazyFrame:
    """Loads the dataset enriched with diagnoses_made events."""
    return compose_enriched(ensure_event_cache(), ensure_diagnosis_cache())
