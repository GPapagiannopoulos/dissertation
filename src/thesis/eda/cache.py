"""Caching module for the dashboard data."""

from hashlib import sha256
from importlib import resources
from pathlib import Path
from typing import Final

from thesis.config import settings

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
            (p.name, p.stat().st_size, p.stat().st_mtime_ns)
            for p in source.glob("*.parquet")
        ),
        "manifest": sha256(
            resources.files("thesis").joinpath("mimic4_ehr.yaml").read_bytes()
        ),
        "mappings": {
            p.name: sha256(p.read_bytes())
            for p in [
                settings.mimic4_ehr_d_icd_diagnoses,
                settings.mimic4_ehr_d_icd_procedures,
                settings.mimic4_ehr_d_icd_diagnoses,
                settings.mimic4_ehr_d_hcpcs,
                settings.mimic4_ehr_d_labitems,
            ]
        },
    }
