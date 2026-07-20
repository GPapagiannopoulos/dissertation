"""Module for caching the diagnoses_made LazyFrame."""

from pathlib import Path
from typing import Final

# Bump when diagnosis logic changes
ENRICH_VERSION: Final[int] = 1


def _fingerprint(base_sidecar_path: Path, uo_sources: list[Path]) -> dict:
    """Fiingerprints the cached version of the diagnoses_made LazyFrame."""
    return {
        "enrich_version": ENRICH_VERSION,
    }
