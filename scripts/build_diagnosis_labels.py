"""One-off script for generating a parquet with the positive diagnostic labels."""

from pathlib import Path

import polars as pl

from thesis.eda.cache import ensure_event_cache
from thesis.feature_engineering.driver import diagnose_all

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "meds_output" / "labels"
file_path = DEST / "diagnosis_labels.parquet"


def main() -> None:
    """Generates and sinks the positive labels in a parquet file."""
    diagnosis_labels = diagnose_all(pl.scan_parquet(ensure_event_cache()))
    if not DEST.exists():
        print(f"{DEST} not found - making directory.")
        DEST.mkdir(parents=True)

    print(f"Sinking parquet at {file_path}")
    diagnosis_labels.sink_parquet(file_path)
    print(pl.scan_parquet(file_path).collect(engine="streaming").describe())


if __name__ == "__main__":
    main()
