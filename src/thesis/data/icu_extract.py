"""Memory-bounded extraction of ICU source tables into Parquet.

``chartevents.csv.gz`` (~3.3 GB gzip, ~433M rows) cannot be streamed by Polars:
gzip is not splittable, so ``scan_csv`` inflates the whole file at once and
exhausts memory. This module instead reads the raw CSV in bounded pyarrow
batches, filters each batch down to the ``itemid`` values of interest, and sinks
a small Parquet that the feature-engineering criteria can ``scan_parquet``
cheaply.

The multithreaded PyArrow default holds many decoded blocks in memory and
causes an OOM crash. With these settings the weight extraction peaks at
~2.8 GB
"""

from collections.abc import Collection
from pathlib import Path

import polars as pl
import pyarrow as pa
from pyarrow import csv as pacsv

from thesis.config import settings

# Charted body weights in ``chartevents`` (see ``d_items``). 226531 is recorded
# in pounds and carries a blank unit; convert downstream before use.
WEIGHT_ITEMIDS: frozenset[int] = frozenset(
    {
        226512,  # Admission Weight (Kg)
        224639,  # Daily Weight (kg)
        226846,  # Feeding Weight (kg)
        226531,  # Admission Weight (lbs.)
    }
)

# Urine output in ``outputevents`` (mimic-code ``kdigo_uo`` concept). GU irrigant
# volumes (227488/227489) are retained so the criterion can net them out.
URINE_ITEMIDS: frozenset[int] = frozenset(
    {
        226559,  # Foley
        226560,  # Void
        226561,  # Condom Cath
        226584,  # Ileoconduit
        226563,  # Suprapubic
        226564,  # R Nephrostomy
        226565,  # L Nephrostomy
        226567,  # Straight Cath
        226557,  # R Ureteral Stent
        226558,  # L Ureteral Stent
        227488,  # GU Irrigant Volume (In)
        227489,  # GU Irrigant/Urine Volume Out
    }
)

# ``chartevents`` records numeric values in ``valuenum``; ``outputevents`` in
# ``value``. subject_id/hadm_id/stay_id are all kept to align with the
# hadm_id-keyed creatinine arm without a later join.
_WEIGHT_COLUMNS: list[str] = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "charttime",
    "itemid",
    "valuenum",
    "valueuom",
]
_URINE_COLUMNS: list[str] = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "charttime",
    "itemid",
    "value",
    "valueuom",
]

_BLOCK_SIZE: int = 8 << 20  # 8 MiB input blocks — keeps peak memory bounded.


def extract_itemids(
    source: Path,
    out: Path,
    columns: list[str],
    itemids: Collection[int],
    *,
    block_size: int = _BLOCK_SIZE,
) -> None:
    """Filter a gzipped ICU CSV to ``itemids`` and sink it to Parquet.

    Reads ``source`` in bounded pyarrow batches so a non-streamable gzip CSV
    can be filtered without materializing the whole table. Only ``columns`` whose
    ``itemid`` is in ``itemids`` are decoded before being accumulated. The result is
    written atomically via a ``.tmp`` sibling that is renamed on completion, so a
    crash mid-write never leaves a truncated Parquet in place.

    Args:
        source: Path to the raw ``.csv.gz`` table (e.g. ``chartevents.csv.gz``).
        out: Destination Parquet path; parent directories are created.
        columns: Column names to decode from the CSV (must include ``itemid``).
        itemids: The ``itemid`` values to retain.
        block_size: pyarrow input block size in bytes. Larger values raise peak
            memory; the 8 MiB default is what keeps the weight extraction under
            ~2.8 GB.
    """
    read_options = pacsv.ReadOptions(block_size=block_size, use_threads=False)
    convert_options = pacsv.ConvertOptions(include_columns=columns)
    reader = pacsv.open_csv(
        source, read_options=read_options, convert_options=convert_options
    )

    wanted = list(itemids)
    kept: list[pl.DataFrame] = [
        pl.from_arrow(pa.Table.from_batches([batch])).filter(
            pl.col("itemid").is_in(wanted)
        )
        for batch in reader
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    pl.concat(kept).write_parquet(tmp)
    tmp.rename(out)


def extract_weights() -> None:
    """Extract charted body weights from ``chartevents`` into Parquet."""
    extract_itemids(
        settings.mimic4_ehr_chartevents,
        settings.mimic4_ehr_weight_parquet,
        _WEIGHT_COLUMNS,
        WEIGHT_ITEMIDS,
    )


def extract_urine_output() -> None:
    """Extract urine output from ``outputevents`` into Parquet."""
    extract_itemids(
        settings.mimic4_ehr_outputevents,
        settings.mimic4_ehr_urine_output_parquet,
        _URINE_COLUMNS,
        URINE_ITEMIDS,
    )


if __name__ == "__main__":
    extract_urine_output()
    extract_weights()
