# Thesis — MIMIC-IV EHR Analysis

## Overview

A research codebase for exploratory data analysis (EDA) of the MIMIC-IV clinical database,
built around PyHealth's `MIMIC4Dataset` loader and a Streamlit dashboard.

## Tech stack

| Layer | Library |
|---|---|
| Data loading | PyHealth (`MIMIC4Dataset`) + Polars (LazyFrame) |
| Config | Pydantic-Settings (`Settings` class) + `.env` |
| EDA dashboard | Streamlit |
| Linting | Ruff (Google docstring convention) |

## Project layout

```
src/thesis/
  config.py          # Settings (reads MIMIC4_EHR_DATA_PATH from .env)
  constants.py       # DTYPE_TO_POLARS_DTYPE_MAP: str → pl.DataType
  mimic4_ehr.yaml    # Manifest: table attributes + dtype_mapping for casting
  data/
    EDASource.py     # Protocol (interface) for the dashboard
    sources.py       # PolarsEDASource (adapter) + cast_frame() helper
  eda/
    dashboard.py     # Streamlit app entry point
```

## Configuration

Create a `.env` file at the repo root:

```
MIMIC4_EHR_DATA_PATH=/path/to/mimic-iv
```

## Running the dashboard

```
streamlit run src/thesis/eda/dashboard.py
```

On Windows the `if __name__ == "__main__"` guard in `dashboard.py` is required.

## Data pipeline

1. `MIMIC4Dataset` (PyHealth) reads CSVs as all-string Parquet, caches under the default
   pyhealth cache dir (`~/.cache/pyhealth/…`). `global_event_df` returns a `pl.LazyFrame`.
2. `cast_frame(lf, dtype_map)` in `sources.py` applies the dtype mappings declared in
   `mimic4_ehr.yaml` to the LazyFrame before `.collect()`.
3. `PolarsEDASource` wraps the collected `pl.DataFrame` and exposes the `EDASource` protocol
   for the dashboard.

## Adding dtype casts for a table

Edit `mimic4_ehr.yaml` and add a `dtype_mapping` block under the relevant table:

```yaml
admissions:
  ...
  dtype_mapping:
    admissions/hadm_id: "UInt"
    admissions/hospital_expire_flag: "UInt"
```

Supported dtype strings are defined in `constants.py`:

| String | Polars type |
|---|---|
| `"String"` | `pl.String` (no-op) |
| `"UInt"` | `pl.UInt16` |
| `"Date"` | `pl.Date` (parsed as `%Y-%m-%d`) |

Add new entries to `DTYPE_TO_POLARS_DTYPE_MAP` when you need additional types.

## Key constraints

- PyHealth uses its **own** `mimic4_ehr.yaml` config (bundled inside the package) for table
  loading. The thesis `mimic4_ehr.yaml` is solely for `dtype_mapping`; it does not control
  which columns PyHealth loads.
- Column names in `global_event_df` follow PyHealth's `{table}/{attribute}` convention
  (e.g. `patients/anchor_age`). The `patient_id` column is always a String regardless of
  dtype_mapping.
- `cast_frame` silently skips columns that are absent from the frame, so dtype_mapping
  entries that don't match any column are harmless.
- Use `UInt32` (not `UInt16`) for ID-like columns such as `hadm_id` or `stay_id` which
  can exceed 65 535.
