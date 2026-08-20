# Thesis — MIMIC-IV EHR Analysis

A research codebase for exploratory data analysis (EDA) of the [MIMIC-IV](https://physionet.org/content/mimiciv/)
clinical database. It loads MIMIC-IV via PyHealth's `MIMIC4Dataset`, applies a set of pure,
individually tested Polars transforms, and exposes the result through a small Streamlit
dashboard for eyeballing distributions.

The EDA dashboard is one part of a larger project whose end goal is training ML models on
this data, so the **data-loading / transform layer is the durable, reusable asset**; the
dashboard is a deliberately thin lens over it.

## Project structure

```
src/thesis/
  eda/                 # the `eda` extra: loading, transforms, phenotyping, the dashboard
    config.py          # Pydantic Settings + EAVFieldInformation; loads mimic4_ehr.yaml
    constants.py       # DTYPE_TO_POLARS_DTYPE_MAP: dtype string -> pl.DataType
    mimic4_ehr.yaml    # Manifest: per-table attributes, dtype_mapping, eav_fields
    dashboard.py       # Streamlit app: load -> transform -> PolarsEDASource -> tabs
    filters.py         # valid_fields() and field-exclusion helpers for the dashboard
    data/
      eda_source.py    # EDASource port (Protocol) + NumericSummary, MixedUnitsError, EmptyHistError
      sources.py       # PolarsEDASource adapter + module-level transform helpers
    feature_engineering/ # KDIGO phenotyping over the cached event substrate
  modelling/           # the `modelling` extra: MEDS ETL, the MOTOR port, training, evaluation
tests/
  eda/                 # Builder + parametrize suites for the transforms and the adapter
  modelling/           # mirrors src/thesis/modelling package for package
```

The two top-level packages match the two optional-dependency extras exactly: nothing under
`thesis.modelling` imports from `thesis.eda`, or the other way round. They cannot share an
environment (pyhealth pins numpy>=2.2, femr pins numpy<2), so each has its own venv.

### The pipeline, end to end

1. `MIMIC4Dataset` (PyHealth) reads the CSVs and exposes `global_event_df` as a Polars
   `LazyFrame` in an Entity-Attribute-Value shape (`{table}/{attribute}` columns).
2. Module-level helpers in `eda/data/sources.py` transform the frame **before** `.collect()`
   (each is pure — frame in, frame out — and unit-tested):
   - `cleanse_float_values` — strip commas/ranges so string columns are safe to cast to float.
   - `cast_frame` — apply the dtype casts declared in `mimic4_ehr.yaml`.
   - `mimic4_add_descriptions_to_icd_codes` / `replace_mimic4_non_icd_codes` — join dictionary
     CSVs to attach human-readable `description` columns.
3. The collected `pl.DataFrame` is wrapped in `PolarsEDASource`, which implements the
   `EDASource` port (event/patient counts, field listings, categorical/numeric summaries,
   and histograms) consumed by the dashboard.

## Requirements

- Python **3.13+**
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- A local copy of the MIMIC-IV EHR data (CSV form) plus its dictionary tables

## Setup

Install dependencies (creates the virtualenv from `pyproject.toml` / lockfile):

```bash
uv sync
```

Create a `.env` file at the repository root pointing at your MIMIC-IV data. All five paths
are required:

```dotenv
MIMIC4_EHR_DATA_PATH=/path/to/mimic-iv          # EHR root directory
MIMIC4_EHR_D_ICD_PROCEDURES=/path/to/d_icd_procedures.csv
MIMIC4_EHR_D_ICD_DIAGNOSES=/path/to/d_icd_diagnoses.csv
MIMIC4_EHR_D_HCPCS=/path/to/d_hcpcs.csv
MIMIC4_EHR_D_LABITEMS=/path/to/d_labitems.csv
```

The `D_*` files are the MIMIC-IV dictionary tables used to map codes to descriptions.
Settings are loaded and validated at startup by `Settings` in `config.py`.

## Running the dashboard

```bash
streamlit run src/thesis/eda/dashboard.py
```

Pick an event type and field from the sidebar; the main area shows an **Overview**
(event/patient counts, field dtypes), a **Field summary** (categorical value-count bars or
numeric describe stats + histogram, with cohort filters for EAV fields), and a **Preview**.

### Dev sample vs. full data

By default the loader uses PyHealth's `dev` mode, which reads a small sample — fast to start
and enough to explore structure. To load the **full** dataset, set `MIMIC4_EHR_DEV_MODE`
(bound to `mimic4_ehr_dev_mode` in `Settings`, default `true`):

```bash
# full dataset for one run
MIMIC4_EHR_DEV_MODE=false streamlit run src/thesis/eda/dashboard.py
```

Or set `MIMIC4_EHR_DEV_MODE=false` in `.env` to make it the default. Loading the full data
is significantly heavier, so keep the dev sample for day-to-day work.

> On Windows the `if __name__ == "__main__"` guard in `dashboard.py` is required.

## Development

Run the test suite and linter (both configured in `pyproject.toml`):

```bash
uv run pytest
uv run ruff check
```

Tests follow a **builder + parametrize** style: small-DataFrame builder fixtures (in
`tests/eda/data/conftest.py`) overriding only the fields a case cares about, literal expected
frames, and `polars.testing.assert_frame_equal`. Ruff enforces the Google docstring
convention.
