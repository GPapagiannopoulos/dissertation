# Hospital-Acquired Acute Kidney Injury Prediction on MIMIC-IV v3.1

This repository contains the code and results for my dissertation as part of Imperial
College London's MSc Computing education. It examines the performance of LoRA ensembles
compared to full fine-tunes and gradient-boosted trees.

The task is HA-AKI prediction over a 48h horizon, on a landmark grid with predictions
over 12 hours. The backbone is MOTOR, fully ported into PyTorch.

The repository contains two independent stacks:

- `eda` — loading MIMIC-IV through PyHealth, a set of pure Polars transforms, KDIGO
  diagnosis, and a Streamlit dashboard for high level data exploration.
- `modelling` — the MEDS/OMOP ETL, the MOTOR port, the training loops (full fine-tune,
  LoRA, linear probe), the XGBoost baseline, and the evaluation suite.

No patient data or checkpoints are available in this repository. Both MIMIC-IV and MOTOR
are gated and require mandatory training to be completed. Information on how to get access
can be found below.

## Repository layout

```
src/thesis/
  eda/                    # the `eda` extra
    config.py             # Pydantic Settings + EAVFieldInformation; loads mimic4_ehr.yaml
    mimic4_ehr.yaml        # per-table attributes, dtype_mapping, eav_fields
    dashboard.py          # Streamlit app
    data/                 # EDASource port, PolarsEDASource adapter, transform helpers
    feature_engineering/  # KDIGO phenotyping; produces the AKI labels stage 4 reads
  modelling/              # the `modelling` extra
    etl/                  # MIMIC-IV -> MEDS -> MOTOR's tokens
    cohort/               # the subject split and the landmark labeller
    backbone/             # the PyTorch MOTOR port: layers, encoder, tokenizer, weights
    finetune/             # sequences, batching, the training loop, the head, LoRA, probe
    baseline/             # the XGBoost comparator
    evaluation/           # metrics, subject-level bootstrap intervals, model comparison
    ensemble/             # diversity, uncertainty, decision curves, selective prediction
    figures/              # matplotlib composition for the report
scripts/
  etl/ train/ evaluate/ sweeps/ tools/    # thin drivers, one per stage
tests/
  eda/ modelling/         # mirrors src/thesis, one test file per helper
```

The two top-level packages are meant to run independently: nothing under
`thesis.modelling` imports from `thesis.eda` or vice versa.

## Getting MIMIC-IV

MIMIC-IV is credentialed-access data. The process for requesting access can be found
here: [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/)
It is necessary to sign up on PhysioNet and complete the CITI "Data or Specimens Only Research"
to access the dataset.

This work uses MIMIC-IV v3.1 specifically.
Reidentifying individuals, or attempting to, violates the DUA.

## Getting MOTOR checkpoints

MOTOR's checkpoints and dictionary are also gated. The process of requesting access mirror that
for MIMIC-IV, and more details can be found here: [MOTOR](https://huggingface.co/StanfordShahLab/motor-t-base)

## Further pre-requisites

Beyond MIMIC and MOTOR, the modelling stack needs the OMOP Athena vocabulary export, and access to a CUDA GPU:

| what                          | where it goes | why                                                 |
|-------------------------------|---------------|-----------------------------------------------------|
| OMOP Athena vocabulary export | `athena/`     | maps MIMIC-native codes onto standard OMOP concepts |
| a CUDA GPU                    | —             | every training and scoring stage                    |

The Athena export must include the vocabularies the concept map reads:

- SNOMED - Systematic Nomenclature of Medicine - Clinical Terms (IHTSDO)
- ICD9CM - International Classification of Diseases, Ninth Revision, Clinical Modification, Volume 1 and 2 (NCHS)
- ICD9Proc - International Classification of Diseases, Ninth Revision, Clinical Modification, Volume 3 (NCHS)
- HCPCS - Healthcare Common Procedure Coding System (CMS)
- LOINC - Logical Observation Identifiers Names and Codes (Regenstrief Institute)
- RxNorm - RxNorm (NLM)
- NDC - National Drug Code (FDA and manufacturers)
- Gender - OMOP Gender
- Race - Race and Ethnicity Code Set (USBC)
- CMS Place of Service - Place of Service Codes for Professional Claims (CMS)
- ATC - WHO Anatomic Therapeutic Chemical Classification
- ICD10PCS - ICD-10 Procedure Coding System (CMS)
- Ethnicity - OMOP Ethnicity
- ICD10CM - International Classification of Diseases, Tenth Revision, Clinical Modification (NCHS)
- RxNorm Extension - OMOP RxNorm Extension
- OMOP Extension - OMOP Extension (OHDSI)

## Install

Python **3.13+** and [`uv`](https://docs.astral.sh/uv/).

The two modules cannot currently share an environment: pyhealth pins `numpy>=2.2`, femr pins
`numpy<2`. They are declared as conflicting and each is run from its own virtual environment.
The modelling stack as a result using numpy 1.26.4, and the results have not been validated
on a more recent version of numpy.

```bash
# the EDA stack -> .venv
uv sync --extra eda

# the modelling stack -> .venv-modelling
UV_PROJECT_ENVIRONMENT=.venv-modelling uv sync --extra modelling
```

Run each stack with its own interpreter. `.venv-modelling/bin/python` for anything under
`scripts/etl`, `scripts/train` or `scripts/evaluate`; `.venv/bin/python` (or `uv run`) for the
dashboard and the EDA tests.

## Configuration

Create `.env` at the repository root. All nine paths are required and are validated by
`Settings` at startup:

```dotenv
MIMIC4_EHR_DATA_PATH=/path/to/mimic-iv                     # EHR root directory
MIMIC4_EHR_D_ICD_PROCEDURES=/path/to/d_icd_procedures.csv
MIMIC4_EHR_D_ICD_DIAGNOSES=/path/to/d_icd_diagnoses.csv
MIMIC4_EHR_D_HCPCS=/path/to/d_hcpcs.csv
MIMIC4_EHR_D_LABITEMS=/path/to/d_labitems.csv
MIMIC4_EHR_CHARTEVENTS=/path/to/chartevents.csv.gz
MIMIC4_EHR_OUTPUTEVENTS=/path/to/outputevents.csv.gz
MIMIC4_EHR_WEIGHT_PARQUET=/path/to/weight.parquet          # written by icu_extract
MIMIC4_EHR_URINE_OUTPUT_PARQUET=/path/to/urine_output.parquet

MIMIC4_EHR_DEV_MODE=true                                   # optional, defaults to true
```

The `D_*` files are MIMIC's dictionary tables, used to attach human-readable descriptions. The
two parquets are written by `eda/data/icu_extract.py`, which reads `chartevents.csv.gz` in
pyarrow batches. Polars cannot stream it, because gzip is not splittable and
`scan_csv` inflates the whole 3.3 GB file at once.

## Running the EDA dashboard

```bash
uv run streamlit run src/thesis/eda/dashboard.py
```

Pick an event type and field from the sidebar. The main view provides an Overview , a Field summary
and a Preview.

MIMIC-IV arrives in an Entity-Attribute-Value shape, so a column like `labevents/valuenum`
mixes every measurement in the table. Numeric summaries therefore filter to one cohort and
raise `MixedUnitsError` if the units are not homogeneous. Which fields
need filtering is declared per-table under `eav_fields` in `mimic4_ehr.yaml`.

### Dev sample vs. full data

The loader defaults to PyHealth's `dev` mode, which loads only the first 1,000 subjects. For the full dataset:

```bash
MIMIC4_EHR_DEV_MODE=false uv run streamlit run src/thesis/eda/dashboard.py
```

> On Windows the `if __name__ == "__main__"` guard in `dashboard.py` is required.

## Reproducing the modelling pipeline

The ETL pipeline uses one-off scripts to transform data in preparation for the next stage.
Every driver takes its paths as flags but defaults to the conventional repo-relative layout, so
with `mimic_data/`, `motor_model/` and `athena/` in place most stages need no arguments. Every
stage refuses an existing destination to avoid accidentally overwritting hours of work, and every
stage processes shards one at a time to limit the memory requirement.

Run each with `.venv-modelling/bin/python`, in order:

| #   | script                                 | produces                                  | notes                                       |
|-----|----------------------------------------|-------------------------------------------|---------------------------------------------|
| 1   | `etl/run_base_meds.py`                 | `meds_output/base/`                       | MIMIC-IV → MEDS, ~3.8 GB, 200 shards        |
| 2   | `etl/build_meds_reader_db.py`          | `meds_output/reader_db/`                  | MEDS → queryable database, ~25 min          |
| 2.5 | `etl/build_concept_map.py`             | `meds_output/concept_map.parquet`         | MIMIC codes → MOTOR tokens, ~44k rows       |
| 2.6 | `etl/apply_concept_map.py`             | `meds_output/normalized/`                 | rewrites the shards onto MOTOR's vocabulary |
| 3   | `etl/build_subject_split.py`           | `labels/subject_split.parquet`            | subject-level stratified 70/15/15           |
| 3b  | `etl/build_diagnosis_labels.py`        | KDIGO AKI diagnoses                       | reads the EDA stack's phenotyping           |
| 3c  | `etl/identify_surviving_admissions.py` | `labels/surviving_aki_admissions.parquet` |                                             |
| 4   | `etl/build_labels.py`                  | `labels/landmark_labels.parquet`          | the 12-hourly landmark grid and its labels  |
| 5.2 | `etl/build_sequences.py`               | `meds_output/sequences/`                  | tokenised, positioned, aged sequences       |
| 7a  | `etl/build_baseline_features.py`       | `meds_output/baseline/`                   | the XGBoost feature table                   |

## Training the arms

```bash
# LoRA over the frozen backbone
.venv-modelling/bin/python scripts/train/train_motor_aki_lora.py \
    --dest motor_output/runs/lora-seed0 \
    --seed 0 \
    --total-steps 30000 \
    --lora-r 8 --lora-alpha 32 \
    --lora-targets q_proj k_proj v_proj ff_proj o_proj

# the monolithic full fine-tune
.venv-modelling/bin/python scripts/train/train_motor_aki.py \
    --dest motor_output/runs/aki-seed0 --seed 0

# the XGBoost baseline (see the memory note below)
.venv-modelling/bin/python scripts/train/train_baseline_aki.py \
    --dest motor_output/runs/xgb-seed0 --nthread 12 --max-bin 64

# the frozen-backbone linear probe
.venv-modelling/bin/python scripts/train/run_linear_probe.py --dest motor_output/probe
```

`scripts/sweeps/` holds the bash drivers that train and rank a whole axis in turn.
Each skips training and scoring independently, so an interrupted sweep resumes correctly.

There is no resume. Checkpoints are written every few thousand steps, so a killed run still
yields a scorable ladder, but it restarts from step 0.

## Evaluation

We found that the small subsample read during in-loop evaluations was misleading. To avoid
selecting the wrong checkpoint, we recommend evaluating on the full cohort.

```bash
.venv-modelling/bin/python scripts/evaluate/scoring/score_checkpoints.py \
    --run motor_output/runs/lora-seed0 --stride 2
```

Each checkpoint's per-landmark predictions are banked as an npz of
`(scores, targets, subjects, times)`. Downstream analysis is arithmetic over those
bundles.

```bash
scripts/evaluate/comparison/     # paired subject-level intervals between arms
scripts/evaluate/ensembles/      # diversity, snapshot ensembles, greedy selection
scripts/evaluate/clinical/       # decision curves and calibration curves, and their figures
scripts/evaluate/uncertainty/    # aleatoric/epistemic split, selective prediction
scripts/evaluate/horizon/        # re-score at 72h or 7d with no retraining
scripts/evaluate/build_results_summary.py   # regenerates the tables RESULTS.md indexes
```

Two conventions to keep in mind:

- This work selected checkpoints based on minimum validation loss scored on the entire cohort.
- Confidence intervals resample subjects to account for within-admission landmarks being highly correlated.

## Hardware and time budget

Everything here was developed on a laptop RTX 4060 (8 GB VRAM) with 14 GB of host RAM. Plan
accordingly:

| job                                      | cost                  |
|------------------------------------------|-----------------------|
| LoRA fine-tune, 30,000 steps             | ~8.7 h, 2.4 GiB VRAM  |
| monolithic full fine-tune, 30,000 steps  | ~14 h, 5.4 GiB VRAM   |
| XGBoost, 600 rounds                      | ~14 h across two fits |
| scoring one checkpoint on the full fold  | ~13 min               |
| `meds_output/` + `motor_output/` on disk | tens of GB            |

We recommend running the XGBoost fit under a memory cap. The matrices do not fit in 14 GB and the
process has OOM on several occasions:

```bash
systemd-run --user --unit=xgb-aki --working-directory="$PWD" \
    -p MemoryMax=9500M -p MemorySwapMax=1G -p OOMScoreAdjust=800 \
    .venv-modelling/bin/python scripts/train/train_baseline_aki.py \
        --dest motor_output/runs/xgb-seed0 --nthread 12 --max-bin 64
```

## Development

```bash
uv run ruff check                                  # lint
uv run ruff format --check                         # formatting
uv run pytest -m eda                               # the EDA suite
.venv-modelling/bin/pytest tests/modelling         # the modelling suite
```

Install the hooks with `uv run pre-commit install`. They run ruff with `--fix`, ruff-format, and
a 25 MB cap on any newly added file.

Tests follow a builder + parametrize style: small DataFrame builder fixtures accept overrides to match
the context of specific cases, and use `polars.testing.assert_frame_equal` with literal expected frames.
Ruff enforces the Google docstring convention.

## Disclaimer

This is research code accompanying a dissertation. It is not a clinical device and under no circumstances
must it be used to support clinical decision making. None of the models have been externally validated,
and their use is likely to cause patients harm.

No licence file is currently present; all rights are reserved pending one. If you use this work,
please cite the dissertation and the underlying [MIMIC-IV](https://physionet.org/content/mimiciv/)
and [MOTOR](https://arxiv.org/abs/2301.03150) publications.
