"""Gather every scored result into one summary.

Run from the repo root:

    .venv-modelling/bin/python scripts/build_results_summary.py

Reads only artifacts other jobs already wrote -- checkpoint rankings, the XGBoost
comparisons, the probe, the diversity reports -- so the summary is regenerated rather
than maintained, and it cannot drift from the runs it describes. Everything comes
from `binary_metrics`, so no number here is defined differently from any other.
"""

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "motor_output" / "runs"
COMPARISON = ROOT / "motor_output" / "comparison"
DEST = COMPARISON / "results_summary.json"

# (label, file, key) for results already scored side by side
COMPARISONS: list[tuple[str, Path, str]] = [
    ("XGBoost, full", COMPARISON / "comparison.json", "xgboost"),
    (
        "XGBoost, recency removed",
        COMPARISON / "ablation_comparison.json",
        "xgboost_norecency",
    ),
    ("MOTOR v1, bare head", COMPARISON / "comparison.json", "motor"),
]

# v3 was scored before the `selection/` convention, so its candidates sit here
RANKING_FILES: list[tuple[str, Path]] = [
    ("MOTOR v3 full fine-tune, seed 0", COMPARISON / "v3_candidates.json"),
]

# (label, run folder) -- each contributes its highest-AUPRC scored checkpoint
SINGLE_RUNS: list[tuple[str, str]] = [
    ("MOTOR v3 full fine-tune, seed 1", "aki-seed1"),
    ("MOTOR v3 full fine-tune, seed 2", "aki-seed2"),
    ("LoRA r=8 q/v, 30k schedule", "lora-seed0-30k"),
    ("LoRA r=8 q/v, seed 0", "lora-seed0"),
    ("LoRA r=8 q/v, seed 1", "lora-seed1"),
    ("LoRA r=8 q/v, seed 2", "lora-seed2"),
    ("LoRA bagged 0.632, seed 10", "lora-bag10"),
    ("LoRA bagged 0.632, seed 11", "lora-bag11"),
    ("LoRA bagged 0.632, seed 12", "lora-bag12"),
]

ENSEMBLES: list[tuple[str, Path]] = [
    ("LoRA ensemble of 3, seed-only", COMPARISON / "diversity_seed_only.json"),
    ("LoRA ensemble of 3, bagged 0.632", COMPARISON / "diversity_bagged.json"),
]


def from_comparison(path: Path, key: str) -> dict[str, Any] | None:
    """One model's metrics out of a `run_comparison` file."""
    if not path.is_file():
        return None
    models = json.loads(path.read_text()).get("models", {})
    return models.get(key)


def best_of(ranking: Path) -> dict[str, Any] | None:
    """The highest-AUPRC row of a list of scored candidates."""
    if not ranking.is_file():
        return None
    rows = json.loads(ranking.read_text())
    return max(rows, key=lambda row: row["auprc"]) if rows else None


def from_run(run: Path) -> dict[str, Any] | None:
    """A run's highest-AUPRC scored checkpoint."""
    return best_of(run / "selection" / "checkpoint_ranking.json")


def from_probe(path: Path) -> dict[str, Any] | None:
    """The frozen-backbone probe's best result."""
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("best")


def from_diversity(path: Path) -> dict[str, Any] | None:
    """An ensemble's metrics, renamed onto the single-model keys."""
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    return {
        "auprc": report["ensemble_auprc"],
        "brier": report.get("ensemble_brier"),
        "ece": report.get("ensemble_ece"),
        "n_members": report["n_members"],
        "mean_flag_overlap": report["mean_flag_overlap"],
        "gain_over_best": report["gain_over_best"],
        "gain_over_best_lo": report.get("gain_over_best_lo"),
        "gain_over_best_hi": report.get("gain_over_best_hi"),
    }


def collect() -> list[dict[str, Any]]:
    """Every result that has been scored, most accurate first.

    Returns:
        list[dict[str, Any]]: One row per model, each carrying its label, its source
            and whatever metrics that source holds. Sources that have not been run
            are skipped rather than faked.
    """
    rows: list[dict[str, Any]] = []

    def add(label: str, source: str, metrics: dict[str, Any] | None) -> None:
        """Keeps a scored result, skipping sources that have not been run."""
        if metrics is not None:
            rows.append({"model": label, "source": source, **metrics})

    for label, path, key in COMPARISONS:
        add(label, str(path.relative_to(ROOT)), from_comparison(path, key))

    for label, path in RANKING_FILES:
        add(label, str(path.relative_to(ROOT)), best_of(path))

    probe = ROOT / "motor_output" / "probe" / "probe_metrics.json"
    add("MOTOR frozen + linear probe", "probe", from_probe(probe))

    for label, name in SINGLE_RUNS:
        add(label, f"runs/{name}", from_run(RUNS / name))

    for label, path in ENSEMBLES:
        add(label, str(path.relative_to(ROOT)), from_diversity(path))

    return sorted(rows, key=lambda row: row["auprc"], reverse=True)


def format_table(rows: list[dict[str, Any]]) -> str:
    """The summary as a markdown table, ready to paste into the write-up."""
    header = "| model | AUPRC | AUROC | Brier | ECE | source |"
    lines = [header, "|---|---|---|---|---|---|"]

    def cell(row: dict[str, Any], key: str, places: int = 4) -> str:
        """One metric, or an em dash where the source does not carry it."""
        value = row.get(key)
        return f"{value:.{places}f}" if isinstance(value, int | float) else "—"

    for row in rows:
        lines.append(
            f"| {row['model']} | {cell(row, 'auprc')} | {cell(row, 'auroc')} | "
            f"{cell(row, 'brier', 5)} | {cell(row, 'ece')} | `{row['source']}` |"
        )
    return "\n".join(lines)


def main() -> None:
    """Writes the summary and prints it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEST)
    args = parser.parse_args()

    rows = collect()
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    summary = {"fold": "validation", "n_models": len(rows), "models": rows}
    args.dest.write_text(json.dumps(summary, indent=2))

    print(format_table(rows))
    print(f"\nwrote {args.dest}")
    (args.dest.with_suffix(".md")).write_text(format_table(rows) + "\n")
    print(f"wrote {args.dest.with_suffix('.md')}")


if __name__ == "__main__":
    main()
