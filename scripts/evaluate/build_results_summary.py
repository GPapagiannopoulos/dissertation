"""Gather every scored result into one summary.

Run from the repo root:

    .venv-modelling/bin/python scripts/evaluate/build_results_summary.py

Reads only artifacts other jobs already wrote. Everything comes from `binary_metrics`,
so no number here is defined differently from any other.
"""

import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "motor_output" / "runs"
COMPARISON = ROOT / "motor_output" / "comparison"
DEST = COMPARISON / "results_summary.json"
NEWGRID_DIR = COMPARISON / "newgrid"
NEWGRID = NEWGRID_DIR / "comparison.json"
PAIRING600 = NEWGRID_DIR / "ensemble_vs_xgboost600.json"
# compositions explored 2026-08-31 over already-banked bundles: the XGBoost/LoRA
# hybrid, the rank ladder as an ensemble, and the three narrow placements. Nothing
# here re-selects a checkpoint -- every stem still comes from a VALIDATION ladder.
EXPLORE = {
    "validation": NEWGRID_DIR / "explore_validation.json",
    "testing": NEWGRID_DIR / "explore_test.json",
}

# the test fold, scored under the design frozen in REPORT_HANDOVER.md section 7.
# Nothing here re-selects a checkpoint, because every stem was chosen on validation.
# The DATE is never written down -- `test_provenance` reads it off the inputs, so a
# later re-score cannot leave a stale claim about when the fold was touched.
TEST_INTERVALS = NEWGRID_DIR / "test_fold_intervals.json"
TEST_PAIRED = NEWGRID_DIR / "paired_lora_vs_monolithic_test.json"
TEST_XGBOOST = NEWGRID_DIR / "ensemble_vs_xgboost600_test.json"
SCORER = ROOT / "scripts" / "evaluate" / "score_checkpoints.py"
# the probe was scored on test AFTER the freeze, as a capacity floor. It is not a run
# folder, so nothing walking `runs/*/selection_test` finds it.
TEST_PROBE = ROOT / "motor_output" / "probe-ng" / "probe_metrics_test.json"

# arms in TEST_INTERVALS that are ONE model, and so name the same checkpoint as a
# SINGLE_RUNS row. Both rows carry the same point estimate and DIFFERENT intervals,
# because the interval report resamples 2,000 times where `score_checkpoints.py`
# banks `INTERVAL_RESAMPLES`. The report's band is kept and the roster's row dropped.
TEST_INTERVAL_SINGLES: dict[str, str] = {
    "MOTOR monolithic, single checkpoint": "ng-aki-seed1",
}

# which paired deltas are worth a row, and how to label them. The left arm is always
# the one the thesis claims for, so a POSITIVE AUPRC delta favours the claim.
TEST_PAIRINGS: list[tuple[str, str]] = [
    ("lora_last2-monolithic_seeds", "LoRA 23 - monolithic, 3 seeds"),
    ("lora_last2-monolithic_snapshot", "LoRA 23 - monolithic, 1 run"),
    ("lora_last2-monolithic_single", "LoRA 23 - monolithic, single checkpoint"),
    ("lora_last2-lora_all_last", "LoRA 23 - LoRA all-last"),
]

# (label, file, key) for results already scored side by side. The old-grid
# `comparison.json` and `ablation_comparison.json` rows were removed on 2026-08-25:
# they were measured on the 48h-start grid and cannot share a table with these.
COMPARISONS: list[tuple[str, Path, str]] = [
    ("XGBoost, 600 rounds", PAIRING600, "xgboost600"),
    ("XGBoost, 300 rounds (round cap, not converged)", NEWGRID, "xgboost"),
    ("MOTOR full fine-tune, seed 1, paired against it", NEWGRID, "motor"),
]

RANKING_FILES: list[tuple[str, Path]] = []

# (label, run folder) -- each contributes the checkpoint its ranking selects BY
# VALIDATION LOSS, which is the project's frozen rule. A folder that has not been
# scored yet contributes nothing, so a config queued but not reached is simply
# absent rather than an error.
SINGLE_RUNS: list[tuple[str, str]] = [
    # the monolithic arm
    ("MOTOR full fine-tune, seed 0, 30k", "ng-aki-seed0"),
    ("MOTOR full fine-tune, seed 1, 30k", "ng-aki-seed1"),
    ("MOTOR full fine-tune, seed 2, 30k", "ng-aki-seed2"),
    ("MOTOR full fine-tune, seed 0, 15k", "ng-aki-seed0-15k"),
    ("MOTOR full fine-tune, seed 0, lr 3e-6", "ng-aki-lowlr"),
    # the rank ladder: q/k/v/ff, alpha held at 32, one 30k cosine each
    ("LoRA r=2 a=32 q/k/v/ff", "ng-all4-r2-a32"),
    ("LoRA r=4 a=32 q/k/v/ff", "ng-all4-r4-a32"),
    ("LoRA r=8 a=32 q/k/v/ff", "ng-all4-r8"),
    ("LoRA r=16 a=32 q/k/v/ff", "pilot-all4-r16-a32"),
    ("LoRA r=32 a=32 q/k/v/ff", "ng-all4-r32-a32"),
    # the alpha control: the same ranks with alpha scaled as 4r
    ("LoRA r=16 a=64 q/k/v/ff", "ng-all4-r16"),
    ("LoRA r=32 a=128 q/k/v/ff", "ng-all4-r32"),
    # placement, all at r=8
    ("LoRA r=8 q/k/v/ff/o", "ng-all5-r8"),
    ("LoRA r=8 o only", "ng-o-r8"),
    ("LoRA r=8 q/k/v", "ng-qkv-r8"),
    ("LoRA r=8 ff only", "ng-ff-r8"),
    ("LoRA r=8 q/v", "ng-qv-seed1"),
]

# (label, section, index) into an `explore_ensembles.py` report. The hybrid is
# handled separately because it is a blend of two arms rather than a member set.
EXPLORE_ENSEMBLES: list[tuple[str, str, int]] = [
    ("LoRA ensemble, rank ladder r=2..32 by loss", "rank_ensemble", 0),
    ("LoRA ensemble, rank ladder x (by-loss + last)", "rank_ensemble", 1),
    ("LoRA ensemble, placements q/k/v + ff + o by loss", "placement_ensemble", 0),
    ("LoRA ensemble, placements x (by-loss + last)", "placement_ensemble", 1),
]

ENSEMBLES: list[tuple[str, Path]] = [
    ("LoRA ensemble, 12 configs by loss", COMPARISON / "diversity_ng12.json"),
    (
        "LoRA ensemble, 12 configs x (by-loss + last)",
        COMPARISON / "diversity_ng23.json",
    ),
]


def from_comparison(path: Path, key: str) -> dict[str, Any] | None:
    """One model's metrics out of a `run_comparison` file.

    Falls back to the top level, because the ad-hoc pairing scripts write each
    model as its own key rather than under `models`.
    """
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    return report.get("models", report).get(key)


def best_of(ranking: Path) -> dict[str, Any] | None:
    """The highest-AUPRC row of a list of scored candidates."""
    if not ranking.is_file():
        return None
    rows = json.loads(ranking.read_text())
    return max(rows, key=lambda row: row["auprc"]) if rows else None


def by_loss(ranking: Path) -> dict[str, Any] | None:
    """The lowest-loss row of a list of scored candidates.

    This is the project's frozen selection rule. Taking the highest-AUPRC row
    instead -- as this script did until 2026-08-25 -- reports each run at a
    checkpoint chosen on the metric being reported, which is the selection
    optimism the rule exists to avoid.
    """
    if not ranking.is_file():
        return None
    rows = json.loads(ranking.read_text())
    return min(rows, key=lambda row: row["loss"]) if rows else None


def from_run(run: Path) -> dict[str, Any] | None:
    """A run's by-loss scored checkpoint."""
    return by_loss(run / "selection" / "checkpoint_ranking.json")


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


def from_explore(path: Path, section: str, index: int) -> dict[str, Any] | None:
    """One member set out of an `explore_ensembles.py` report.

    Args:
        path (Path): The report for the fold being tabulated.
        section (str): `rank_ensemble` or `placement_ensemble`.
        index (int): 0 for the by-loss set, 1 for the same set plus each `last.pt`.

    Returns:
        dict[str, Any] | None: The ensemble's metrics on the single-model keys, or
            None if the report has not been written.
    """
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    rows = report.get(section) or []
    if index >= len(rows):
        return None
    entry = rows[index]
    return {
        **entry["ensemble"],
        "n_members": entry["n_members"],
        "mean_flag_overlap": entry["mean_flag_overlap"],
        "gain_over_best": entry["gain_over_best"],
        "gain_over_best_lo": entry["gain_over_best_lo"],
        "gain_over_best_hi": entry["gain_over_best_hi"],
        "resamples": report["resamples"],
    }


def from_hybrid(path: Path) -> dict[str, Any] | None:
    """The 50/50 XGBoost + LoRA-ensemble blend.

    The weight is fixed at a half in advance, not tuned: the sweep's argmax is 0.45 on
    BOTH folds and untuned 50/50 sits 0.0001 below it, so nothing here was fitted on
    the fold being reported.

    Args:
        path (Path): The report for the fold being tabulated.

    Returns:
        dict[str, Any] | None: The blend's metrics, or None if it has not been run.
    """
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    hybrid = report.get("hybrid")
    if hybrid is None:
        return None
    return {
        **hybrid["hybrid50"],
        "n_members": hybrid["n_lora_members"] + 1,
        "resamples": report["resamples"],
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

    probe = ROOT / "motor_output" / "probe-ng" / "probe_metrics.json"
    add("MOTOR frozen + linear probe", "probe", from_probe(probe))

    for label, name in SINGLE_RUNS:
        add(label, f"runs/{name}", from_run(RUNS / name))

    for label, path in ENSEMBLES:
        add(label, str(path.relative_to(ROOT)), from_diversity(path))

    explore = EXPLORE["validation"]
    source = str(explore.relative_to(ROOT))
    for label, section, index in EXPLORE_ENSEMBLES:
        add(label, source, from_explore(explore, section, index))
    add("HYBRID: XGBoost 600 + LoRA 23, 50/50", source, from_hybrid(explore))

    return sorted(rows, key=lambda row: row["auprc"], reverse=True)


def ranking_resamples() -> int:
    """How many resamples `score_checkpoints.py` banked behind a ladder's intervals.

    Read from that script rather than repeated here, so the two files cannot drift
    into quoting different budgets for the same column.

    Returns:
        int: Its `INTERVAL_RESAMPLES`.
    """
    spec = importlib.util.spec_from_file_location("score_checkpoints", SCORER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.INTERVAL_RESAMPLES)


def test_probe_row() -> dict[str, Any] | None:
    """The frozen-backbone probe on the test fold, if it has been scored.

    Nothing is refitted there -- the head, its weight decay and its epoch were all
    chosen on validation -- so this is a floor, not a competitor, and it moves no
    frozen claim either way.

    Returns:
        dict[str, Any] | None: One row, or None if the probe has not been scored.
    """
    if not TEST_PROBE.is_file():
        return None
    report = json.loads(TEST_PROBE.read_text())
    return {
        "model": "MOTOR frozen + linear probe",
        "run": None,
        "n_members": 1,
        "resamples": report["resamples"],
        **report["metrics"],
    }


def test_sources() -> list[Path]:
    """Every test-fold artifact the block was actually built from.

    Returns:
        list[Path]: The interval and pairing reports plus each run's test ladder,
            skipping any that do not exist.
    """
    paths = [TEST_INTERVALS, TEST_PAIRED, TEST_XGBOOST, TEST_PROBE, EXPLORE["testing"]]
    paths += [
        RUNS / name / "selection_test" / "checkpoint_ranking.json"
        for _, name in SINGLE_RUNS
    ]
    return [path for path in paths if path.is_file()]


def test_provenance(sources: list[Path]) -> str:
    """When the test fold was last read, taken from the inputs themselves.

    Args:
        sources (list[Path]): Every artifact the test-fold block was built from.

    Returns:
        str: The newest source's date and how many artifacts back the block, or a
            note that none exist yet.
    """
    if not sources:
        return "not yet scored"
    newest = max(path.stat().st_mtime for path in sources)
    date = dt.date.fromtimestamp(newest).isoformat()
    return (
        f"{len(sources)} artifacts, newest {date}; scored under the design frozen "
        "in REPORT_HANDOVER.md section 7"
    )


def test_single_rows() -> list[dict[str, Any]]:
    """Each individual run at its frozen checkpoint, on the test fold.

    The stem comes from the run's VALIDATION ladder, never its test one. Only the
    frozen stems were ever scored on test, but a run can hold two of them, and picking
    the better of those on test would be exactly the selection optimism the freeze
    exists to prevent -- `ng-aki-seed2` selects step_018000 on validation and
    step_006000 on test.

    Returns:
        list[dict[str, Any]]: One row per run that has been scored on test, each
            already carrying the intervals `score_checkpoints.py` banked. Runs with no
            `selection_test/` are skipped rather than faked.
    """
    rows: list[dict[str, Any]] = []
    for label, name in SINGLE_RUNS:
        chosen = by_loss(RUNS / name / "selection" / "checkpoint_ranking.json")
        scored = RUNS / name / "selection_test" / "checkpoint_ranking.json"
        if chosen is None or not scored.is_file():
            continue
        stem = Path(chosen["checkpoint"]).stem
        match = [
            row
            for row in json.loads(scored.read_text())
            if Path(row["checkpoint"]).stem == stem
        ]
        if match:
            rows.append(
                {
                    "model": label,
                    "run": name,
                    "n_members": 1,
                    "resamples": ranking_resamples(),
                    **match[0],
                }
            )
    return rows


def test_explore_rows() -> list[dict[str, Any]]:
    """The compositions explored after the freeze, on the test fold.

    Every one of these is an average (or, for the hybrid, a fixed-weight blend) of
    bundles already banked under the frozen stems, so none of them touched the fold
    a second time and none re-selected a checkpoint.

    Returns:
        list[dict[str, Any]]: One row per composition that has been computed.
    """
    path = EXPLORE["testing"]
    rows: list[dict[str, Any]] = []
    for label, section, index in EXPLORE_ENSEMBLES:
        metrics = from_explore(path, section, index)
        if metrics is not None:
            rows.append({"model": label, "run": None, **metrics})
    hybrid = from_hybrid(path)
    if hybrid is not None:
        rows.append(
            {"model": "HYBRID: XGBoost 600 + LoRA 23, 50/50", "run": None, **hybrid}
        )
    return rows


def test_rows() -> list[dict[str, Any]]:
    """Every test-fold arm with its point estimate and 95% interval.

    Returns:
        list[dict[str, Any]]: One row per arm, sorted by AUPRC, or an empty list
            if the intervals have not been computed yet.
    """
    singles = test_single_rows() + test_explore_rows()
    if not TEST_INTERVALS.is_file():
        probe = test_probe_row()
        return sorted(
            singles + ([probe] if probe else []),
            key=lambda row: row["auprc"],
            reverse=True,
        )
    report = json.loads(TEST_INTERVALS.read_text())
    rows = [
        {
            "model": label,
            "run": TEST_INTERVAL_SINGLES.get(label),
            "n_members": arm["n_members"],
            "resamples": report["resamples"],
            **arm["point"],
            **{f"{key}_lo": band["lo"] for key, band in arm["interval"].items()},
            **{f"{key}_hi": band["hi"] for key, band in arm["interval"].items()},
        }
        for label, arm in report["arms"].items()
    ]
    # one model reported twice, under two labels and two budgets, reads as two arms
    covered = set(TEST_INTERVAL_SINGLES.values())
    kept = [row for row in singles if row["run"] not in covered]
    probe = test_probe_row()
    return sorted(
        rows + kept + ([probe] if probe else []),
        key=lambda row: row["auprc"],
        reverse=True,
    )


def paired_test_rows() -> list[dict[str, Any]]:
    """The banked paired deltas on the test fold, LoRA-vs-monolithic and vs the tree.

    Returns:
        list[dict[str, Any]]: One row per pairing, each carrying the AUPRC, AUROC and
            ECE deltas with their intervals. Pairings whose artifact is missing are
            skipped rather than faked.
    """
    rows: list[dict[str, Any]] = []

    if TEST_PAIRED.is_file():
        paired = json.loads(TEST_PAIRED.read_text())["paired"]
        for key, label in TEST_PAIRINGS:
            if key in paired:
                rows.append({"pairing": label, **paired[key]})

    if TEST_XGBOOST.is_file():
        # this file's deltas sit at the top level and are ensemble MINUS xgboost,
        # the same orientation as the rows above
        rows.append(
            {
                "pairing": "LoRA 23 - XGBoost, 600 rounds",
                **{
                    metric: {"value": band["value"], "lo": band["lo"], "hi": band["hi"]}
                    for metric, band in json.loads(TEST_XGBOOST.read_text())[
                        "paired"
                    ].items()
                },
            }
        )
    return rows


def format_test_table(rows: list[dict[str, Any]]) -> str:
    """The test-fold arms as a markdown table, AUPRC and AUROC carrying intervals."""
    lines = [
        "| arm | members | AUPRC [95% CI] | AUROC [95% CI] | Brier | ECE | p@1% "
        "| resamples |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def band(row: dict[str, Any], key: str) -> str:
        """One metric with its interval, or bare where no interval was computed."""
        low, high = row.get(f"{key}_lo"), row.get(f"{key}_hi")
        if isinstance(low, float) and isinstance(high, float):
            return f"{row[key]:.5f} [{low:.5f}, {high:.5f}]"
        return f"{row[key]:.5f}"

    for row in rows:
        lines.append(
            f"| {row['model']} | {row['n_members']} | {band(row, 'auprc')} | "
            f"{band(row, 'auroc')} | {row['brier']:.5f} | {row['ece']:.5f} | "
            f"{row['precision_at_1pct']:.4f} | {row['resamples']:,} |"
        )
    lines.append("")
    lines.append(
        "Intervals rest on different numbers of subject-level resamples, so the "
        f"column is reported: a band at {ranking_resamples():,} draws is a ladder's "
        "own and is coarser than one at 2,000. Point estimates are unaffected."
    )
    return "\n".join(lines)


def format_paired_table(rows: list[dict[str, Any]]) -> str:
    """The paired test-fold deltas, one row per comparison.

    A delta whose interval excludes zero is marked, because that is the only claim
    the write-up may make; a point estimate on its own is not a result.
    """
    lines = [
        "| comparison | AUPRC delta [95% CI] | AUROC delta [95% CI] "
        "| ECE delta [95% CI] |",
        "|---|---|---|---|",
    ]

    def cell(row: dict[str, Any], metric: str) -> str:
        """One delta with its interval, starred where the interval clears zero."""
        band = row.get(metric)
        if not isinstance(band, dict):
            return "—"
        clears = band["lo"] > 0 or band["hi"] < 0
        return (
            f"{band['value']:+.5f} [{band['lo']:+.5f}, {band['hi']:+.5f}]"
            f"{' *' if clears else ''}"
        )

    for row in rows:
        lines.append(
            f"| {row['pairing']} | {cell(row, 'auprc')} | {cell(row, 'auroc')} | "
            f"{cell(row, 'ece')} |"
        )
    lines.append("")
    lines.append("`*` the interval excludes zero. ECE is an error, so a NEGATIVE")
    lines.append("delta favours the left arm; AUPRC and AUROC favour it when POSITIVE.")
    return "\n".join(lines)


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
    tests = test_rows()
    pairings = paired_test_rows()

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "fold": "validation",
        "checkpoint_rule": "minimum validation loss",
        "n_models": len(rows),
        "models": rows,
        # the test fold is a SEPARATE block, never merged into `models`: it was
        # scored once under a frozen design and its rows must not be re-sorted
        # alongside validation numbers a reader could mistake them for
        "test_fold": {
            "scored": test_provenance(test_sources()),
            "arms": tests,
            "paired": pairings,
        },
    }
    args.dest.write_text(json.dumps(summary, indent=2))

    document = ["## Validation fold", "", format_table(rows), "", "## Test fold", ""]
    if tests:
        document += [format_test_table(tests), ""]
    else:
        document += [
            f"No `{TEST_INTERVALS.name}` yet — run "
            "`scripts/evaluate/test_fold_intervals.py` to compute the arm intervals.",
            "",
        ]
    if pairings:
        document += [
            "### Paired differences, test fold",
            "",
            format_paired_table(pairings),
            "",
        ]

    rendered = "\n".join(document)
    print(rendered)
    (args.dest.with_suffix(".md")).write_text(rendered)
    print(f"wrote {args.dest}")
    print(f"wrote {args.dest.with_suffix('.md')}")


if __name__ == "__main__":
    main()
