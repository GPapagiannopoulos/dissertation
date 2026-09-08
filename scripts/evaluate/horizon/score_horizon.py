"""Re-score banked predictions against a different prediction horizon.

Run from the repo root:

    .venv-modelling/bin/python scripts/evaluate/horizon/score_horizon.py \
        --runs motor_output/runs/lora-cfg-* motor_output/runs/lora-sched-seed1 \
        --labels meds_output/labels/landmark_labels_72h.parquet

Needs no GPU and no forward pass. The landmark grid does not depend on the horizon --
it comes from the admission windows at a fixed spacing, and `apply_time_horizons`
only rewrites `boolean_value` -- so a model's scores at 72h are the same numbers it
already produced at 48h. Only the targets move.

What that measures is ZERO-SHOT horizon transfer: a model trained to forecast 48h is
asked to rank 72h or 7d risk. Ranking should transfer, since anyone at high 48h risk
is at high 72h risk; calibration should not, because prevalence rises with the
window. Both are reported.

AUPRC is not comparable across horizons -- it moves with the base rate -- so `lift`
(AUPRC / base rate) and AUROC are the cross-horizon numbers.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from thesis.modelling.ensemble.diversity import align_members, load_member
from thesis.modelling.evaluation.metrics import binary_metrics

ROOT = Path(__file__).resolve().parents[3]
SPLIT = ROOT / "meds_output" / "labels" / "subject_split.parquet"


def best_bundle(run: Path) -> Path:
    """The prediction npz for a run's lowest-LOSS scored checkpoint.

    Loss rather than AUPRC: selecting on the metric being reported is optimistic,
    and the two rules disagree on half the runs here.

    Args:
        run (Path): A run folder holding `selection/checkpoint_ranking.json`.

    Returns:
        Path: The banked npz for that checkpoint.

    Raises:
        FileNotFoundError: If the run has not been scored.
    """
    ranking = run / "selection" / "checkpoint_ranking.json"
    if not ranking.is_file():
        raise FileNotFoundError(f"{run} has no checkpoint_ranking.json; score it.")
    rows = json.loads(ranking.read_text())
    best = min(rows, key=lambda row: row["loss"])
    return run / "selection" / f"{Path(best['checkpoint']).stem}_predictions.npz"


def horizon_targets(
    labels: Path, split: Path, fold: str, subjects: np.ndarray, times: np.ndarray
) -> np.ndarray:
    """The labels of a different horizon, on the banked rows' own order.

    Args:
        labels (Path): A `build_labels.py` output at some horizon.
        split (Path): The subject split.
        fold (str): Which fold the bundles were scored on.
        subjects (np.ndarray): Banked subject ids, in bundle order.
        times (np.ndarray): Banked prediction times, microseconds, in bundle order.

    Returns:
        np.ndarray: `boolean_value` as float, one per banked row.

    Raises:
        ValueError: If any banked landmark is missing from the label set, which
            would mean the grid moved and the two are not comparable.
    """
    fold_subjects = (
        pl.scan_parquet(split).filter(pl.col("fold") == fold).select("subject_id")
    )
    fresh = (
        pl.scan_parquet(labels)
        .join(fold_subjects, on="subject_id", how="inner")
        .select(
            "subject_id",
            pl.col("prediction_time").cast(pl.Int64).alias("time"),
            pl.col("boolean_value").cast(pl.Float64).alias("target"),
        )
        .unique(subset=["subject_id", "time"])
    )
    banked = pl.DataFrame(
        {"subject_id": subjects, "time": times, "row": np.arange(len(subjects))}
    )
    joined = (
        banked.lazy()
        .join(fresh, on=["subject_id", "time"], how="left")
        .sort("row")
        .collect()
    )
    missing = int(joined["target"].null_count())
    if missing:
        raise ValueError(
            f"{missing} of {len(subjects)} banked landmarks are absent from {labels}. "
            f"The landmark grid differs, so the horizons are not comparable."
        )
    return joined["target"].to_numpy()


def summarise(scores: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Metrics plus the base-rate lift that survives a horizon change."""
    metrics = binary_metrics(scores, targets)
    base = float(targets.mean())
    return {
        "auprc": float(metrics["auprc"]),
        "auroc": float(metrics["auroc"]),
        "ece": float(metrics["ece"]),
        "base_rate": base,
        "lift": float(metrics["auprc"]) / base,
    }


def _parse_args() -> argparse.Namespace:
    """Reads the members and the horizon to re-score against."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+")
    parser.add_argument("--bundles", type=Path, nargs="+")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--fold", default="validation")
    parser.add_argument("--dest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Re-scores every member and their mean at the new horizon."""
    args = _parse_args()
    if not args.runs and not args.bundles:
        raise SystemExit("Pass --runs or --bundles.")

    paths = list(args.bundles or []) + [best_bundle(run) for run in args.runs or []]
    # the run folder ALONE is not unique: a run contributing two checkpoints, or two
    # bundles sitting in one comparison folder, both collapse onto one key and the
    # later row silently overwrites the earlier. That is how the XGBoost row was lost
    # from `horizon_72h_comparators.json` and `horizon_7d_comparators.json`.
    names = [
        f"{path.parents[1].name}/{path.stem.removesuffix('_predictions')}"
        for path in paths
    ]
    members = [load_member(path) for path in paths]
    scores, trained_targets, subjects = align_members(members)

    first = members[0]
    order = np.lexsort((first.times, first.subjects))
    times = first.times[order]

    fresh = horizon_targets(args.labels, args.split, args.fold, subjects, times)
    print(f"{len(names)} members, {len(fresh):,} landmarks, fold {args.fold}")
    print(f"trained horizon base rate {trained_targets.mean() * 100:.2f}%")
    print(f"{args.labels.name} base rate {fresh.mean() * 100:.2f}%\n")

    rows = {}
    header = f"{'model':<26} {'auprc':>8} {'auroc':>8} {'lift':>7} {'ece':>8}"
    print(f"{header}     (trained horizon -> new horizon)")
    for name, row in zip(names, scores, strict=True):
        old, new = summarise(row, trained_targets), summarise(row, fresh)
        rows[name] = {"trained": old, "horizon": new}
        print(
            f"{name:<26} {new['auprc']:>8.4f} {new['auroc']:>8.4f} "
            f"{new['lift']:>7.2f} {new['ece']:>8.4f}     "
            f"auprc {old['auprc']:.4f}->{new['auprc']:.4f}  "
            f"auroc {old['auroc']:.4f}->{new['auroc']:.4f}"
        )

    mean = scores.mean(axis=0)
    old, new = summarise(mean, trained_targets), summarise(mean, fresh)
    rows["ENSEMBLE"] = {"trained": old, "horizon": new}
    print(
        f"\n{'ENSEMBLE (mean)':<26} {new['auprc']:>8.4f} {new['auroc']:>8.4f} "
        f"{new['lift']:>7.2f} {new['ece']:>8.4f}     "
        f"auprc {old['auprc']:.4f}->{new['auprc']:.4f}  "
        f"auroc {old['auroc']:.4f}->{new['auroc']:.4f}"
    )

    if args.dest:
        args.dest.parent.mkdir(parents=True, exist_ok=True)
        args.dest.write_text(
            json.dumps({"labels": str(args.labels), "models": rows}, indent=2)
        )
        print(f"\nwrote {args.dest}")


if __name__ == "__main__":
    main()
