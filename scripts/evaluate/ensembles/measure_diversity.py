r"""Measure how much a set of ensemble members disagree.

Run from the repo root with the modelling interpreter:

    .venv-modelling/bin/python scripts/evaluate/ensembles/measure_diversity.py \
        --runs motor_output/runs/lora-seed0 motor_output/runs/lora-seed1 \
               motor_output/runs/lora-seed2 \
        --label seed-only

Needs no GPU: it reads the prediction bundles `score_checkpoints.py` already banked.
"""

import argparse
import json
from pathlib import Path

from thesis.modelling.ensemble.diversity import (
    diversity_report,
    format_report,
    load_member,
)

ROOT = Path(__file__).resolve().parents[3]


def _parse_args() -> argparse.Namespace:
    """Reads the member set off the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        help="run folders; each contributes its highest-AUPRC scored checkpoint",
    )
    parser.add_argument(
        "--bundles",
        type=Path,
        nargs="+",
        help="explicit prediction npz files, instead of --runs",
    )
    parser.add_argument("--label", default="members")
    parser.add_argument("--alert-budget", type=float, default=0.01)
    parser.add_argument("--dest", type=Path, default=None, help="write the report")
    return parser.parse_args()


def best_bundle(run: Path) -> Path:
    """The prediction npz for a run's highest-AUPRC checkpoint.

    Args:
        run (Path): A run folder holding `selection/checkpoint_ranking.json`.

    Returns:
        Path: The npz `score_checkpoints.py` banked for that checkpoint.

    Raises:
        FileNotFoundError: If the run has not been scored, or the npz is missing.
    """
    ranking = run / "selection" / "checkpoint_ranking.json"
    if not ranking.is_file():
        raise FileNotFoundError(
            f"{run} has no {ranking.name}; score it first with "
            f"scripts/evaluate/scoring/score_checkpoints.py."
        )
    rows = json.loads(ranking.read_text())
    best = max(rows, key=lambda row: row["auprc"])
    bundle = run / "selection" / f"{Path(best['checkpoint']).stem}_predictions.npz"
    if not bundle.is_file():
        raise FileNotFoundError(f"{bundle} is missing; re-run the scoring for {run}.")
    return bundle


def main() -> None:
    """Loads the members and prints their diversity."""
    args = _parse_args()
    if not args.runs and not args.bundles:
        raise SystemExit("Pass --runs or --bundles.")

    paths = list(args.bundles or []) + [best_bundle(run) for run in args.runs or []]
    for path in paths:
        print(f"member  {path}")

    report = diversity_report(
        [load_member(path) for path in paths], k=args.alert_budget
    )
    print()
    print(format_report(report, args.label))

    if args.dest:
        args.dest.parent.mkdir(parents=True, exist_ok=True)
        args.dest.write_text(json.dumps({"label": args.label, **report}, indent=2))
        print(f"\nwrote {args.dest}")


if __name__ == "__main__":
    main()
