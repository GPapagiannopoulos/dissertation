r"""Writes a `manifest.json` for a run that predates runtime manifests.

Run from the repo root with the modelling environment's interpreter:

    .venv-modelling/bin/python scripts/write_manifest.py \
        --run motor_output/runs/aki-seed1 \
        --params '{"total_steps": 15000, "compiled": true}'

Everything derivable is derived: wall clock, active clock with suspends
subtracted, seconds per step, peak VRAM, the evaluation trajectory, the
checkpoint inventory, the environment and the git state. The learning-rate
schedule is reconstructed from the logged `lr` column, since the warmup is
linear and the decay a cosine over the step budget.

What no artifact records is `accumulate`, `token_budget`, `epochs` and whether
the encoder was compiled. Those come in through `--params`, and anything not
supplied is written as `null` rather than guessed -- a manifest that quietly
invents a hyperparameter is worse than one with a hole in it.

The git state is read at the time this runs, NOT at the time the run trained.
For a run whose code has since changed that is a lie of omission, so the file
records `provenance: "reconstructed"` and a reader should treat `git` as an
upper bound on freshness.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from thesis.modelling.motor.manifest import build_manifest, write_manifest

ROOT = Path(__file__).resolve().parents[1]

RECOVERABLE = ("encoder_lr", "warmup", "total_steps", "eval_every", "eval_batches")
"""Parameters this script infers from the log rather than needing to be told."""

UNRECOVERABLE = ("accumulate", "token_budget", "epochs", "compiled", "head_lr", "seed")
"""Parameters that leave no trace in any artifact and must be supplied."""


def _parse_args() -> argparse.Namespace:
    """Reads the run's configuration off the command line."""
    parser = argparse.ArgumentParser(description="Write a run manifest post-hoc.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--params",
        type=str,
        default="{}",
        help="JSON object of parameters no artifact records: "
        f"{', '.join(UNRECOVERABLE)}",
    )
    return parser.parse_args()


def recover_schedule(log_path: Path) -> dict[str, Any]:
    """Reconstructs the learning-rate schedule from the logged rates.

    `learning_rate_at` ramps linearly to `peak` over `warmup` steps, then decays
    as a cosine to zero over the remainder. Both halves are invertible from the
    `lr` column, which is logged every 50 steps.

    Args:
        log_path (Path): A run's `log.jsonl`.

    Returns:
        dict[str, Any]: `encoder_lr`, `warmup` and `total_steps`, each null if
            the log is too short to identify it. The warmup is only resolvable to
            the 50-step logging granularity, which is recorded in
            `warmup_resolution`.
    """
    records = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if line and '"train"' in line
    ]
    rates = [(r["step"], r["lr"]) for r in records if r.get("lr")]
    if len(rates) < 3:
        return dict.fromkeys(("encoder_lr", "warmup", "total_steps"))

    peak_step, peak = max(rates, key=lambda pair: pair[1])

    # During warmup lr = peak * (step + 1) / warmup, so the first logged rate
    # pins the warmup length; after it, the maximum logged rate IS the peak.
    first_step, first_rate = rates[0]
    warmup = None
    if first_rate < peak:
        warmup = round(peak * (first_step + 1) / first_rate)

    # lr = peak * 0.5 * (1 + cos(pi * progress)), inverted at the last record
    last_step, last_rate = rates[-1]
    total = None
    if warmup and last_step > warmup and 0 < last_rate < peak:
        progress = math.acos(2.0 * last_rate / peak - 1.0) / math.pi
        if progress > 0:
            total = round(warmup + (last_step - warmup) / progress)

    return {
        "encoder_lr": peak,
        "warmup": warmup,
        "total_steps": total,
        "warmup_resolution": 50,
        "peak_observed_at_step": peak_step,
    }


def recover_evaluation(log_path: Path) -> dict[str, Any]:
    """The evaluation cadence and size, from the validation records themselves."""
    records = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if line and '"validation"' in line
    ]
    if not records:
        return {"eval_every": None, "evaluation_labels": None}
    steps = [r["step"] for r in records]
    return {
        "eval_every": (steps[1] - steps[0]) if len(steps) > 1 else steps[0],
        # the label count stands in for --eval-batches, which is not logged
        "evaluation_labels": records[0]["n"],
    }


def main() -> None:
    """Recovers what it can, merges what it is told, and writes the manifest."""
    args = _parse_args()
    if not args.run.is_dir():
        raise FileNotFoundError(f"No run folder at {args.run}.")

    supplied = json.loads(args.params)
    log_path = args.run / "log.jsonl"

    parameters: dict[str, Any] = dict.fromkeys(UNRECOVERABLE)
    parameters.update(recover_schedule(log_path))
    parameters.update(recover_evaluation(log_path))
    parameters.update(supplied)
    parameters["_supplied_keys"] = sorted(supplied)

    manifest = build_manifest(
        args.run, parameters=parameters, provenance="reconstructed", repo_root=ROOT
    )
    written = write_manifest(args.run, manifest)

    results = manifest["results"]
    print(f"wrote {written}")
    if "unavailable" not in results:
        print(
            f"  {results['steps']} steps in {results['wall_clock']} "
            f"(active {results['active_clock']}, {results['seconds_per_step']} s/step)"
        )
        print(
            f"  peak {results['peak_gib']} GiB, {len(manifest['checkpoints'])} "
            f"checkpoints"
        )
        if results["suspected_stalls"]:
            print(f"  {len(results['suspected_stalls'])} suspected stall(s) detected")
    missing = [key for key in UNRECOVERABLE if parameters.get(key) is None]
    if missing:
        print(f"  not recoverable, left null: {', '.join(missing)}")


if __name__ == "__main__":
    main()
