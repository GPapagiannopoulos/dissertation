r"""Score the ALREADY-FITTED linear probe on the test fold.

Run from the repo root with the modelling environment's interpreter:

    .venv-modelling/bin/python scripts/evaluate/score_probe_test.py \
        --probe motor_output/probe-ng

Two phases, mirroring `run_linear_probe.py`. Phase one runs the frozen released
encoder over the TEST fold and caches each labelled position's 768-wide output;
phase two applies the head already saved in `probe.pt` and scores it.

**Nothing is fitted here.** The head, its weight decay and its epoch were all
selected on validation and are loaded as-is. Re-fitting or re-sweeping on the test
fold would be the selection optimism the project's freeze exists to prevent, and it
would make this number incomparable to every other test-fold arm.

The probe was NOT part of the arm set frozen in REPORT_HANDOVER.md section 7; it is
scored here as a capacity floor for the results table and must be reported as added
after the freeze. It competes with nothing, so it moves no claim either way.

Needs a GPU for phase one, roughly 15 minutes, and writes ~0.95 GB of features.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from thesis.modelling.backbone.checkpoint import released_encoder
from thesis.modelling.evaluation.intervals import bootstrap_interval
from thesis.modelling.evaluation.metrics import binary_metrics
from thesis.modelling.finetune.probe import _evaluate_head

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "scripts" / "train" / "run_linear_probe.py"
FOLD = "testing"
METRICS = ("auprc", "auroc", "brier", "ece", "precision_at_1pct")


def _driver():
    """Imports the probe driver, so feature extraction is the identical code path.

    Returns:
        module: `run_linear_probe`, loaded from its path rather than by name --
            `scripts/` is not a package and has no `__init__.py`.
    """
    spec = importlib.util.spec_from_file_location("run_linear_probe", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    """Reads the probe folder and the paths phase one needs."""
    driver = _driver()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe", type=Path, default=ROOT / "motor_output" / "probe-ng"
    )
    parser.add_argument("--sequences", type=Path, default=driver.SEQUENCES)
    parser.add_argument("--split", type=Path, default=driver.SPLIT)
    parser.add_argument("--oracle", type=Path, default=driver.ORACLE)
    parser.add_argument("--dictionary", type=Path, default=driver.DICTIONARY)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Caches the test features if needed, then scores the frozen head."""
    args = _parse_args()
    driver = _driver()
    args.dest = args.probe

    checkpoint = args.probe / "probe.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{checkpoint} does not exist; fit the probe first.")

    cached = all(path.is_file() for path in driver._cache_paths(args.probe, FOLD))
    if args.refresh or not cached:
        print(f"== phase 1: frozen forward pass over {FOLD!r}", flush=True)
        encoder = released_encoder(args.oracle)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder.to(device)
        driver._build_cache(args, encoder, FOLD)
        del encoder
        torch.cuda.empty_cache()
    else:
        print("== phase 1: cached test features found, skipping", flush=True)

    print("== phase 2: applying the head fitted on validation", flush=True)
    arrays = driver._load_cached(args.probe, FOLD)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head = torch.nn.Linear(arrays.features.shape[1], 1)
    head.load_state_dict(saved["head"])
    head.to(device).eval()
    print(f"  head selected on validation: {saved['metrics']}", flush=True)

    scores, loss = _evaluate_head(
        head, arrays, batch_size=args.batch_size, device=device
    )

    report = binary_metrics(scores, arrays.targets.astype(float))
    report["loss"] = loss
    for metric in METRICS:
        low, high = bootstrap_interval(
            scores,
            arrays.targets.astype(float),
            arrays.subjects,
            metric=metric,
            resamples=args.resamples,
            seed=args.seed,
        )
        report[f"{metric}_lo"], report[f"{metric}_hi"] = low, high

    np.savez(
        args.probe / f"{FOLD}_predictions.npz",
        scores=scores,
        targets=arrays.targets,
        subjects=arrays.subjects,
        times=arrays.times,
    )
    dest = args.probe / "probe_metrics_test.json"
    dest.write_text(
        json.dumps(
            {
                "fold": FOLD,
                "head": "fitted on training, selected on validation loss; NOT refitted",
                "note": "not part of the frozen arm set; added after the freeze",
                "resamples": args.resamples,
                "seed": args.seed,
                "metrics": report,
            },
            indent=1,
        )
    )
    print(
        f"\n{FOLD}: AUPRC {report['auprc']:.5f} "
        f"[{report['auprc_lo']:.5f}, {report['auprc_hi']:.5f}] | "
        f"AUROC {report['auroc']:.5f} | ECE {report['ece']:.5f}"
    )
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
