r"""Stage 6b driver: rank one run's checkpoints against the WHOLE validation fold.

Run from the repo root with the modelling environment's interpreter:

    .venv-modelling/bin/python scripts/evaluate/score_checkpoints.py \
        --run motor_output/runs/aki-seed0-v3

This is where model selection happens, and it is deliberately not in the training
loop. The in-loop evaluation reads a fixed ~1,600-patient slice, and measured on
the 15,000-step run it ranked step 3,000 above step 10,000 (0.12939 vs 0.12970)
where the full fold puts them the other way round by ten times that margin
(0.13341 vs 0.13052). No metric choice repairs a sample that small, and enlarging
it in-loop costs more than this pass does.

Every candidate's predictions are banked so the chosen one can be paired against
another model without a second forward pass.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from thesis.modelling.evaluation.compare import score_motor

ROOT = Path(__file__).resolve().parents[2]
SEQUENCES = ROOT / "meds_output" / "sequences"
SPLIT = ROOT / "meds_output" / "labels" / "subject_split.parquet"
ORACLE = ROOT / "motor_output" / "oracle_fp32.npz"
DICTIONARY = ROOT / "motor_model" / "dictionary"


def _parse_args() -> argparse.Namespace:
    """Reads the run's configuration off the command line."""
    parser = argparse.ArgumentParser(description="Rank a run's checkpoints.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, default=SEQUENCES)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--oracle", type=Path, default=ORACLE)
    parser.add_argument("--dictionary", type=Path, default=DICTIONARY)
    parser.add_argument("--fold", default="validation")
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="score every Nth periodic checkpoint. Each costs ~12 minutes, and "
        "neighbouring saves differ by less than the metrics disagree with each "
        "other, so scoring all of them buys precision nobody uses",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        metavar="STEM",
        help="score exactly these checkpoint stems (e.g. step_018000 last) and no "
        "others, overriding --stride. This is how a frozen set is scored on the test "
        "fold: naming the set means no other test-fold number is ever produced, so "
        "there is no fuller ladder on disk to re-select from later",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="where to write the table and the banked predictions; "
        "defaults to <run>/selection",
    )
    return parser.parse_args()


def candidates(run: Path, stride: int, stems: list[str] | None = None) -> list[Path]:
    """The checkpoints to score, in step order, always including the last.

    Args:
        run (Path): A folder `run_training` wrote.
        stride (int): Keep every Nth periodic save.
        stems (list[str] | None): Score exactly these stems instead of striding.

    Returns:
        list[Path]: Checkpoint paths, ascending by step.

    Raises:
        FileNotFoundError: If the folder holds no checkpoint at all, or if a named
            stem is missing -- silently scoring a smaller set than was asked for is
            how a frozen roster quietly becomes a different one.
    """
    if stems is not None:
        named = [run / f"{stem}.pt" for stem in dict.fromkeys(stems)]
        missing = [path.name for path in named if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{run} holds no {', '.join(missing)}.")
        return named

    periodic = sorted(run.glob("step_*.pt"))[::stride]
    final = run / "last.pt"
    chosen = [*periodic, *([final] if final.is_file() else [])]
    if not chosen:
        raise FileNotFoundError(
            f"Found no checkpoints in {run}. Point --run at a folder run_training "
            f"wrote, and check that checkpoint_every was not zero."
        )
    return chosen


def lora_record(run: Path, saved: dict) -> dict | None:
    """The adapter configuration to rebuild, or None for a full fine-tune.

    Preferred from the checkpoint, which carries its own since stage 8. Runs written
    before that fall back to the manifest, which records the driver's arguments.

    Args:
        run (Path): The run folder, for the manifest fallback.
        saved (dict): A loaded checkpoint.

    Returns:
        dict | None: Keyword arguments for `lora_config`.
    """
    if "lora" in saved:
        return saved["lora"]

    manifest = run / "manifest.json"
    if not manifest.is_file():
        return None
    parameters = json.loads(manifest.read_text()).get("parameters", {})
    if parameters.get("arm") != "lora":
        return None
    return {
        "r": parameters["lora_r"],
        "lora_alpha": parameters["lora_alpha"],
        "lora_dropout": parameters["lora_dropout"],
        "target_modules": parameters["lora_targets"],
    }


def main() -> None:
    """Scores every candidate and writes the ranking."""
    args = _parse_args()
    dest = args.dest or args.run / "selection"
    dest.mkdir(parents=True, exist_ok=True)

    chosen = candidates(args.run, args.stride, args.checkpoints)
    print(f"run   {args.run}")
    print(f"fold  {args.fold}")
    print(f"scoring {len(chosen)} checkpoints at ~12 min each", flush=True)

    rows = []
    for checkpoint in chosen:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        step = int(saved["step"])
        lora = lora_record(args.run, saved)
        arm = f"lora r={lora['r']} alpha={lora['lora_alpha']}" if lora else "full"
        print(f"== {checkpoint.name} (step {step}, {arm})", flush=True)

        metrics, bundle = score_motor(
            checkpoint,
            args.sequences,
            args.split,
            args.oracle,
            args.dictionary,
            fold=args.fold,
            resamples=200,
            lora=lora,
        )
        np.savez(dest / f"{checkpoint.stem}_predictions.npz", *bundle)

        rows.append(
            {
                **metrics,
                "checkpoint": checkpoint.name,
                "step": float(step),
                "fold": args.fold,
            }
        )
        print(
            f"   loss {metrics['loss']:.5f} | auprc {metrics['auprc']:.4f} "
            f"[{metrics['auprc_lo']:.4f}, {metrics['auprc_hi']:.4f}] | "
            f"auroc {metrics['auroc']:.4f} | ece {metrics['ece']:.4f}",
            flush=True,
        )

    (dest / "checkpoint_ranking.json").write_text(json.dumps(rows, indent=2))

    header = (
        f"{'checkpoint':>18}{'step':>8}{'loss':>10}{'auprc':>9}{'auroc':>9}{'ece':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['checkpoint']:>18}{row['step']:>8.0f}{row['loss']:>10.5f}"
            f"{row['auprc']:>9.4f}{row['auroc']:>9.4f}{row['ece']:>9.4f}"
        )

    # Reported, never applied. Loss and AUPRC routinely name different winners --
    # on the 15,000-step run, steps 10,000 and 14,000 -- because they measure
    # calibration and top-of-ranking respectively. Which one decides is a thesis
    # question, and resolving it here would bury it in a script.
    print(f"\nlowest loss:   {min(rows, key=lambda r: r['loss'])['checkpoint']}")
    print(f"highest auprc: {max(rows, key=lambda r: r['auprc'])['checkpoint']}")
    print(f"wrote {dest / 'checkpoint_ranking.json'}")


if __name__ == "__main__":
    main()
