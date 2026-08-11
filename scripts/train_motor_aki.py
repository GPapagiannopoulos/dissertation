"""Stage 6 driver: fine-tune the MOTOR backbone on the AKI landmark task.

Run it from the repo root with the modelling interpreter:

    .venv-modelling/bin/python scripts/train_motor_aki.py \
        --dest motor_output/runs/aki-seed0 --seed 0 --max-hours 9

As elsewhere, this resolves paths, prints them, calls the one library function and
lets exceptions propagate -- the traceback is the error report for a hand-run job.
"""

import argparse
import itertools
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import torch

from thesis.modelling.motor.checkpoint import released_encoder
from thesis.modelling.motor.data import (
    DEFAULT_TOKEN_BUDGET,
    fold_subjects,
    iter_epoch,
)
from thesis.modelling.motor.head import MotorClassifier
from thesis.modelling.motor.tokenizer import build_ancestor_expansion, load_token_table
from thesis.modelling.motor.training import (
    positive_rate,
    run_training,
    validation_stream,
)

ROOT = Path(__file__).resolve().parent.parent
SEQUENCES = ROOT / "meds_output" / "sequences"
SPLIT = ROOT / "meds_output" / "labels" / "subject_split.parquet"
DICTIONARY = ROOT / "motor_model" / "dictionary"
ORACLE = ROOT / "motor_output" / "oracle_fp32.npz"
VOCAB_SIZE = 65536


def _parse_args() -> argparse.Namespace:
    """Reads the run's configuration off the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=400,
        help="batches per in-loop evaluation. A PROGRESS SIGNAL, not a selection: "
        "400 costs ~55s per evaluation (under 10%% of the run) and gives a steadier "
        "line than 150 did, but no subsample ranks checkpoints -- "
        "scripts/score_checkpoints.py does that against the whole fold",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="stop after this many EVALUATIONS with no validation-loss drop. "
        "Defaults to 0 (disabled): the 15,000-step run improved through 24 "
        "consecutive subsample stalls, so patience truncates the checkpoint sweep "
        "on evidence that cannot support it",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="how far validation loss must DROP to reset patience. Zero, because "
        "loss moves in the third decimal -- the previous run gained 0.0049 in total "
        "between steps 1,000 and 10,000 -- so the 0.002 floor that suited AUPRC's "
        "noise would stop the run before it improved at all",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=4,
        help="save a step_NNNNNN.pt every this many EVALUATIONS regardless of score, "
        "so a selection made on the subsample can be revisited against the full fold",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="upper bound; the step "
        "count or the wall clock is what actually stops the run",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="run the encoder eagerly. Compiled is the default: measured at 311.7 "
        "vs 364.8 ms/batch (1.17x) and 3.10 vs 3.83 GiB, with a max logit "
        "difference of 0.0 -- bit-identical. This escape hatch exists because "
        "compilation adds a failure mode that does not exist eagerly",
    )
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    """Pins every generator the run touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _training_stream(
    expansion: pl.LazyFrame, subjects: pl.LazyFrame, seed: int, epochs: int, budget: int
) -> Iterator[dict[str, torch.Tensor | int]]:
    """Chains epochs into one stream, each shuffled differently."""
    return itertools.chain.from_iterable(
        iter_epoch(
            SEQUENCES,
            expansion,
            subjects=subjects,
            token_budget=budget,
            seed=seed * 100 + epoch,
        )
        for epoch in range(epochs)
    )


def main() -> None:
    """Builds everything the loop needs and runs it."""
    args = _parse_args()
    _seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"sequences   {SEQUENCES}")
    print(f"split       {SPLIT}")
    print(f"checkpoint  {ORACLE}")
    print(f"destination {args.dest}")
    print(f"device      {device} seed {args.seed}")

    table = load_token_table(DICTIONARY, vocab_size=VOCAB_SIZE)
    # collected once and re-wrapped: it is joined per batch and rebuilding the plan
    # 300,000 times over a run is pure overhead
    expansion = build_ancestor_expansion(table).collect().lazy()

    train_subjects = fold_subjects(SPLIT, "training").collect().lazy()
    val_subjects = fold_subjects(SPLIT, "validation").collect().lazy()

    prevalence = positive_rate(SEQUENCES / "labels", SPLIT, "training")
    print(f"training prevalence {prevalence:.4%}")

    model = MotorClassifier(released_encoder(ORACLE), positive_rate=prevalence)
    model.to(device)
    print(f"parameters  {sum(p.numel() for p in model.parameters()):,}")

    if not args.no_compile:
        # `dynamic=True` is load-bearing, not a default worth copying past. Batches
        # are built to a token budget and padded to the next power of two, so 24
        # consecutive batches carried 22 distinct shapes across six length buckets.
        # A static compile would specialise on each and spend more time compiling
        # than it saves.
        #
        # The ENCODER only. `MotorClassifier.forward` raises on several guards --
        # the empty-label check, the out-of-range gather, the clock width -- and a
        # raise inside a traced region is a graph break. Those guards turn what
        # would be a device-side assert, which kills the CUDA context, into an
        # exception that kills one step.
        model.encoder = torch.compile(model.encoder, dynamic=True)
        print("encoder     compiled (dynamic=True); first steps include warmup")

    best = run_training(
        model,
        _training_stream(
            expansion, train_subjects, args.seed, args.epochs, args.token_budget
        ),
        validation_stream(
            lambda: iter_epoch(
                SEQUENCES,
                expansion,
                subjects=val_subjects,
                token_budget=args.token_budget,
                seed=args.seed,
            )
        ),
        args.dest,
        device=device,
        total_steps=args.total_steps,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        warmup=args.warmup,
        accumulate=args.accumulate,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        checkpoint_every=args.checkpoint_every,
        patience=args.patience or None,
        min_delta=args.min_delta,
        max_hours=args.max_hours,
    )
    print(f"best {best}")


if __name__ == "__main__":
    main()
