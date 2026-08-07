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
    parser.add_argument("--eval-batches", type=int, default=150)
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="upper bound; the step "
        "count or the wall clock is what actually stops the run",
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
        max_hours=args.max_hours,
    )
    print(f"best {best}")


if __name__ == "__main__":
    main()
