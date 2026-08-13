"""Stage 8 driver: fine-tune the AKI head and LoRA adapters over a frozen backbone.

Run it from the repo root with the modelling interpreter:

    .venv-modelling/bin/python scripts/train_motor_aki_lora.py \
        --dest motor_output/runs/lora-seed0 --seed 0 --max-hours 9

Everything but the model construction is the full fine-tune's: the same stream, the
same collate, the same loop and the same metrics, so the two arms are comparable by
construction.
"""

import argparse
import itertools
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import torch

from thesis.modelling.motor.data import (
    DEFAULT_TOKEN_BUDGET,
    bag_subjects,
    fold_subjects,
    iter_epoch,
)
from thesis.modelling.motor.lora import (
    LORA_DEFAULTS,
    adapter_state,
    config_record,
    lora_classifier,
    lora_config,
    trainable_summary,
)
from thesis.modelling.motor.manifest import build_manifest, write_manifest
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
    parser.add_argument("--total-steps", type=int, default=15000)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument(
        "--adapter-lr",
        type=float,
        default=2e-4,
        help="peak rate for the adapters, which is the optimizer's ENCODER group. "
        "Twenty times the full fine-tune's 1e-5: LoRA's B starts at zero and its "
        "update is scaled by alpha/r = 4, so the pretrained-weight rate does not "
        "transfer. The one number here most worth a short pilot",
    )
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=400)
    parser.add_argument("--checkpoint-every", type=int, default=4)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=LORA_DEFAULTS["r"])
    parser.add_argument("--lora-alpha", type=int, default=LORA_DEFAULTS["lora_alpha"])
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=LORA_DEFAULTS["lora_dropout"],
        help="the port has no dropout anywhere, so this is the arm's one source of "
        "member diversity beyond LoRA's random A initialisation",
    )
    parser.add_argument(
        "--lora-targets",
        nargs="+",
        default=list(LORA_DEFAULTS["target_modules"]),
        help="module names to adapt, matched by suffix across all twelve blocks",
    )
    parser.add_argument(
        "--bag-fraction",
        type=float,
        default=None,
        help="train on this share of the training subjects, drawn by --seed. "
        "0.632 is a bootstrap's expected distinct share. Omitted, the member sees "
        "every subject and differs from its siblings only by initialisation and "
        "batch order",
    )
    parser.add_argument("--no-compile", action="store_true")
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
    # the adapters are randomly initialised, so unlike the full fine-tune the seed
    # moves the weights as well as the batch order
    _seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"sequences   {SEQUENCES}")
    print(f"split       {SPLIT}")
    print(f"checkpoint  {ORACLE}")
    print(f"destination {args.dest}")
    print(f"device      {device} seed {args.seed}")

    table = load_token_table(DICTIONARY, vocab_size=VOCAB_SIZE)
    expansion = build_ancestor_expansion(table).collect().lazy()

    train_subjects = fold_subjects(SPLIT, "training").collect().lazy()
    if args.bag_fraction is not None:
        # the bag is seeded off the member's own seed, so members differ in DATA as
        # well as in initialisation and batch order
        train_subjects = bag_subjects(train_subjects, args.bag_fraction, args.seed)
        print(
            f"bagged      {train_subjects.select(pl.len()).collect().item():,} "
            f"subjects at fraction {args.bag_fraction}"
        )
    # never bagged: every member is scored on the same fold, or the numbers are not
    # comparable to each other or to anything else in the project
    val_subjects = fold_subjects(SPLIT, "validation").collect().lazy()

    prevalence = positive_rate(SEQUENCES / "labels", SPLIT, "training")
    print(f"training prevalence {prevalence:.4%}")

    config = lora_config(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=tuple(args.lora_targets),
    )
    model = lora_classifier(ORACLE, prevalence, config)
    model.to(device)

    summary = trainable_summary(model)
    print(
        f"adapters    {summary['adapters']:.0f} at r={args.lora_r} "
        f"alpha={args.lora_alpha} on {', '.join(args.lora_targets)}"
    )
    print(
        f"parameters  {summary['trainable']:,.0f} trainable of "
        f"{summary['total']:,.0f} ({summary['fraction']:.3%})"
    )

    if not args.no_compile:
        # after the wrap, never before: the adapters have to be inside the graph
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
        encoder_lr=args.adapter_lr,
        head_lr=args.head_lr,
        warmup=args.warmup,
        accumulate=args.accumulate,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        checkpoint_every=args.checkpoint_every,
        patience=args.patience or None,
        min_delta=args.min_delta,
        max_hours=args.max_hours,
        # the adapters and the head alone; the frozen backbone is rebuilt from the
        # oracle at load time, so a checkpoint carrying it would be 456x the size
        save_state=adapter_state,
        # so every file says what has to be rebuilt to load it, from step one --
        # alpha scales the adapter and cannot be read back off the saved tensors
        checkpoint_extra={"lora": config_record(config)},
    )
    print(f"best {best}")

    # the manifest is what a scorer reads to rebuild this run's wrap, so the LoRA
    # arguments have to reach it -- they are argparse's, and ride in for free
    manifest = build_manifest(
        args.dest,
        parameters={
            **{key: value for key, value in vars(args).items() if key != "dest"},
            "compiled": not args.no_compile,
            "arm": "lora",
            "dest": str(args.dest),
        },
        provenance="runtime",
        repo_root=ROOT,
    )
    print(f"wrote {write_manifest(args.dest, manifest)}")


if __name__ == "__main__":
    main()
