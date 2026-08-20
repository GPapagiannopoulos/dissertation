r"""Diagnostic driver: linear-probe the frozen MOTOR backbone on the AKI task.

Run from the repo root with the modelling environment's interpreter:

    .venv-modelling/bin/python scripts/train/run_linear_probe.py \
        --dest motor_output/probe

Two phases, and the first is cached. Phase one runs the **frozen released
encoder** over the training and validation folds and writes each labelled
position's 768-wide output to a memory-mapped file. Phase two fits a linear head
on that table, sweeping weight decay and selecting on **validation loss**.

Re-running with the cache already present skips phase one, so the head can be
re-fit in minutes. Pass `--refresh` to force the forward passes again.

The encoder is the released one, not a fine-tuned checkpoint -- the question is
what MOTOR's *pretrained* representation carries, before any adaptation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from thesis.modelling.backbone.checkpoint import released_encoder
from thesis.modelling.backbone.tokenizer import (
    build_ancestor_expansion,
    load_token_table,
)
from thesis.modelling.finetune.data import fold_subjects, iter_epoch
from thesis.modelling.finetune.probe import (
    FEATURE_DTYPE,
    ProbeArrays,
    extract_features,
    fold_label_count,
    run_fit_probe,
)

ROOT = Path(__file__).resolve().parents[2]
SEQUENCES = ROOT / "meds_output" / "sequences"
SPLIT = ROOT / "meds_output" / "labels" / "subject_split.parquet"
ORACLE = ROOT / "motor_output" / "oracle_fp32.npz"
DICTIONARY = ROOT / "motor_model" / "dictionary"
DEST = ROOT / "motor_output" / "probe"

HIDDEN_SIZE = 768


def _parse_args() -> argparse.Namespace:
    """Reads the run's configuration off the command line."""
    parser = argparse.ArgumentParser(description="Linear-probe the MOTOR backbone.")
    parser.add_argument("--sequences", type=Path, default=SEQUENCES)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--oracle", type=Path, default=ORACLE)
    parser.add_argument("--dictionary", type=Path, default=DICTIONARY)
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recompute the cached features even if they are already on disk.",
    )
    return parser.parse_args()


def _cache_paths(dest: Path, fold: str) -> tuple[Path, Path]:
    """Where one fold's features and its label sidecar live."""
    return dest / f"{fold}_features.npy", dest / f"{fold}_labels.npz"


def _load_cached(dest: Path, fold: str) -> ProbeArrays:
    """Reads one fold's cache back, leaving the features on disk as a map."""
    features_path, labels_path = _cache_paths(dest, fold)
    sidecar = np.load(labels_path)
    return ProbeArrays(
        features=np.load(features_path, mmap_mode="r"),
        targets=sidecar["targets"],
        subjects=sidecar["subjects"],
        times=sidecar["times"],
    )


def _build_cache(args: argparse.Namespace, encoder: torch.nn.Module, fold: str) -> None:
    """Runs the frozen encoder over one fold and writes its features to disk."""
    features_path, labels_path = _cache_paths(args.dest, fold)

    expected = fold_label_count(args.sequences, args.split, fold)
    print(f"  {fold}: {expected:,} labels -> {features_path.name}", flush=True)

    # opened as a memory map so the fold's several GB never enter the heap; the
    # host routinely has the XGBoost baseline holding 9.5 of its 14 GB
    destination = np.lib.format.open_memmap(
        features_path, mode="w+", dtype=FEATURE_DTYPE, shape=(expected, HIDDEN_SIZE)
    )

    table = load_token_table(args.dictionary, vocab_size=args.vocab_size)
    expansion = build_ancestor_expansion(table).collect().lazy()
    subjects = fold_subjects(args.split, fold).collect().lazy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arrays = extract_features(
        encoder,
        iter_epoch(
            args.sequences,
            expansion,
            subjects=subjects,
            token_budget=args.token_budget,
            seed=0,
        ),
        destination,
        device=device,
        amp_dtype=torch.bfloat16,
    )

    if arrays.targets.size != expected:
        raise ValueError(
            f"Fold {fold!r} was sized for {expected:,} labels but the stream "
            f"produced {arrays.targets.size:,}."
        )

    destination.flush()
    np.savez(
        labels_path,
        targets=arrays.targets,
        subjects=arrays.subjects,
        times=arrays.times,
    )
    print(
        f"  {fold}: cached, base rate {arrays.targets.mean():.4%}",
        flush=True,
    )


def main() -> None:
    """Caches the frozen features if needed, then fits the head."""
    args = _parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    print(f"sequences {args.sequences}")
    print(f"dest      {args.dest}")

    folds = ("training", "validation")
    missing = [
        fold
        for fold in folds
        if args.refresh
        or not all(path.is_file() for path in _cache_paths(args.dest, fold))
    ]

    if missing:
        print(f"== phase 1: frozen forward passes for {missing}", flush=True)
        encoder = released_encoder(args.oracle)
        encoder.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        for fold in missing:
            _build_cache(args, encoder, fold)
        del encoder
        torch.cuda.empty_cache()
    else:
        print("== phase 1: cached features found, skipping forward passes", flush=True)

    print("== phase 2: fitting the head on validation loss", flush=True)
    train = _load_cached(args.dest, "training")
    validation = _load_cached(args.dest, "validation")
    print(
        f"train {train.features.shape[0]:,} x {train.features.shape[1]} | "
        f"valid {validation.features.shape[0]:,}",
        flush=True,
    )

    best = run_fit_probe(
        train,
        validation,
        args.dest,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        epochs=args.epochs,
        seed=args.seed,
    )
    print(f"\nbest probe: {json.dumps(best)}")
    print(f"wrote {args.dest}")


if __name__ == "__main__":
    main()
