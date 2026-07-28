"""Helpers for mapping MIMIC-IV native codes to the standardized SSSOMOP of MOTOR."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import msgpack
import polars as pl


def code_inventory(events: pl.LazyFrame) -> pl.LazyFrame:
    """Determines the number of events per code in the dataset."""
    return (
        events.group_by(pl.col("code"))
        .agg(pl.len().alias("count"))
        .sort(pl.col("code"), nulls_last=True)
    )


@dataclass(frozen=True)
class MotorVocab:
    """Custom class representing the vocabulary of MOTOR."""

    code_tokens: frozenset[str]
    numeric_codes: frozenset[str]
    text_codes: frozenset[str]
    all_parents: Mapping[str, tuple]


def load_motor_vocab(dictionary_path: Path, *, vocab_size: int) -> MotorVocab:
    """Extracts the vocabulary of MOTOR.

    The released dictionary holds a candidate list far longer than the model's
    vocabulary; only its first vocab_size entries are real tokens, their position
    in the list being the embedding row they index. Entries past the cut, and
    entries of the unused type, never reach the model.

    Args:
        dictionary_path: Path to the msgpack dictionary shipped with the weights
        vocab_size: Number of leading entries that form the vocabulary, as
            declared by the model's own config

    Returns:
        MotorVocab: the token sets, split by entry type, plus the full ancestor map

    Raises:
        ValueError: If vocab_size is not positive
        ValueError: If vocab_size exceeds the entries the dictionary holds
    """
    if vocab_size <= 0:
        raise ValueError(f"The minimum vocabulary size is 1. Received {vocab_size}")
    with open(dictionary_path, "rb") as f:
        dictionary = msgpack.load(f, use_list=False, strict_map_key=False)
    rollup = dictionary["ontology_rollup"]
    if vocab_size > len(rollup):
        raise ValueError(
            f"vocab_size {vocab_size} exceeds the {len(rollup)} entries in "
            f"{dictionary_path}. The tokens would not line up with the model's "
            f"embedding table; check the vocab_size the model config declares."
        )
    vocab = rollup[:vocab_size]
    all_parents = dictionary["all_parents"]

    code_tokens: set[str] = set()
    numeric_codes: set[str] = set()
    text_codes: set[str] = set()

    for vocab_entry in vocab:
        match vocab_entry["type"]:
            case 0:
                code_tokens.add(vocab_entry["code_string"])
            case 1:
                numeric_codes.add(vocab_entry["code_string"])
            case 2:
                text_codes.add(vocab_entry["code_string"])
            case _:
                continue

    return MotorVocab(
        frozenset(code_tokens),
        frozenset(numeric_codes),
        frozenset(text_codes),
        all_parents,
    )
