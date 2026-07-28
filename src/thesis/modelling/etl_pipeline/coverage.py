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
    """Extraccts the vocabulary of MOTOR."""
    with open(dictionary_path, "rb") as f:
        dictionary = msgpack.load(f, use_list=False, strict_map_key=False)
    vocab = dictionary["ontology_rollup"][:vocab_size]
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
