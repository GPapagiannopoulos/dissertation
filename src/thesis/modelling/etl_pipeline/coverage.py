"""Stage 2.5: maps the codes our MEDS data emits onto codes MOTOR understands.

Every layer produces one code shape <vocabulary_id>/<concept_code> (e.g.
LOINC/2160-0). What differs is whether OMOP considers a concept standard, and
whether MOTOR kept it. MEDS standardises the container, not the vocabulary, so
meds_etl emits concepts such as ICD10CM/I50.84 untouched. These are not standard;
MOTOR's dictionary holds standard concepts only. Closing that gap is this module's job.

The work runs in two phases:

1. Resolve -- our code to a standard OMOP concept. One helper per kind of gap:
   resolve_direct (already a token), resolve_sssom (MIMIC itemids, which have
   no OMOP vocabulary at all), resolve_maps_to (valid but non-standard, e.g.
   ICD10CM/NDC) and the manual table (MIMIC inventions such as MIMIC_IV_Gender/M).
   They run most-authoritative first, each anti-joined against what remains, so a
   curated mapping always wins over one we derive.
2. Climb -- that concept to the nearest ancestor MOTOR actually holds. This acts on
   targets, not source codes, which is why the resolvers take no vocabulary argument
   and why a standard-but-unknown code still needs a target: it is where the climb
   starts.

Every resolver shares one output schema so the driver can chain them::

    code   (String)  the code as it appears in the MEDS events
    target (String)  the OMOP concept it resolved to
    method (String)  which layer resolved it, one of the METHOD_* constants

A code the layer cannot resolve emits no row rather than a null target; the
driver's anti-joins depend on absence to pass work down the chain.
"""

import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import msgpack
import polars as pl

METHOD_DIRECT: Literal["direct"] = "direct"
METHOD_SSSOM: Literal["sssom"] = "sssom"
METHOD_MAPS_TO: Literal["maps_to"] = "maps_to"
METHOD_MANUAL: Literal["manual"] = "manual"


def code_inventory(events: pl.LazyFrame) -> pl.LazyFrame:
    """Determines the number of events per code in the dataset.

    The inventory is the denominator for every coverage claim we make, so null codes
    are counted rather than dropped: their events are real, they simply cannot map.

    Args:
        events: MEDS events, of which only ``code`` (String) is read. Stage 1's
            shards carry 21 columns; the rest are ignored

    Returns:
        pl.LazyFrame: sorted by code, nulls last::

            code  (String)  a distinct code appearing in the events
            count (UInt32)  how many events carry it
    """
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
    weights: Mapping[str, float]
    all_parents: Mapping[str, tuple]

    @functools.cached_property
    def tokens(self) -> frozenset[str]:
        """Returns the union of all codes in the vocabulary."""
        return self.code_tokens.union(self.numeric_codes.union(self.text_codes))


def load_motor_vocab(dictionary_path: Path, *, vocab_size: int) -> MotorVocab:
    """Extracts the vocabulary of MOTOR.

    The released dictionary holds a candidate list far longer than the model's
    vocabulary; only its first vocab_size entries are real tokens, their position
    in the list being the embedding row they index. Entries past the cut, and
    entries of the unused type, never reach the model. This represents unused
    model capacity built into MOTOR.

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
    weights: dict[str, float] = {}

    for vocab_entry in vocab:
        code = vocab_entry["code_string"]
        match vocab_entry["type"]:
            case 0:
                code_tokens.add(code)
            case 1:
                numeric_codes.add(code)
            case 2:
                text_codes.add(code)
            case _:
                continue
        weight = vocab_entry["weight"]
        if weight < weights.get(code, math.inf):
            weights[code] = weight

    return MotorVocab(
        frozenset(code_tokens),
        frozenset(numeric_codes),
        frozenset(text_codes),
        weights,
        all_parents,
    )


def resolve_direct(codes: pl.LazyFrame, vocab: MotorVocab) -> pl.LazyFrame:
    """Resolves codes MOTOR already holds, which map to themselves.

    The first and cheapest layer: no external data, and nothing downstream can improve
    on a code that is already a token. Membership is tested against the union of the
    token sets.

    Args:
        codes: any frame with a code (String) column; other columns are ignored
        vocab: the extracted MOTOR vocabulary

    Returns:
        pl.LazyFrame: the shared resolver schema, method = "direct"::

            code   (String)  the code as it appears in the MEDS events
            target (String)  identical to code, since it is already a token
            method (String)  "direct"

        Codes absent from the vocabulary emit no row, passing to the next layer.
    """
    return (
        codes.filter(pl.col("code").is_in(list(vocab.tokens)))
        .with_columns(
            pl.col("code").alias("target"), pl.lit(METHOD_DIRECT).alias("method")
        )
        .select(pl.col("code"), pl.col("target"), pl.col("method"))
    )


def resolve_sssom(codes: pl.LazyFrame, code_metadata: pl.LazyFrame) -> pl.LazyFrame:
    """Resolves MIMIC itemids through the crosswalks meds_etl ships.

    MIMIC itemids (MIMIC_IV_LABITEM/50912, MIMIC_IV_ITEM/220045) belong to no
    OMOP vocabulary, so no Athena lookup can reach them. meds_etl bundles MIT-LCP
    SSSOM crosswalks for exactly these and records the result in parent_codes.
    Those crosswalks are curated, so this layer runs before the general Athena bridge.

    Note the metadata covers only the two itemid families. Diagnoses and drugs are
    absent from it entirely, which is why resolve_maps_to exists.

    Args:
        codes: any frame with a code (String) column; other columns are ignored
        code_metadata: the MEDS sidecar, metadata/codes.parquet::

            code         (String)       the MEDS code
            description  (String)       human label, unused here
            parent_codes (List(String)) standard concepts, null when unmapped

    Returns:
        pl.LazyFrame: the shared resolver schema, method = "sssom"::

            code   (String)  the code as it appears in the MEDS events
            target (String)  a standard OMOP concept from parent_codes
            method (String)  "sssom"

        One row per parent: the column is a list, so a code may emit several rows
        Codes whose parent_codes is null and codes the metadata does not mention
        emit no row. Codes the metadata knows but the events never carried are not
        invented.
    """
    metadata = (
        code_metadata.select(pl.col("code"), pl.col("parent_codes"))
        .explode("parent_codes")
        .rename({"parent_codes": "target"})
        .drop_nulls("target")
        .with_columns(pl.lit(METHOD_SSSOM).alias("method"))
    )

    return codes.join(metadata, on=pl.col("code"), how="inner").select(
        pl.col("code"), pl.col("target"), pl.col("method")
    )


def resolve_maps_to(
    codes: pl.LazyFrame, concept: pl.LazyFrame, relationship: pl.LazyFrame
) -> pl.LazyFrame:
    """Resolves valid but non-standard codes through OMOP's 'Maps to' bridge.

    ICD10CM, ICD9CM and NDC are legitimate OMOP vocabularies, so meds_etl emits
    them untouched, but they are source concepts: OMOP expects analysis to happen on
    the standard concept each maps to (SNOMED for conditions, RxNorm for drugs).
    CONCEPT_RELATIONSHIP carries that bridge as a directed edge,
    ``concept_id_1 --[Maps to]--> concept_id_2``.

    Already-standard codes map to themselves, and those self-edges are kept
    deliberately: ICD10PCS/0T773DZ is standard yet absent from MOTOR's kept
    vocabulary, so it needs a target purely to give the climb a starting point.
    Filtering self-edges out, as femr's own ontology loader does, would silently strip
    the procedure codes.

    Athena keys everything on integer concept_id, so concept is consulted
    twice: once to turn our code string into a source id, once to turn the target id
    back into a string. That costs two scans of a 766 MB file, which is the deliberate
    trade against holding ~9.9M rows in memory.

    Args:
        codes: any frame with a code (String) column; other columns are ignored
        concept: Athena CONCEPT.csv, of which concept_id, vocabulary_id
            and concept_code are read. All columns arrive as String
        relationship: Athena CONCEPT_RELATIONSHIP.csv, of which
            concept_id_1, concept_id_2, relationship_id and
            invalid_reason are read

    Returns:
        pl.LazyFrame: the shared resolver schema, method = "maps_to"::

            code   (String)  the code as it appears in the MEDS events
            target (String)  the standard concept it maps to
            method (String)  "maps_to"

        One row per mapping: roughly 5% of codes map to several standard concepts,
        so fan-out is expected and picking a primary is the assembly step's job. Codes
        Athena does not know, codes carrying no 'Maps to' edge, and edges flagged with
        an invalid_reason all emit no row.
    """
    lookup = concept.select(
        (pl.col("vocabulary_id") + "/" + pl.col("concept_code")).alias("code"),
        pl.col("concept_id").cast(pl.Int64),
    )
    mappings = relationship.filter(
        pl.col("relationship_id") == "Maps to",
        pl.col("invalid_reason").is_null(),
    ).select(
        pl.col("concept_id_1").cast(pl.Int64).alias("concept_id"),
        pl.col("concept_id_2").cast(pl.Int64).alias("target_id"),
    )
    targets = lookup.select(
        pl.col("concept_id").alias("target_id"), pl.col("code").alias("target")
    )

    return (
        codes.join(lookup, on="code", how="inner")
        .join(mappings, on="concept_id", how="inner")
        .join(targets, on="target_id", how="inner")
        .with_columns(pl.lit(METHOD_MAPS_TO).alias("method"))
        .select(pl.col("code"), pl.col("target"), pl.col("method"))
    )


def resolve_manual(codes: pl.LazyFrame, mapping: Mapping[str, str]) -> pl.LazyFrame:
    """Resolves the curated mapping for MIMIC codes no crosswalk can reach.

    MIMIC invents a handful of families outright: ``MIMIC_IV_Gender/M`` is a column
    value, not a code in any vocabulary, so neither the SSSOM crosswalks nor Athena
    can say anything about it. The mapping in ``codes`` is the hand-written bridge for
    those, and because we wrote it, it is the least authoritative layer and runs last.

    The mapping is a parameter rather than an import so the layer stays independent of
    the table it happens to be driven by, in keeping with the other resolvers.

    Args:
        codes: any frame with a ``code`` (String) column; other columns are ignored
        mapping: MEDS code to OMOP concept, e.g. ``MANUAL_CONCEPT_MAP``

    Returns:
        pl.LazyFrame: the shared resolver schema, method = "manual"::

            code   (String)  the code as it appears in the MEDS events
            target (String)  the curated OMOP concept
            method (String)  "manual"

        Codes the mapping does not mention emit no row. Entries the mapping holds for
        codes the events never carried are not invented.
    """
    mapping_frame = pl.LazyFrame(
        {
            "code": pl.Series(list(mapping), dtype=pl.String),
            "target": pl.Series(list(mapping.values()), dtype=pl.String),
        }
    )

    return (
        codes.join(mapping_frame, on="code", how="inner")
        .with_columns(pl.lit(METHOD_MANUAL).alias("method"))
        .select(pl.col("code"), pl.col("target"), pl.col("method"))
    )


def _immediate_parents(
    code: str, all_parents: Mapping[str, tuple[str, ...]]
) -> tuple[str, ...]:
    """Extracts the immediate ontology parents of a code.

    The all_parents property of the vocabulary contains a flat transitive
    list of all ancestors. In order to compute the closest ancestor we
    need to determine the immediate parents. We use a reduction to determine
    the immediate parents of a target, given that an ancestor is an immediate
    parent iff it is not an ancestor to any of the other target ancestors.

    Args:
        code (str): the code whose immediate parents we want to extract
        all_parents (Mapping[str, tuple[str, ...]]): the map containing
            all transitive ancestors.

    Returns:
        tuple[str, ...]: a tuple containing all immediate parents to the
            target code. A sorted tuple is used because it is immutable,
            and because we want to guarantee the order of the parents.
    """
    immediate_parents: list[str] = []
    transitive_ancestors = [x for x in all_parents.get(code, ()) if x != code]
    for anc in transitive_ancestors:
        if not any(
            anc in all_parents[other] for other in transitive_ancestors if other != anc
        ):
            immediate_parents.append(anc)

    return tuple(sorted(immediate_parents))


def climb_to_vocab(targets: pl.LazyFrame, vocab: MotorVocab) -> pl.LazyFrame:
    """Converts valid OMOP codes to valid MOTOR vocab codes.

    MOTOR has a limited vocabulary of vocab_size unique tokens.
    These are all standardized OMOP codes. However, not all OMOP codes
    have a valid embedding. For every code not directly in the vocabulary
    we are called to find the nearest ontology parent so that the event may
    be parsed by the model.

    The weight of each code is stored as the negative Shanon entropy. This means
    that the most negative weight represents the most informative

    Args:
        targets (pl.LazyFrame): the codes emitted by the previous code
            resolution steps
        vocab (MotorVocab): custom container for MOTOR's vocabulary
    """
    # dictionary holding already discovered OMOP->vocab mappings
    # including how many levels away it is
    pass
