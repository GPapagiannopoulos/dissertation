"""Turns the long feature table into the matrix XGBoost trains on.

The long table holds one row per `(landmark, code)`; XGBoost wants one row per
landmark, with every code a column. That pivot is the whole module.

There are ~8,900 codes carrying three features each, so a dense frame would
be ~26,800 columns over 2,965,363 rows. Instead each shard becomes a `scipy`
CSR block and is handed straight to XGBoost through a `DataIter`, which bins it to
`max_bin` buckets on arrival.

Binning alone was not enough here. The training fold's ~0.9 billion nonzeros make an
in-memory `QuantileDMatrix` peak at 8.2 GiB on a 14 GB host -- measured twice, killed
before boosting round zero both times -- so `build_dmatrix` takes a `cache` folder and
builds an `ExtMemQuantileDMatrix`, whose pages live on disk. The cap that stops such a
run from taking the desktop with it belongs outside this module, on the process.

A code a landmark has never seen contributes no entry at all, and XGBoost learns
a default branch direction for such rows rather than imputing.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from scipy.sparse import coo_matrix, csr_matrix

FEATURE_COLUMNS: tuple[str, ...] = ("last_value", "hours_since", "n_obs")
"""The per-code features, in the order they are laid out within a code's block.

A code occupying slot `s` owns columns `s * 3 + 0 .. s * 3 + 2`, so its three
features stay adjacent. That is purely for reading `feature_importances_`; the
trees are indifferent to column order.
"""


def build_code_index(dest: Path, *, fold: str = "training") -> pl.DataFrame:
    """Assigns every code a column slot, frozen on one fold.

    Codes are sorted before numbering so the index is reproducible across runs, and
    shards are read one at a time rather than through a single glob.

    Args:
        dest (Path): The folder `run_build_features` wrote.
        fold (str): The fold whose codes define the columns. Defaults to "training".

    Returns:
        pl.DataFrame: Columns `code` and `code_slot`, sorted by code.

    Raises:
        FileNotFoundError: If dest holds no feature shards.
        ValueError: If no code survives, which means the fold matched no landmark.
    """
    pairs = shard_pairs(dest)
    seen: list[pl.DataFrame] = []
    for features, spine in pairs:
        wanted = (
            pl.scan_parquet(spine).filter(pl.col("fold") == fold).select("landmark_id")
        )
        seen.append(
            pl.scan_parquet(features)
            .join(wanted, on="landmark_id", how="semi")
            .select("code")
            .unique()
            .collect()
        )

    codes = pl.concat(seen).unique().sort("code")
    if not codes.height:
        raise ValueError(
            f"No codes survived for fold {fold!r}. Check the fold name against the "
            f"spine's `fold` column."
        )
    return codes.with_row_index("code_slot").select("code", "code_slot")


def feature_names(index: pl.DataFrame) -> list[str]:
    """Names every column, so feature importance is readable.

    Args:
        index (pl.DataFrame): The frame `build_code_index` returned.

    Returns:
        list[str]: `<code>::<feature>` for each slot, in column order.
    """
    codes = index.sort("code_slot")["code"].to_list()
    return [f"{code}::{name}" for code in codes for name in FEATURE_COLUMNS]


def n_columns(index: pl.DataFrame) -> int:
    """The matrix's width for a given code index.

    Args:
        index (pl.DataFrame): The frame `build_code_index` returned.

    Returns:
        int: The number of columns, i.e. codes times features.
    """
    return index.height * len(FEATURE_COLUMNS)


def long_to_csr(long: pl.DataFrame, *, n_rows: int, width: int) -> csr_matrix:
    """Pivots a shard's long table into one sparse block.

    Each `(row, code_slot)` pair emits up to one entry per feature, and emits none
    for a feature that is null there. That is what keeps a value-less code -- a
    diagnosis, a procedure -- contributing recency and a count while its
    `last_value` cell stays genuinely absent.

    Args:
        long (pl.DataFrame): Columns `row`, `code_slot`, and every entry of
            `FEATURE_COLUMNS`.
        n_rows (int): The block's height, i.e. the fold's landmarks in this shard.
        width (int): The block's width, from `n_columns`.

    Returns:
        csr_matrix: The block, float32.

    Raises:
        ValueError: If a `(row, code_slot)` pair repeats. `coo_matrix` sums
            duplicates, so a repeat would silently double a feature instead of
            raising, and the resulting model would look fine.
    """
    if long.select("row", "code_slot").is_duplicated().any():
        raise ValueError(
            "A (row, code_slot) pair repeats in the long table. Building the matrix "
            "would sum the duplicates rather than fail. Check that the asof join "
            "emitted one row per (landmark, code)."
        )

    rows = long["row"].to_numpy()
    slots = long["code_slot"].to_numpy()

    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []

    for offset, name in enumerate(FEATURE_COLUMNS):
        column = long[name]
        present = column.is_not_null().to_numpy()
        row_parts.append(rows[present])
        column_parts.append(slots[present] * len(FEATURE_COLUMNS) + offset)
        # drop_nulls preserves order, so this aligns with `present` while sidestepping
        # the dtype promotion to_numpy would apply to a null-bearing integer column.
        value_parts.append(column.drop_nulls().to_numpy().astype(np.float32))

    return coo_matrix(
        (
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(n_rows, width),
        dtype=np.float32,
    ).tocsr()


def shard_matrix(
    features: Path,
    spine: Path,
    index: pl.DataFrame,
    *,
    fold: str,
) -> tuple[csr_matrix, np.ndarray] | None:
    """Builds one shard's block and its labels, in matching row order.

    The roster comes from the **spine**, not the long table, so a landmark carrying
    no visible code survives as an all-missing row instead of vanishing. Rows are
    numbered off that roster and the long table is joined onto it, which is what
    guarantees `y[i]` belongs to `X[i]` -- sorting the two independently and trusting
    the orders to agree is the bug this avoids.

    Args:
        features (Path): One shard's long feature parquet.
        spine (Path): The matching shard's spine parquet.
        index (pl.DataFrame): The frame `build_code_index` returned.
        fold (str): Which fold to keep.

    Returns:
        tuple[csr_matrix, np.ndarray] | None: The block and its uint8 labels, or
            None when the fold holds no landmark in this shard.
    """
    roster = (
        pl.scan_parquet(spine)
        .filter(pl.col("fold") == fold)
        .select("landmark_id", "boolean_value")
        .sort("landmark_id")
        .with_row_index("row")
        .collect()
    )
    if not roster.height:
        return None

    long = (
        pl.scan_parquet(features)
        .join(roster.lazy().select("landmark_id", "row"), on="landmark_id")
        .join(index.lazy(), on="code")
        .select("row", "code_slot", *FEATURE_COLUMNS)
        .collect()
    )

    block = long_to_csr(long, n_rows=roster.height, width=n_columns(index))
    labels = roster["boolean_value"].cast(pl.UInt8).to_numpy()
    return block, labels


class ShardIterator(xgb.DataIter):
    """Feeds the matrix builder one shard at a time.

    This is what keeps the full CSR off the heap: each block is quantised to
    `max_bin` buckets as it arrives and the raw float32 entries are released.

    Quantising a block does not, on its own, make the matrix fit. The binned index
    is smaller than the CSR but still proportional to the ~0.9 billion nonzeros the
    training fold carries, and a `QuantileDMatrix` holds all of it in memory --
    measured at an 8.2 GiB peak on a 14 GB host, killed before boosting round zero.
    Passing `cache` spills those pages to disk instead, which is the only
    configuration of this pipeline that fits here.
    """

    def __init__(
        self,
        pairs: Iterable[tuple[Path, Path]],
        index: pl.DataFrame,
        *,
        fold: str,
        cache: Path | None = None,
    ) -> None:
        """Records what to iterate over.

        Args:
            pairs (Iterable[tuple[Path, Path]]): `(features, spine)` shard paths.
            index (pl.DataFrame): The frame `build_code_index` returned.
            fold (str): Which fold to keep.
            cache (Path | None): Prefix for the on-disk page cache, i.e. a path
                xgboost appends its own suffixes to rather than a folder. None keeps
                every page in memory, which is what a `QuantileDMatrix` expects.
        """
        self._pairs = list(pairs)
        self._index = index
        self._fold = fold
        self._position = 0
        # `release_data` stays at its default: the transformation here is the pivot
        # to float32 CSR, which is memory intensive rather than compute intensive,
        # so a block is dropped as soon as xgboost has binned it.
        super().__init__(cache_prefix=None if cache is None else str(cache))

    def next(self, input_data) -> bool:  # noqa: D102 - xgboost's own signature
        while self._position < len(self._pairs):
            features, spine = self._pairs[self._position]
            self._position += 1
            batch = shard_matrix(features, spine, self._index, fold=self._fold)
            if batch is None:
                continue
            block, labels = batch
            input_data(data=block, label=labels)
            return True
        return False

    def reset(self) -> None:  # noqa: D102 - xgboost's own signature
        self._position = 0


def build_dmatrix(
    dest: Path,
    index: pl.DataFrame,
    *,
    fold: str,
    ref: xgb.DMatrix | None = None,
    max_bin: int = 256,
    names: Sequence[str] | None = None,
    cache: Path | None = None,
) -> xgb.DMatrix:
    """Assembles one fold's matrix from the shards on disk.

    `ref` is not optional in practice. A quantile matrix built without it computes
    its own bin boundaries, so the same creatinine would land in different buckets in
    training and in validation and every score would be quietly wrong -- plausible
    numbers, no exception. Pass the training matrix as `ref` for validation and test.

    `cache` chooses between the two matrix classes, and on this host it is what makes
    the difference between a run and an OOM kill. Without it the binned pages stay
    resident; with it they go to disk and xgboost streams them each boosting round.
    The prefix is derived as `cache / fold` rather than taken directly, because two
    matrices sharing one prefix would overwrite each other's pages -- and a run
    builds exactly one matrix per fold, so deriving it makes that collision
    impossible rather than merely unlikely.

    Args:
        dest (Path): The folder `run_build_features` wrote.
        index (pl.DataFrame): The frame `build_code_index` returned.
        fold (str): Which fold to build.
        ref (xgb.DMatrix | None): The training matrix, whose bin boundaries this one
            must reuse. None only when building the training matrix itself.
        max_bin (int): Buckets per feature. Must match across folds. It also sets the
            per-node histogram, which is `n_columns * max_bin * 16` bytes and is the
            second thing that has to fit in memory once the pages no longer do.
        names (Sequence[str] | None): Column names, from `feature_names`.
        cache (Path | None): A folder for the on-disk page cache. None builds an
            in-memory `QuantileDMatrix`, which only suits data small enough to hold.

    Returns:
        xgb.DMatrix: The fold's matrix, quantile-binned either way.

    Raises:
        FileNotFoundError: If dest holds no feature shards.
        ValueError: If the feature and spine folders disagree on shard names.
    """
    prefix = None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        prefix = cache / fold

    iterator = ShardIterator(shard_pairs(dest), index, fold=fold, cache=prefix)
    matrix: xgb.DMatrix = (
        xgb.QuantileDMatrix(iterator, max_bin=max_bin, ref=ref)
        if prefix is None
        else xgb.ExtMemQuantileDMatrix(iterator, max_bin=max_bin, ref=ref)
    )
    # Names cannot be passed to the constructor alongside an iterator -- xgboost
    # rejects any per-matrix argument there, since it expects them per batch.
    if names is not None:
        matrix.feature_names = list(names)
    return matrix


def shard_pairs(dest: Path) -> list[tuple[Path, Path]]:
    """Pairs each feature shard with its spine shard.

    Args:
        dest (Path): The folder `run_build_features` wrote.

    Returns:
        list[tuple[Path, Path]]: `(features, spine)` paths, sorted by name.

    Raises:
        FileNotFoundError: If dest holds no feature shards.
        ValueError: If the two folders disagree on shard names.
    """
    features = sorted((dest / "features").glob("*.parquet"))
    spines = sorted((dest / "spine").glob("*.parquet"))
    if not features:
        raise FileNotFoundError(
            f"Found no parquet shards in {dest / 'features'}. Point dest at the "
            f"folder run_build_features wrote."
        )
    if [path.name for path in features] != [path.name for path in spines]:
        raise ValueError(
            f"{dest / 'features'} and {dest / 'spine'} hold different shards. The "
            f"feature build did not finish, or the folder has been edited."
        )
    return list(zip(features, spines, strict=True))
