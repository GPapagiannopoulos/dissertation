"""Module implementing a baseline XGBoost tree to which models are compared."""

import warnings

# `last_observation` resolves a landmark against an ordered event stream with an asof
# join, which cannot verify its inputs are ordered once `by` groups are given and says
# so once per shard. It sorts its own inputs. The warning is raised from Polars' Rust
# core when the plan executes, not when it is built, so it surfaces in the caller's
# collect and cannot be scoped to a context manager around the function -- hence one
# filter for the package, as in `motor/__init__.py`.
warnings.filterwarnings(
    "ignore",
    message="Sortedness of columns cannot be checked when 'by' groups provided",
    category=UserWarning,
)
