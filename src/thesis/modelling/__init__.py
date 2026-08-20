"""Modelling stack: MEDS ETL, labelling, the MOTOR port, training and evaluation.

Several stages resolve a moment against an ordered frame with an asof join, which
doesn't verify its inputs are ordered once `by` groups are given and says so once
per shard. Each of them sorts its own inputs. The warning is raised from Polars'
Rust core when the plan executes, not when it is built, so it surfaces in the
caller's collect and cannot be scoped to a context manager around any one of them
-- hence one filter for the whole stack.
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message="Sortedness of columns cannot be checked when 'by' groups provided",
    category=UserWarning,
)
