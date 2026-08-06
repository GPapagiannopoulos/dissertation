"""Module defining the port of MOTOR from JAX to PyTorch.

The port is necessary because JAX does not define a library
for implementing (Q)LoRA, or any PEFT methodology for that
matter.
"""

import warnings

# Both the tokeniser and the sequence builder resolve a moment against an ordered
# frame with an asof join, which doesn't verify its inputs are ordered once `by`
# groups are given and says so once per shard. Both sort their own inputs. The
# warning is raised from Polars' Rust core when the plan executes, not when it is
# built, so it surfaces in the caller's collect and cannot be scoped to a context
# manager around either function -- hence one filter for the package.
warnings.filterwarnings(
    "ignore",
    message="Sortedness of columns cannot be checked when 'by' groups provided",
    category=UserWarning,
)
