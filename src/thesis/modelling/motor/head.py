"""The binary prediction head over MOTOR's features.

The head reads three things at each labelled position, concatenated:

1. **The hidden state at that position** -- what the encoder was always giving it.
2. **A masked mean over the previous `pool_window` positions.** The encoder emits one
   vector per event, and reading only the last one asks a single summary -- written
   largely to describe *that* event, often something routine -- to carry the whole
   prediction. Pooling widens the read without adding a mechanism.
3. **`CLOCK_FEATURES`**, the elapsed time since that event. The encoder has no
   representation of it: the landmark sits a median 4.60 hours after the event whose
   state is read, and a prediction one minute later is otherwise indistinguishable
   from one 94 hours later.

None of this touches the backbone. The pooled vectors are the encoder's own outputs
and the clock is a scalar arriving beside them, so the pretrained weights see exactly
the input distribution they saw before -- which is the whole reason this is a head
change rather than a synthetic token in the sequence.

**The pooling window reaches backwards only.** Positions after the label are events
that happened after the moment being predicted from; averaging them in is leakage,
and it would read as a strong result rather than as an error.
"""

import math

import torch

from thesis.modelling.motor.batching import CLOCK_FEATURES
from thesis.modelling.motor.model import MotorEncoder

DEFAULT_POOL_WINDOW: int = 32
"""How many positions back the mean reaches, the label's own position included.

Fixed rather than tuned. The validation fold has been inspected many times already
and every hyperparameter chosen on it makes the final number more optimistic; the
head can down-weight the pooled half to nothing if the window is wrong, so the cost
of a poor choice is bounded. Sweeping it is also not cheap -- the probe caches one
vector per label, so each candidate window needs its own extraction pass.
"""


class MotorClassifier(torch.nn.Module):
    """A MOTOR encoder with a linear head read at the labelled positions.

    The head runs in float32 whatever the stack runs in, so the logits reaching the
    loss are not the ones that lose precision.

    Attributes:
        encoder (MotorEncoder): The backbone, whose features the head reads.
        pool_window (int): How many positions the masked mean covers.
        n_clock_features (int): How many per-label scalars the head expects.
        head (torch.nn.Linear): Concatenated features down to one logit.
    """

    def __init__(
        self,
        encoder: MotorEncoder,
        positive_rate: float | None = None,
        *,
        pool_window: int = DEFAULT_POOL_WINDOW,
        n_clock_features: int = len(CLOCK_FEATURES),
    ) -> None:
        """Wraps an encoder, optionally starting the head at a base rate.

        Args:
            encoder (MotorEncoder): A built backbone. Its width sets the head's.
            positive_rate (float | None): The training set's prevalence. Given, the
                weight starts at zero and the bias at its logit, so the untrained
                model predicts the base rate rather than a saturated random one.
            pool_window (int): Positions covered by the masked mean, the label's own
                included. **Zero disables pooling**, which with `n_clock_features` at
                zero reproduces the original 768-wide head exactly -- the only way a
                checkpoint saved before this change still loads.
            n_clock_features (int): Width of the `label_clocks` tensor `collate`
                emits. Zero drops the clock.

        Raises:
            ValueError: If the positive rate is not strictly between zero and one,
                or if either width is negative.
        """
        super().__init__()
        if pool_window < 0:
            raise ValueError(
                f"A pooling window covers zero or more positions, got {pool_window}."
            )
        if n_clock_features < 0:
            raise ValueError(f"A clock cannot have {n_clock_features} features.")

        self.encoder = encoder
        self.pool_window = pool_window
        self.n_clock_features = n_clock_features

        hidden = encoder.out_norm.weight.numel()
        width = hidden * (2 if pool_window else 1) + n_clock_features
        self.head = torch.nn.Linear(width, 1)

        if positive_rate is not None:
            if not 0.0 < positive_rate < 1.0:
                raise ValueError(
                    f"A positive rate is a probability strictly inside (0, 1), got "
                    f"{positive_rate}."
                )
            torch.nn.init.zeros_(self.head.weight)
            torch.nn.init.constant_(
                self.head.bias, math.log(positive_rate / (1.0 - positive_rate))
            )

    def _pool(
        self,
        flat: torch.Tensor,
        valid: torch.Tensor,
        label_indices: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """The masked mean over each label's preceding window.

        Two things can make a slot in the window unreal, and both must be excluded
        rather than averaged in as zero. A slot can run off the **start** of the
        sequence, when the label sits fewer than `pool_window` positions in; and a
        slot can be **padding**, since `collate` pads to the next power of two. The
        encoder's output at a padding position is not zero -- it is whatever twelve
        pretrained layers make of filler -- so including it adds a constant piece of
        nonsense rather than diluting toward nothing.

        Dividing by the count of real slots rather than by `pool_window` matters more
        than it looks: a fixed divisor would shrink the pooled vector in proportion
        to how few events a patient has, and sparse records belong to less-monitored,
        mostly negative patients. That is not noise, it is a distortion correlated
        with the label.

        Args:
            flat (torch.Tensor): Encoder features as (batch * seq_len, hidden).
            valid (torch.Tensor): `valid_tokens`, flattened to (batch * seq_len,).
            label_indices (torch.Tensor): Flat offsets of the labelled positions.
            seq_len (int): Positions per sequence, for decomposing those offsets.

        Returns:
            torch.Tensor: One pooled vector per label, (n_labels, hidden).
        """
        rows = torch.div(label_indices, seq_len, rounding_mode="floor")
        positions = label_indices - rows * seq_len

        # offsets 0..window-1 reach BACKWARDS from the label; a forward window would
        # average in events that happen after the prediction time
        offsets = torch.arange(self.pool_window, device=label_indices.device)
        wanted = positions.unsqueeze(1) - offsets.unsqueeze(0)

        # clamping keeps the gather in bounds, so out-of-range slots read position 0
        # rather than raising -- `in_range` is what stops them being counted
        in_range = wanted >= 0
        gather = rows.unsqueeze(1) * seq_len + wanted.clamp(min=0)

        window = flat.index_select(0, gather.reshape(-1)).view(
            gather.shape[0], gather.shape[1], -1
        )
        mask = in_range & valid.index_select(0, gather.reshape(-1)).view(gather.shape)

        weights = mask.unsqueeze(-1).to(window.dtype)
        # the label's own position is always real, so the count is never zero; the
        # clamp is belt and braces rather than a live branch
        return (window * weights).sum(1) / weights.sum(1).clamp(min=1.0)

    def forward(
        self,
        indices: torch.Tensor,
        seq_len: int,
        ages: torch.Tensor,
        normed_ages: torch.Tensor,
        valid_tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        label_indices: torch.Tensor,
        label_clocks: torch.Tensor,
    ) -> torch.Tensor:
        """Scores one batch at its labelled positions.

        Every argument but the last two is `MotorEncoder.forward`'s, so a batch from
        `collate` splats in once its `LABEL_METADATA` has been taken out.

        Args:
            indices (torch.Tensor): The (read, write) embedding pairs.
            seq_len (int): How many positions each sequence holds.
            ages (torch.Tensor): Each position's age in days.
            normed_ages (torch.Tensor): The z-scored ages.
            valid_tokens (torch.Tensor): Which positions hold a real event.
            segment_ids (torch.Tensor): Which subject each position belongs to.
            label_indices (torch.Tensor): Flat `row * seq_len + position` offsets of
                the positions to score, as `collate` emits them.
            label_clocks (torch.Tensor): Per-label scalars, (n_labels, n_clock),
                matching `CLOCK_FEATURES`.

        Returns:
            torch.Tensor: One float32 logit per label, shaped (n_labels,).

        Raises:
            ValueError: If the batch carries no label, if one indexes past the batch,
                or if the clock tensor does not match the head. Checked here because
                an out-of-range gather is a device-side assert on CUDA, which kills
                the context rather than the step.
        """
        if label_indices.numel() == 0:
            raise ValueError(
                "A batch carries at least one label; the mean of an empty loss is NaN."
            )
        if label_clocks.shape != (label_indices.shape[0], self.n_clock_features):
            raise ValueError(
                f"Expected clocks shaped "
                f"{(label_indices.shape[0], self.n_clock_features)}, got "
                f"{tuple(label_clocks.shape)}."
            )

        features = self.encoder(
            indices, seq_len, ages, normed_ages, valid_tokens, segment_ids
        )
        flat = features.flatten(0, -2)

        largest = int(label_indices.max())
        if largest >= flat.shape[0]:
            raise ValueError(
                f"A label at flat offset {largest} indexes past the batch's "
                f"{flat.shape[0]} positions."
            )

        picked = flat.index_select(0, label_indices)

        # autocast would cast the head back down and hand the loss a low-precision
        # logit, which is exactly what running the head separately is meant to avoid
        with torch.autocast(picked.device.type, enabled=False):
            dtype = self.head.weight.dtype
            parts = [picked.to(dtype)]
            if self.pool_window:
                parts.append(
                    self._pool(
                        flat, valid_tokens.reshape(-1), label_indices, seq_len
                    ).to(dtype)
                )
            if self.n_clock_features:
                parts.append(label_clocks.to(dtype))
            return self.head(torch.cat(parts, dim=-1)).squeeze(-1)
