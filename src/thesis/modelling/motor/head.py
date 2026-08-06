"""The binary prediction head over MOTOR's features."""

import math

import torch

from thesis.modelling.motor.model import MotorEncoder


class MotorClassifier(torch.nn.Module):
    """A MOTOR encoder with a linear head read at the labelled positions.

    The head runs in float32 whatever the stack runs in, so the logits reaching the
    loss are not the ones that lose precision.

    Attributes:
        encoder (MotorEncoder): The backbone, whose features the head reads.
        head (torch.nn.Linear): Hidden size down to one logit.
    """

    def __init__(
        self, encoder: MotorEncoder, positive_rate: float | None = None
    ) -> None:
        """Wraps an encoder, optionally starting the head at a base rate.

        Args:
            encoder (MotorEncoder): A built backbone. Its width sets the head's.
            positive_rate (float | None): The training set's prevalence. Given, the
                weight starts at zero and the bias at its logit, so the untrained
                model predicts the base rate rather than a saturated random one.

        Raises:
            ValueError: If the positive rate is not strictly between zero and one.
        """
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(encoder.out_norm.weight.numel(), 1)

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

    def forward(
        self,
        indices: torch.Tensor,
        seq_len: int,
        ages: torch.Tensor,
        normed_ages: torch.Tensor,
        valid_tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        label_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Scores one batch at its labelled positions.

        Every argument but the last is `MotorEncoder.forward`'s, so a batch from
        `collate` splats in once its `labels` have been taken out for the loss.

        Args:
            indices (torch.Tensor): The (read, write) embedding pairs.
            seq_len (int): How many positions each sequence holds.
            ages (torch.Tensor): Each position's age in days.
            normed_ages (torch.Tensor): The z-scored ages.
            valid_tokens (torch.Tensor): Which positions hold a real event.
            segment_ids (torch.Tensor): Which subject each position belongs to.
            label_indices (torch.Tensor): Flat `row * seq_len + position` offsets of
                the positions to score, as `collate` emits them.

        Returns:
            torch.Tensor: One float32 logit per label, shaped (n_labels,).

        Raises:
            ValueError: If the batch carries no label, or if one indexes past the
                batch. Checked here because an out-of-range gather is a device-side
                assert on CUDA, which kills the context rather than the step.
        """
        if label_indices.numel() == 0:
            raise ValueError(
                "A batch carries at least one label; the mean of an empty loss is NaN."
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

        return self.head(picked.to(self.head.weight.dtype)).squeeze(-1)
