"""The assembled MOTOR encoder.

This module owns the order layers run in, the tensors that are shared across all twelve
layers, and the mapping from the released checkpoint's haiku parameter names onto the
submodules.
"""

from collections.abc import Mapping

import torch

from thesis.modelling.motor.constants import MOTOR_RMS_EPS
from thesis.modelling.motor.layers import (
    HaikuRMSNorm,
    HierarchicalEmbedding,
    MotorBlock,
    local_attention_mask,
    rotary_tables,
)


class MotorEncoder(torch.nn.Module):
    """MOTOR's transformer backbone, from token indices to features.

    The rotary tables and the attention mask are built once here and passed down,
    because they are the same for every block. The weights of each block are
    however independent.

    Attributes:
        embedding (HierarchicalEmbedding): The ancestor-sum input embedding.
        in_norm (HaikuRMSNorm): Normalises the embedded sequence.
        blocks (torch.nn.ModuleList): The twelve blocks, in checkpoint order.
        out_norm (HaikuRMSNorm): Normalises the features the encoder returns.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        n_heads: int,
        n_layers: int,
        attention_width: int,
        eps: float = MOTOR_RMS_EPS,
    ) -> None:
        """Builds an encoder at the given widths.

        Args:
            vocab_size (int): How many ontology tokens the embedding table holds.
            hidden_size (int): The model width.
            intermediate_size (int): The feed-forward expansion's width.
            n_heads (int): How many attention heads share the channels.
            n_layers (int): How many blocks the stack holds.
            attention_width (int): How many positions back a query may reach.
            eps (float): The norms' epsilon, inside the square root.

        Raises:
            ValueError: If the heads do not divide the model width evenly.
        """
        super().__init__()
        if hidden_size % n_heads:
            raise ValueError(
                f"{n_heads} heads do not divide a width of {hidden_size} evenly."
            )

        self.head_size = hidden_size // n_heads
        self.attention_width = attention_width
        self.embedding = HierarchicalEmbedding(vocab_size, hidden_size)
        self.in_norm = HaikuRMSNorm(hidden_size, eps=eps)
        self.blocks = torch.nn.ModuleList(
            MotorBlock(hidden_size, intermediate_size, n_heads, eps=eps)
            for _ in range(n_layers)
        )
        self.out_norm = HaikuRMSNorm(hidden_size, eps=eps)

    @property
    def compute_dtype(self) -> torch.dtype:
        """The dtype the stack runs in, which the embedding table does not share."""
        return self.out_norm.weight.dtype

    def half_stack(self) -> "MotorEncoder":
        """Casts everything other than the embedding table to float16.

        An embedding row is a sum of ontology ancestors, so rounding the table first
        rounds every summand.

        Returns:
            MotorEncoder: This encoder, cast in place, for chaining.
        """
        self.in_norm.half()
        self.blocks.half()
        self.out_norm.half()
        return self

    def load_haiku(self, params: Mapping[str, torch.Tensor]) -> None:
        """Copies a whole checkpoint in, keyed by haiku's names.

        Keys are relative to the encoder's haiku scope, so "rms_norm::scale" rather
        than the 52-character absolute path.

        Args:
            params (Mapping[str, torch.Tensor]): The checkpoint parameters.

        Raises:
            KeyError: If any parameter is missing. Checked up front, so a checkpoint
                from a differently shaped model cannot half-load.
            ValueError: If a parameter does not fit the module it belongs to.
        """
        expected = ["embed::embeddings", "rms_norm::scale", "rms_norm_1::scale"]
        expected += [
            f"{self._block_prefix(index)}{leaf}"
            for index in range(len(self.blocks))
            for leaf in (
                "rms_norm::scale",
                "linear::w",
                "linear::b",
                "linear_1::w",
                "linear_1::b",
            )
        ]
        if missing := [key for key in expected if key not in params]:
            raise KeyError(
                f"The checkpoint is missing {len(missing)} parameters: {missing}."
            )

        with torch.no_grad():
            self.in_norm.weight.copy_(params["rms_norm::scale"])
            self.out_norm.weight.copy_(params["rms_norm_1::scale"])
        self.embedding.load_haiku(params["embed::embeddings"])

        for index, block in enumerate(self.blocks):
            prefix = self._block_prefix(index)
            block.load_haiku(
                params[f"{prefix}rms_norm::scale"],
                params[f"{prefix}linear::w"],
                params[f"{prefix}linear::b"],
                params[f"{prefix}linear_1::w"],
                params[f"{prefix}linear_1::b"],
            )

    @staticmethod
    def _block_prefix(index: int) -> str:
        """Names one block's haiku scope, below the encoder's own."""
        return f"loop_{index}/TransformerBlock/~/"

    def forward(
        self,
        indices: torch.Tensor,
        seq_len: int,
        ages: torch.Tensor,
        normed_ages: torch.Tensor,
        valid_tokens: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Runs the whole backbone over one buffer of events, or a batch of them.

        The per-position tensors are either one dimensional, describing a single
        buffer, or two, describing a batch of equal-length sequences.

        Args:
            indices (torch.Tensor): The sparse (read, write) pairs, shaped
                (n_pairs, 2). See HierarchicalEmbedding. When batched, the write
                index is flat over the whole batch, so position p of sequence b is
                b * seq_len + p.
            seq_len (int): How many positions each sequence holds.
            ages (torch.Tensor): Each position's age in days, float32, shaped
                (seq_len,) or (batch, seq_len). Drives the rotary tables, so it is a
                real quantity and not an index.
            normed_ages (torch.Tensor): The z-scored age per position, shaped like
                the ages. Concatenated with its square onto every block's input.
            valid_tokens (torch.Tensor): Which positions hold a real event, shaped
                like the ages.
            segment_ids (torch.Tensor): Which subject each position belongs to,
                shaped like the ages. A batch of single-subject sequences is all
                zeros.

        Returns:
            torch.Tensor: The features, shaped like the ages with (hidden_size,)
                appended.

        Raises:
            ValueError: If the ages are not one or two dimensional, if they disagree
                with seq_len, or if any other per-position tensor disagrees with them.
        """
        if ages.ndim not in (1, 2):
            raise ValueError(
                f"ages must be shaped (seq_len,) or (batch, seq_len), got "
                f"{tuple(ages.shape)}."
            )
        if ages.shape[-1] != seq_len:
            raise ValueError(
                f"ages must hold {seq_len} positions, got {tuple(ages.shape)}."
            )
        for name, tensor in (
            ("normed_ages", normed_ages),
            ("valid_tokens", valid_tokens),
            ("segment_ids", segment_ids),
        ):
            if tensor.shape != ages.shape:
                raise ValueError(
                    f"{name} must be shaped {tuple(ages.shape)} like the ages, got "
                    f"{tuple(tensor.shape)}."
                )

        # the embedding sums into one flat buffer either way, and the batch is only
        # separated out afterwards; the pairs already carry flat write indices
        x = self.embedding(indices, ages.numel()).unflatten(0, ages.shape)

        # invalid positions are filled with ONES, not zeros: the norm below divides
        # by the row's own magnitude, so a zero row would give 0/0.
        x = torch.where(valid_tokens.unsqueeze(-1), x, x.new_ones(()))
        x = self.in_norm(x).to(self.compute_dtype)

        sin, cos = rotary_tables(ages, self.head_size, dtype=x.dtype)
        if ages.ndim == 2:
            # attention runs at (batch, heads, seq, dim) and every head shares one
            # table, so the head axis is inserted here rather than in every block
            sin, cos = sin.unsqueeze(-3), cos.unsqueeze(-3)
        mask = local_attention_mask(segment_ids, self.attention_width)
        normed_ages = normed_ages.to(x.dtype)

        for block in self.blocks:
            x = block(x, normed_ages, sin, cos, mask)

        return self.out_norm(x)
