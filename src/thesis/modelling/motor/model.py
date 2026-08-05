"""The assembled MOTOR encoder.

This module owns the order layers run in, the tensors that are shared across all twelve
layers, and the mapping from the released checkpoint's haiku parameter names onto the
submodules.
"""

from collections.abc import Mapping

import torch

from thesis.modelling.motor.constants import MOTOR_RMS_EPS
from thesis.modelling.motor.layers import (
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
        in_norm (torch.nn.RMSNorm): Normalises the embedded sequence.
        blocks (torch.nn.ModuleList): The twelve blocks, in checkpoint order.
        out_norm (torch.nn.RMSNorm): Normalises the features the encoder returns.
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
        self.in_norm = torch.nn.RMSNorm(hidden_size, eps=eps)
        self.blocks = torch.nn.ModuleList(
            MotorBlock(hidden_size, intermediate_size, n_heads, eps=eps)
            for _ in range(n_layers)
        )
        self.out_norm = torch.nn.RMSNorm(hidden_size, eps=eps)

    @property
    def compute_dtype(self) -> torch.dtype:
        """The dtype the stack runs in, which the embedding table does not share.

        Released inference casts every parameter to float16 except the embedding
        table, so `encoder.half()` followed by `encoder.embedding.float()` reproduces
        it and this property then reports float16.
        """
        return self.out_norm.weight.dtype

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
        """Runs the whole backbone over one packed buffer of events.

        Args:
            indices (torch.Tensor): The sparse (read, write) pairs, shaped
                (n_pairs, 2). See HierarchicalEmbedding.
            seq_len (int): How many positions the buffer holds.
            ages (torch.Tensor): Each position's age in minutes, float32, shaped
                (seq_len,). Drives the rotary tables, so it is a real quantity and
                not an index.
            normed_ages (torch.Tensor): The z-scored age per position, shaped
                (seq_len,). Concatenated with its square onto every block's input.
            valid_tokens (torch.Tensor): Which positions hold a real event, shaped
                (seq_len,).
            segment_ids (torch.Tensor): Which packed subject each position belongs
                to, shaped (seq_len,). A single subject is all zeros.

        Returns:
            torch.Tensor: The features, shaped (seq_len, hidden_size).

        Raises:
            ValueError: If any per-position tensor disagrees with seq_len.
        """
        for name, tensor in (
            ("ages", ages),
            ("normed_ages", normed_ages),
            ("valid_tokens", valid_tokens),
            ("segment_ids", segment_ids),
        ):
            if tensor.shape != (seq_len,):
                raise ValueError(
                    f"{name} must be shaped ({seq_len},), got {tuple(tensor.shape)}."
                )

        x = self.embedding(indices, seq_len)

        # invalid positions are filled with ONES, not zeros: the norm below divides
        # by the row's own magnitude, so a zero row would give 0/0.
        x = torch.where(valid_tokens.unsqueeze(-1), x, x.new_ones(()))
        x = self.in_norm(x).to(self.compute_dtype)

        sin, cos = rotary_tables(ages, self.head_size, dtype=x.dtype)
        mask = local_attention_mask(segment_ids, self.attention_width)
        normed_ages = normed_ages.to(x.dtype)

        for block in self.blocks:
            x = block(x, normed_ages, sin, cos, mask)

        return self.out_norm(x)
