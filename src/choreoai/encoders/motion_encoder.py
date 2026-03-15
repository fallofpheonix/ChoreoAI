"""
motion_encoder.py — Temporal Transformer encoder for skeleton sequences.

Encodes ``(B, T, K, 3)`` pose tensors into a fixed-size latent vector
``(B, latent_dim)`` via:

  1. Flatten joints → ``(B, T, K*3)``
  2. Linear embedding → ``(B, T, d_model)``
  3. Sinusoidal positional encoding
  4. Transformer encoder layers
  5. Mean pooling over the temporal axis
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.

    Args:
        d_model: Embedding dimensionality.
        max_len: Maximum sequence length.
        dropout: Dropout rate applied after adding the encoding.
    """

    def __init__(
        self, d_model: int, max_len: int = 512, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)          # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer so it moves with .to(device)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to *x*.

        Args:
            x: ``(B, T, d_model)`` input tensor.

        Returns:
            ``(B, T, d_model)`` tensor with positional encoding added.
        """
        x = x + self.pe[: x.size(1)]  # broadcast over batch
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Motion Encoder
# ---------------------------------------------------------------------------


class MotionEncoder(nn.Module):
    """Temporal transformer encoder for 3-D skeleton sequences.

    Args:
        num_joints: Number of skeleton joints ``K``.
        joint_dims: Dimensions per joint (3 for XYZ, 6 for XYZ + velocity).
        d_model: Internal transformer dimensionality.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: FFN hidden size inside each transformer layer.
        dropout: Dropout rate.
        latent_dim: Output embedding dimensionality.
        max_seq_len: Maximum temporal length for positional encoding.

    Example::

        encoder = MotionEncoder(num_joints=17, latent_dim=256)
        poses = torch.randn(8, 120, 17, 3)   # (B, T, K, 3)
        z = encoder(poses)                    # (8, 256)
    """

    def __init__(
        self,
        num_joints: int = 17,
        joint_dims: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        latent_dim: int = 256,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()

        input_dim = num_joints * joint_dims  # K * 3 (or K * 6)

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model=d_model, max_len=max_seq_len, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.out_proj = nn.Linear(d_model, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise projection weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        poses: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Encode a batch of skeleton sequences.

        Args:
            poses: ``(B, T, K, 3)`` pose tensor.
            key_padding_mask: Optional ``(B, T)`` bool tensor.  ``True``
                indicates a *padded* (invalid) position.

        Returns:
            Latent embedding ``(B, latent_dim)``.
        """
        B, T, K, C = poses.shape

        # Flatten joint dimensions: (B, T, K*C)
        x = poses.reshape(B, T, K * C)

        # Project to d_model
        x = self.input_proj(x)               # (B, T, d_model)
        x = self.pos_enc(x)                  # (B, T, d_model)

        # Transformer encoding
        x = self.transformer(
            x, src_key_padding_mask=key_padding_mask
        )  # (B, T, d_model)

        # Masked mean pooling over valid frames
        if key_padding_mask is not None:
            # key_padding_mask: True = padded → 0 weight in mean
            valid = (~key_padding_mask).float().unsqueeze(-1)  # (B, T, 1)
            x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)  # (B, d_model)

        x = self.out_proj(x)     # (B, latent_dim)
        x = self.layer_norm(x)

        return x
