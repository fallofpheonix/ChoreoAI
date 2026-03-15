"""
denoising_transformer.py — Conditional denoising transformer for motion diffusion.

Architecture (MDM-inspired):
  - Flatten pose: ``(B, T, K, 3)`` → ``(B, T, K*3)``
  - Project to ``d_model``
  - Prepend conditioning tokens: latent ``z`` + sinusoidal timestep embedding
  - Transformer encoder
  - Project back to ``(B, T, K*3)``
  - Reshape to ``(B, T, K, 3)``

The network predicts the **noise** added at timestep ``t`` (ε-prediction).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timestep ``t``.

    Args:
        d_model: Embedding dimension (must be even).
        max_timesteps: Maximum number of diffusion timesteps.
    """

    def __init__(self, d_model: int = 256, max_timesteps: int = 1000) -> None:
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sinusoidal embedding"

        half = d_model // 2
        frequencies = torch.exp(
            -math.log(max_timesteps) * torch.arange(half) / (half - 1)
        )
        self.register_buffer("frequencies", frequencies)

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Embed a batch of timesteps.

        Args:
            t: ``(B,)`` integer timestep tensor.

        Returns:
            ``(B, d_model)`` embedding tensor.
        """
        t_float = t.float().unsqueeze(1)                     # (B, 1)
        args = t_float * self.frequencies.unsqueeze(0)       # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, d_model)
        return self.proj(emb)


# ---------------------------------------------------------------------------
# Denoising Transformer
# ---------------------------------------------------------------------------


class DenoisingTransformer(nn.Module):
    """Conditional denoising transformer for skeleton motion diffusion.

    Predicts the noise ``ε`` added to ``x_t`` at timestep ``t``,
    conditioned on an external latent vector ``z`` (from a modality encoder).

    Args:
        num_joints: Number of skeleton joints ``K``.
        joint_dims: Per-joint feature dimensions (3 for XYZ).
        d_model: Transformer model dimension.
        nhead: Number of attention heads.
        num_layers: Transformer depth.
        dim_feedforward: FFN hidden size.
        dropout: Dropout rate.
        latent_dim: Dimensionality of conditioning vector ``z``.
        max_seq_len: Maximum temporal sequence length.
        max_timesteps: Maximum diffusion timesteps.

    Example::

        model = DenoisingTransformer(num_joints=17, latent_dim=256)
        noisy_pose = torch.randn(4, 120, 17, 3)   # (B, T, K, 3)
        t = torch.randint(0, 1000, (4,))           # (B,)
        z = torch.randn(4, 256)                    # (B, latent_dim)
        noise_pred = model(noisy_pose, t, z)        # (4, 120, 17, 3)
    """

    def __init__(
        self,
        num_joints: int = 17,
        joint_dims: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        latent_dim: int = 256,
        max_seq_len: int = 512,
        max_timesteps: int = 1000,
    ) -> None:
        super().__init__()

        self.num_joints = num_joints
        self.joint_dims = joint_dims
        pose_dim = num_joints * joint_dims

        # -- Input projection --
        self.input_proj = nn.Linear(pose_dim, d_model)

        # -- Sinusoidal positional encoding (sequence) --
        position = torch.arange(max_seq_len + 2).unsqueeze(1)  # +2 for cond tokens
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_seq_len + 2, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        # -- Timestep embedding --
        self.timestep_embed = SinusoidalTimestepEmbedding(
            d_model=d_model, max_timesteps=max_timesteps
        )

        # -- Conditioning projection (z → d_model) --
        self.cond_proj = nn.Linear(latent_dim, d_model)

        # -- Transformer encoder --
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # -- Output projection (back to pose dim) --
        self.out_proj = nn.Linear(d_model, pose_dim)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(
        self,
        noisy_pose: Tensor,
        timestep: Tensor,
        conditioning: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict the noise at timestep ``t``.

        Args:
            noisy_pose: ``(B, T, K, 3)`` noisy skeleton tensor.
            timestep: ``(B,)`` integer diffusion timesteps.
            conditioning: ``(B, latent_dim)`` conditioning embedding ``z``.
            key_padding_mask: Optional ``(B, T)`` mask (``True`` = padded).

        Returns:
            Predicted noise ``(B, T, K, 3)``.
        """
        B, T, K, C = noisy_pose.shape
        assert K == self.num_joints and C == self.joint_dims

        pose_dim = K * C

        # Flatten joints: (B, T, K*C)
        x = noisy_pose.reshape(B, T, pose_dim)

        # Project to d_model
        x = self.input_proj(x)          # (B, T, d_model)
        x = x + self.pe[2 : T + 2]      # positional encoding (leave 0,1 for cond tokens)
        x = self.dropout(x)

        # Build conditioning tokens: timestep + z → each becomes (B, 1, d_model)
        t_emb = self.timestep_embed(timestep).unsqueeze(1)    # (B, 1, d_model)
        z_emb = self.cond_proj(conditioning).unsqueeze(1)     # (B, 1, d_model)

        # Add positional slots 0 and 1 to conditioning tokens
        t_emb = t_emb + self.pe[0:1]
        z_emb = z_emb + self.pe[1:2]

        # Prepend conditioning tokens: [t_token, z_token, frame_0, ..., frame_T-1]
        x = torch.cat([t_emb, z_emb, x], dim=1)  # (B, T+2, d_model)

        # Extend key_padding_mask to include conditioning tokens (never masked)
        if key_padding_mask is not None:
            cond_mask = torch.zeros(B, 2, dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cond_mask, key_padding_mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # (B, T+2, d_model)

        # Discard conditioning tokens
        x = x[:, 2:, :]  # (B, T, d_model)

        # Project back to pose space
        x = self.out_proj(x)            # (B, T, K*C)

        return x.reshape(B, T, K, C)   # (B, T, K, 3)
