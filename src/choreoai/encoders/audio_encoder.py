"""
audio_encoder.py — Audio spectrogram encoder with projection head.

Accepts mel-spectrogram tensors ``(B, F, T_a)`` and encodes them into
the shared latent space ``(B, latent_dim)`` using a lightweight
convolutional + transformer architecture.  No external audio processing
library is required at inference time; callers are expected to produce
spectrograms via ``torchaudio.transforms.MelSpectrogram``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class AudioEncoder(nn.Module):
    """Convolutional + temporal transformer encoder for mel spectrograms.

    Architecture:
      1. 2-D convolutional front-end reduces frequency / time resolution.
      2. Linear projection to ``d_model``.
      3. Sinusoidal positional encoding along the time axis.
      4. Transformer encoder layers.
      5. Mean pooling → linear projection to ``latent_dim``.

    Args:
        n_mels: Number of mel frequency bins (height of input spectrogram).
        d_model: Internal transformer dimensionality.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout rate.
        latent_dim: Output embedding dimensionality.
        max_seq_len: Maximum temporal length after CNN down-sampling.

    Example::

        encoder = AudioEncoder(n_mels=80, latent_dim=256)
        spec = torch.randn(4, 80, 300)   # (B, F, T_a)
        z = encoder(spec)                # (4, 256)
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        latent_dim: int = 256,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()

        # 2-D CNN front-end: (B, 1, n_mels, T_a) → (B, C, n_mels', T')
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GELU(),
        )

        # Compute CNN output frequency dim
        cnn_freq_dim = math.ceil(math.ceil(n_mels / 2) / 2)  # after 2× stride-2
        cnn_channels = 64
        cnn_out_dim = cnn_channels * cnn_freq_dim

        self.input_proj = nn.Linear(cnn_out_dim, d_model)

        # Sinusoidal positional encoding
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_proj = nn.Linear(d_model, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, spectrogram: Tensor) -> Tensor:
        """Encode a batch of mel spectrograms.

        Args:
            spectrogram: ``(B, F, T_a)`` mel spectrogram tensor.

        Returns:
            Latent embedding ``(B, latent_dim)``.
        """
        B, F, T_a = spectrogram.shape

        # Add channel dim for 2-D conv: (B, 1, F, T_a)
        x = spectrogram.unsqueeze(1)
        x = self.cnn(x)               # (B, C, F', T')

        # Flatten frequency: (B, T', C*F')
        B_, C_, F_, T_ = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B_, T_, C_ * F_)

        x = self.input_proj(x)        # (B, T', d_model)
        x = x + self.pe[: x.size(1)]  # add positional encoding
        x = self.dropout(x)

        x = self.transformer(x)       # (B, T', d_model)
        x = x.mean(dim=1)             # (B, d_model)

        x = self.out_proj(x)          # (B, latent_dim)
        return self.layer_norm(x)
