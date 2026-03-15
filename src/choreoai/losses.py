"""
losses.py — Contrastive and auxiliary losses for ChoreoAI.

Implements InfoNCE (NT-Xent / CLIP-style) contrastive loss for aligning
paired embeddings in a shared latent space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss with learnable or fixed temperature.

    Given two sets of embeddings ``z1`` and ``z2`` (one from each modality),
    treats diagonal pairs as positives and all off-diagonal pairs as
    in-batch negatives.  Loss is computed symmetrically and averaged.

    Args:
        temperature: Softmax temperature.  If ``learn_temperature`` is
            ``True`` this serves as the initial value.
        learn_temperature: Whether to make the temperature a learnable
            scalar parameter (à la CLIP).

    Example::

        loss_fn = InfoNCELoss(temperature=0.07)
        z1 = motion_encoder(poses)       # (B, D)
        z2 = text_encoder(tokens)        # (B, D)
        loss = loss_fn(z1, z2)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learn_temperature: bool = False,
    ) -> None:
        super().__init__()
        if learn_temperature:
            self.log_temperature = nn.Parameter(
                torch.tensor(temperature).log()
            )
        else:
            self.register_buffer(
                "log_temperature", torch.tensor(temperature).log()
            )

    @property
    def temperature(self) -> Tensor:
        """Current temperature (clamped for numerical stability)."""
        return self.log_temperature.exp().clamp(min=1e-4, max=1.0)

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        """Compute symmetric InfoNCE loss.

        Args:
            z1: Embedding matrix of shape ``(B, D)``.
            z2: Embedding matrix of shape ``(B, D)``.

        Returns:
            Scalar loss tensor.
        """
        assert z1.shape == z2.shape, (
            f"Embedding shapes must match: {z1.shape} vs {z2.shape}"
        )
        B = z1.size(0)

        # L2-normalise embeddings
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Cosine similarity matrix  (B, B)
        logits = (z1 @ z2.T) / self.temperature  # (B, B)

        # Targets: diagonal (i == j) are positives
        targets = torch.arange(B, device=z1.device)

        loss_1 = F.cross_entropy(logits, targets)        # z1 → z2
        loss_2 = F.cross_entropy(logits.T, targets)      # z2 → z1

        return (loss_1 + loss_2) / 2.0


def infonce_loss(
    z1: Tensor,
    z2: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """Functional wrapper for InfoNCE loss.

    Useful when you don't want to manage an :class:`InfoNCELoss` module.

    Args:
        z1: ``(B, D)`` embedding from the first modality.
        z2: ``(B, D)`` embedding from the second modality.
        temperature: Softmax temperature.

    Returns:
        Scalar loss tensor.
    """
    B = z1.size(0)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = (z1 @ z2.T) / temperature
    targets = torch.arange(B, device=z1.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2.0


class MultiModalAlignmentLoss(nn.Module):
    """Aggregate InfoNCE loss across multiple modality pairs.

    Computes a weighted average of pairwise InfoNCE losses between a
    motion embedding and each available auxiliary modality.

    Args:
        temperature: Softmax temperature for each pairwise loss.
        learn_temperature: Whether to make temperature learnable.
        modality_weights: Optional per-modality weights
            ``{"text": 1.0, "image": 1.0, "audio": 1.0}``.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learn_temperature: bool = False,
        modality_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self._infonce = InfoNCELoss(
            temperature=temperature, learn_temperature=learn_temperature
        )
        self.modality_weights: dict[str, float] = modality_weights or {
            "text": 1.0,
            "image": 1.0,
            "audio": 1.0,
        }

    def forward(
        self,
        motion_z: Tensor,
        modality_embeddings: dict[str, Tensor | None],
    ) -> Tensor:
        """Compute multi-modal alignment loss.

        Args:
            motion_z: Motion embeddings ``(B, D)``.
            modality_embeddings: Dict mapping modality name to its
                ``(B, D)`` embedding tensor or ``None`` if absent.

        Returns:
            Weighted average scalar loss.
        """
        total_loss = torch.tensor(0.0, device=motion_z.device)
        total_weight = 0.0

        for modality, z_mod in modality_embeddings.items():
            if z_mod is None:
                continue
            weight = self.modality_weights.get(modality, 1.0)
            total_loss = total_loss + weight * self._infonce(motion_z, z_mod)
            total_weight += weight

        if total_weight == 0.0:
            return total_loss  # zero

        return total_loss / total_weight
