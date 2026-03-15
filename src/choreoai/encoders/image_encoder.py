"""
image_encoder.py — Pre-trained image encoder using timm with projection head.

Loads a ViT-B/16 (or any timm model) backbone, removes the classification
head, and adds a trainable projection to the shared latent space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

try:
    import timm  # type: ignore[import-untyped]
except ImportError as exc:
    raise ImportError(
        "`timm` is required for ImageEncoder. Install with: pip install timm"
    ) from exc


class ImageEncoder(nn.Module):
    """Frozen timm image backbone with trainable projection head.

    Args:
        model_name: Any timm model identifier. Defaults to
            ``"vit_base_patch16_224"`` (ViT-B/16).
        pretrained: Load ImageNet-pretrained weights.
        latent_dim: Output dimensionality.
        freeze_base: Whether to freeze backbone weights.

    Example::

        encoder = ImageEncoder(latent_dim=256)
        img = torch.randn(4, 3, 224, 224)
        z = encoder(img)   # (4, 256)
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        latent_dim: int = 256,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()

        # Load backbone without classification head
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Determine feature dimension from backbone
        feature_dim: int = self.backbone.num_features

        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Encode a batch of images.

        Args:
            images: ``(B, C, H, W)`` normalised image tensor.

        Returns:
            Latent embedding ``(B, latent_dim)``.
        """
        features: Tensor = self.backbone(images)  # (B, feature_dim)
        return self.projection(features)           # (B, latent_dim)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def default_transform() -> "torchvision.transforms.Compose":  # type: ignore[name-defined]  # noqa: F821
        """Return the standard ImageNet preprocessing transform.

        Returns:
            ``torchvision.transforms.Compose`` pipeline.
        """
        from torchvision import transforms  # type: ignore[import-untyped]

        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
