"""
inference.py — Text-to-motion generation pipeline.

Converts a natural-language text prompt into a ``(T, K, 3)`` skeleton
sequence via:

  1. Encode text → latent ``z`` using :class:`~choreoai.encoders.TextEncoder`
  2. Run full DDPM reverse diffusion conditioned on ``z``
  3. Return the generated ``(T, K, 3)`` motion tensor

Example::

    from choreoai.inference import generate_motion

    motion = generate_motion(
        text="a dancer performs a slow waltz",
        text_encoder_ckpt="checkpoints/contrastive/contrastive_epoch0099.pt",
        generator_ckpt="checkpoints/generator/generator_epoch0099.pt",
    )
    print(motion.shape)  # (120, 17, 3)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from choreoai.encoders.text_encoder import TextEncoder
from choreoai.generator.denoising_transformer import DenoisingTransformer
from choreoai.generator.diffusion_scheduler import DDPMScheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------


def _load_text_encoder(
    ckpt_path: str | Path | None,
    latent_dim: int,
    model_name: str,
    device: torch.device,
) -> TextEncoder:
    """Instantiate and optionally load a TextEncoder from a checkpoint.

    Args:
        ckpt_path: Path to a contrastive training checkpoint (optional).
        latent_dim: Shared latent dimensionality.
        model_name: HuggingFace model identifier.
        device: Target device.

    Returns:
        TextEncoder in eval mode.
    """
    encoder = TextEncoder(
        model_name=model_name,
        latent_dim=latent_dim,
        freeze_base=True,
    ).to(device)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        state_dict: dict[str, Any] | None = None
        if "encoders" in ckpt and "text" in ckpt["encoders"]:
            state_dict = ckpt["encoders"]["text"]
        elif "text_encoder" in ckpt:
            state_dict = ckpt["text_encoder"]
        if state_dict is not None:
            encoder.load_state_dict(state_dict)
            logger.info("Loaded text encoder weights from %s", ckpt_path)
        else:
            logger.warning("No text encoder state dict found in %s", ckpt_path)

    encoder.eval()
    return encoder


def _load_generator(
    ckpt_path: str | Path | None,
    num_joints: int,
    latent_dim: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    max_seq_len: int,
    num_timesteps: int,
    device: torch.device,
) -> DenoisingTransformer:
    """Instantiate and optionally load a DenoisingTransformer.

    Args:
        ckpt_path: Path to a generator training checkpoint (optional).
        num_joints: Number of skeleton joints.
        latent_dim: Conditioning embedding dimensionality.
        d_model: Model dimension.
        nhead: Attention heads.
        num_layers: Transformer depth.
        max_seq_len: Maximum sequence length.
        num_timesteps: Diffusion timesteps.
        device: Target device.

    Returns:
        DenoisingTransformer in eval mode.
    """
    model = DenoisingTransformer(
        num_joints=num_joints,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        latent_dim=latent_dim,
        max_seq_len=max_seq_len + 10,
        max_timesteps=num_timesteps,
    ).to(device)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        state_key = "model" if "model" in ckpt else "state_dict"
        model.load_state_dict(ckpt[state_key])
        logger.info("Loaded generator weights from %s", ckpt_path)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_motion(
    text: str | list[str],
    *,
    text_encoder_ckpt: str | Path | None = None,
    generator_ckpt: str | Path | None = None,
    num_joints: int = 17,
    seq_len: int = 120,
    latent_dim: int = 256,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 8,
    num_timesteps: int = 1000,
    text_model: str = "roberta-base",
    device: str | torch.device | None = None,
) -> Tensor:
    """Generate a skeleton motion sequence from a text prompt.

    Args:
        text: A single text string or a list of strings (batch).
        text_encoder_ckpt: Path to contrastive checkpoint containing the
            text encoder weights (optional — uses untrained encoder if ``None``).
        generator_ckpt: Path to the generator checkpoint (optional).
        num_joints: Number of skeleton joints ``K``.
        seq_len: Temporal length of the generated sequence ``T``.
        latent_dim: Shared latent space dimensionality.
        d_model: Denoising transformer model dim.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers in generator.
        num_timesteps: DDPM diffusion timesteps.
        text_model: HuggingFace model name for text encoder.
        device: Target device (defaults to CUDA if available).

    Returns:
        Generated motion tensor of shape ``(B, T, K, 3)`` for batch input or
        ``(T, K, 3)`` for a single string.
    """
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(device)

    batch_input = isinstance(text, list)
    texts = text if batch_input else [text]

    # -- Text encoder --
    text_encoder = _load_text_encoder(
        ckpt_path=text_encoder_ckpt,
        latent_dim=latent_dim,
        model_name=text_model,
        device=_device,
    )

    with torch.no_grad():
        z = text_encoder.encode_texts(texts, device=_device)  # (B, latent_dim)

    # -- Generator --
    generator = _load_generator(
        ckpt_path=generator_ckpt,
        num_joints=num_joints,
        latent_dim=latent_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_seq_len=seq_len,
        num_timesteps=num_timesteps,
        device=_device,
    )

    scheduler = DDPMScheduler(
        num_timesteps=num_timesteps,
        schedule="cosine",
    ).to(_device)

    B = z.size(0)
    shape = (B, seq_len, num_joints, 3)

    logger.info(
        "Generating motion: batch=%d, seq_len=%d, num_joints=%d, device=%s",
        B, seq_len, num_joints, _device,
    )

    motion = scheduler.sample(
        model=generator,
        shape=shape,
        conditioning=z,
        device=_device,
    )  # (B, T, K, 3)

    if not batch_input:
        motion = motion.squeeze(0)  # (T, K, 3)

    return motion
