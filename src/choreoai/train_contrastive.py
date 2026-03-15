"""
train_contrastive.py — Contrastive alignment training loop.

Trains motion and auxiliary encoders to align in a shared latent space
using InfoNCE loss.  Supports text, image, and audio modalities.

Typical usage::

    python -m choreoai.train_contrastive data.manifest=data/manifest.json
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from choreoai.encoders.motion_encoder import MotionEncoder
from choreoai.encoders.text_encoder import TextEncoder
from choreoai.encoders.image_encoder import ImageEncoder
from choreoai.encoders.audio_encoder import AudioEncoder
from choreoai.losses import MultiModalAlignmentLoss
from choreoai.torch_dataset import build_dataloader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def _build_encoders(cfg: DictConfig, device: torch.device) -> dict[str, torch.nn.Module]:
    """Instantiate and move all encoders to *device*."""
    encoders: dict[str, torch.nn.Module] = {}

    encoders["motion"] = MotionEncoder(
        num_joints=cfg.model.num_joints,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        latent_dim=cfg.model.latent_dim,
    ).to(device)

    if cfg.model.get("use_text", True):
        encoders["text"] = TextEncoder(
            model_name=cfg.model.get("text_model", "roberta-base"),
            latent_dim=cfg.model.latent_dim,
            freeze_base=cfg.model.get("freeze_text", True),
        ).to(device)

    if cfg.model.get("use_image", False):
        encoders["image"] = ImageEncoder(
            model_name=cfg.model.get("image_model", "vit_base_patch16_224"),
            latent_dim=cfg.model.latent_dim,
            freeze_base=cfg.model.get("freeze_image", True),
        ).to(device)

    if cfg.model.get("use_audio", False):
        encoders["audio"] = AudioEncoder(
            n_mels=cfg.model.get("n_mels", 80),
            latent_dim=cfg.model.latent_dim,
        ).to(device)

    return encoders


def _trainable_params(encoders: dict[str, torch.nn.Module]) -> list[torch.nn.Parameter]:
    """Collect all trainable parameters across encoders."""
    params: list[torch.nn.Parameter] = []
    for enc in encoders.values():
        params.extend(p for p in enc.parameters() if p.requires_grad)
    return params


def train_one_epoch(
    encoders: dict[str, torch.nn.Module],
    loss_fn: MultiModalAlignmentLoss,
    optimizer: optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    use_wandb: bool = False,
) -> float:
    """Run one full training epoch.

    Args:
        encoders: Dict of encoder modules.
        loss_fn: Multi-modal alignment loss.
        optimizer: Optimizer instance.
        dataloader: Training data loader.
        device: Compute device.
        epoch: Current epoch number (for logging).
        use_wandb: Whether W&B logging is active.

    Returns:
        Mean loss over the epoch.
    """
    for enc in encoders.values():
        enc.train()
    loss_fn.train()

    total_loss = 0.0
    num_batches = len(dataloader)

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Move tensors to device
        poses = batch["poses"].to(device)          # (B, T, K, 3)
        mask = batch["mask_pose"].to(device)       # (B, T) bool; True = valid

        # Motion embedding
        # key_padding_mask: True = padded (invalid) — invert mask
        kpm = ~mask  # (B, T)
        motion_z = encoders["motion"](poses, key_padding_mask=kpm)

        # Auxiliary modality embeddings
        modality_z: dict[str, torch.Tensor | None] = {}

        if "text" in encoders and batch["text"] is not None:
            text_tokens = batch["text"].to(device)
            modality_z["text"] = encoders["text"](text_tokens)

        if "image" in encoders and batch["image"] is not None:
            images = batch["image"].to(device)
            modality_z["image"] = encoders["image"](images)

        if "audio" in encoders and batch["audio"] is not None:
            audio = batch["audio"].to(device)
            modality_z["audio"] = encoders["audio"](audio)

        loss = loss_fn(motion_z, modality_z)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(_trainable_params(encoders), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if use_wandb:
            try:
                import wandb

                wandb.log(
                    {
                        "train/step_loss": loss.item(),
                        "train/step": epoch * num_batches + step,
                    }
                )
            except Exception:
                pass

        if step % max(1, num_batches // 10) == 0:
            logger.info(
                "Epoch %d | step %d/%d | loss %.4f",
                epoch, step, num_batches, loss.item(),
            )

    return total_loss / num_batches


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    encoders: dict[str, torch.nn.Module],
    optimizer: optim.Optimizer,
    epoch: int,
    ckpt_dir: Path,
) -> None:
    """Save a training checkpoint.

    Args:
        encoders: Dict of encoder modules.
        optimizer: Optimizer.
        epoch: Current epoch.
        ckpt_dir: Directory to write checkpoints.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"contrastive_epoch{epoch:04d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "encoders": {k: v.state_dict() for k, v in encoders.items()},
            "optimizer": optimizer.state_dict(),
        },
        ckpt_path,
    )
    logger.info("Checkpoint saved → %s", ckpt_path)


# ---------------------------------------------------------------------------
# Main entry point (Hydra)
# ---------------------------------------------------------------------------


@hydra.main(config_path="../../configs", config_name="train_contrastive", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point for contrastive training.

    Args:
        cfg: Hydra configuration object.
    """
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # W&B initialisation
    use_wandb = cfg.get("wandb", {}).get("enabled", False)
    if use_wandb:
        try:
            import wandb

            wandb.init(
                project=cfg.wandb.get("project", "choreoai"),
                name=cfg.wandb.get("run_name", "contrastive"),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception as exc:
            logger.warning("W&B init failed: %s", exc)
            use_wandb = False

    # Data
    loader = build_dataloader(
        manifest_path=cfg.data.manifest,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.get("num_workers", 4),
        max_seq_len=cfg.data.max_seq_len,
        num_joints=cfg.model.num_joints,
    )

    # Encoders + loss
    encoders = _build_encoders(cfg, device)
    loss_fn = MultiModalAlignmentLoss(
        temperature=cfg.model.get("temperature", 0.07),
        learn_temperature=cfg.model.get("learn_temperature", True),
    ).to(device)

    # Optimiser
    all_params = _trainable_params(encoders) + list(loss_fn.parameters())
    optimizer = optim.AdamW(
        all_params,
        lr=cfg.training.get("lr", 1e-4),
        weight_decay=cfg.training.get("weight_decay", 1e-2),
    )

    ckpt_dir = Path(cfg.training.get("checkpoint_dir", "checkpoints/contrastive"))

    # Training loop
    for epoch in range(cfg.training.epochs):
        epoch_loss = train_one_epoch(
            encoders, loss_fn, optimizer, loader, device,
            epoch=epoch, use_wandb=use_wandb,
        )
        logger.info("Epoch %d | mean loss %.4f", epoch, epoch_loss)

        if use_wandb:
            try:
                import wandb
                wandb.log({"train/epoch_loss": epoch_loss, "epoch": epoch})
            except Exception:
                pass

        if (epoch + 1) % cfg.training.get("save_every", 10) == 0:
            save_checkpoint(encoders, optimizer, epoch, ckpt_dir)

    save_checkpoint(encoders, optimizer, cfg.training.epochs - 1, ckpt_dir)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
