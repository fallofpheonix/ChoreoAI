"""
train_generator.py — Diffusion generator training loop.

Trains a conditional :class:`~choreoai.generator.DenoisingTransformer`
on clean motion sequences using the DDPM objective.

The conditioning vector ``z`` is obtained by running the frozen motion
encoder on the *clean* pose (self-conditioning), or optionally from a
text/image/audio encoder loaded from a contrastive checkpoint.

Typical usage::

    python -m choreoai.train_generator data.manifest=data/manifest.json \\
        training.contrastive_ckpt=checkpoints/contrastive/contrastive_epoch0099.pt
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

from choreoai.encoders.motion_encoder import MotionEncoder
from choreoai.encoders.text_encoder import TextEncoder
from choreoai.generator.denoising_transformer import DenoisingTransformer
from choreoai.generator.diffusion_scheduler import DDPMScheduler
from choreoai.torch_dataset import build_dataloader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: DenoisingTransformer,
    scheduler: DDPMScheduler,
    encoder: nn.Module,
    optimizer: optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    use_wandb: bool = False,
) -> float:
    """Run one epoch of diffusion training.

    Args:
        model: Conditional denoising transformer.
        scheduler: DDPM noise scheduler.
        encoder: Frozen conditioning encoder (motion or text encoder).
        optimizer: Optimizer.
        dataloader: Training data loader.
        device: Compute device.
        epoch: Current epoch index.
        use_wandb: Whether to log to W&B.

    Returns:
        Mean MSE loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    mse = nn.MSELoss()

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        poses = batch["poses"].to(device)    # (B, T, K, 3)
        mask = batch["mask_pose"].to(device) # (B, T)
        kpm = ~mask                          # (B, T) True = padded

        B = poses.shape[0]

        # Sample random timesteps
        t = torch.randint(0, scheduler.num_timesteps, (B,), device=device)

        # Forward diffusion: add noise
        x_t, noise = scheduler.forward_diffusion(poses, t)

        # Conditioning: use frozen encoder on clean poses
        with torch.no_grad():
            z = encoder(poses, key_padding_mask=kpm)  # (B, latent_dim)

        # Predict noise
        noise_pred = model(x_t, t, z, key_padding_mask=kpm)

        # Apply mask to compute loss only on valid frames
        valid = mask.unsqueeze(-1).unsqueeze(-1).float()  # (B, T, 1, 1)
        loss = mse(noise_pred * valid, noise * valid)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if use_wandb:
            try:
                import wandb

                wandb.log(
                    {
                        "gen/step_loss": loss.item(),
                        "gen/step": epoch * num_batches + step,
                    }
                )
            except Exception:
                pass

        if step % max(1, num_batches // 10) == 0:
            logger.info(
                "Epoch %d | step %d/%d | loss %.6f",
                epoch, step, num_batches, loss.item(),
            )

    return total_loss / num_batches


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: DenoisingTransformer,
    optimizer: optim.Optimizer,
    epoch: int,
    ckpt_dir: Path,
) -> None:
    """Save a generator training checkpoint.

    Args:
        model: Denoising transformer.
        optimizer: Optimizer.
        epoch: Current epoch.
        ckpt_dir: Destination directory.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"generator_epoch{epoch:04d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        ckpt_path,
    )
    logger.info("Checkpoint saved → %s", ckpt_path)


# ---------------------------------------------------------------------------
# Main entry point (Hydra)
# ---------------------------------------------------------------------------


@hydra.main(config_path="../../configs", config_name="train_generator", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point for generator training.

    Args:
        cfg: Hydra configuration object.
    """
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # W&B
    use_wandb = cfg.get("wandb", {}).get("enabled", False)
    if use_wandb:
        try:
            import wandb

            wandb.init(
                project=cfg.wandb.get("project", "choreoai"),
                name=cfg.wandb.get("run_name", "generator"),
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

    # Conditioning encoder (frozen)
    encoder = MotionEncoder(
        num_joints=cfg.model.num_joints,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        latent_dim=cfg.model.latent_dim,
    ).to(device)

    # Load contrastive checkpoint if provided
    ckpt_path = cfg.training.get("contrastive_ckpt", None)
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        if "encoders" in ckpt and "motion" in ckpt["encoders"]:
            encoder.load_state_dict(ckpt["encoders"]["motion"])
            logger.info("Loaded motion encoder weights from %s", ckpt_path)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Denoising model
    model = DenoisingTransformer(
        num_joints=cfg.model.num_joints,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.get("gen_layers", 8),
        latent_dim=cfg.model.latent_dim,
        max_seq_len=cfg.data.max_seq_len + 10,
        max_timesteps=cfg.diffusion.get("num_timesteps", 1000),
    ).to(device)

    scheduler = DDPMScheduler(
        num_timesteps=cfg.diffusion.get("num_timesteps", 1000),
        beta_start=cfg.diffusion.get("beta_start", 1e-4),
        beta_end=cfg.diffusion.get("beta_end", 0.02),
        schedule=cfg.diffusion.get("schedule", "cosine"),
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.get("lr", 1e-4),
        weight_decay=cfg.training.get("weight_decay", 1e-2),
    )

    ckpt_dir = Path(cfg.training.get("checkpoint_dir", "checkpoints/generator"))

    for epoch in range(cfg.training.epochs):
        epoch_loss = train_one_epoch(
            model, scheduler, encoder, optimizer, loader,
            device=device, epoch=epoch, use_wandb=use_wandb,
        )
        logger.info("Epoch %d | mean loss %.6f", epoch, epoch_loss)

        if use_wandb:
            try:
                import wandb
                wandb.log({"gen/epoch_loss": epoch_loss, "epoch": epoch})
            except Exception:
                pass

        if (epoch + 1) % cfg.training.get("save_every", 10) == 0:
            save_checkpoint(model, optimizer, epoch, ckpt_dir)

    save_checkpoint(model, optimizer, cfg.training.epochs - 1, ckpt_dir)
    logger.info("Generator training complete.")


if __name__ == "__main__":
    main()
