"""
train_full_model.py — Production-grade distributed training for ChoreoAI.

Includes:
  - DDP (Distributed Data Parallel)
  - AMP (Automatic Mixed Precision)
  - Gradient Checkpointing
  - Early Stopping
  - Motion Metrics (Smoothness, Alignment)
"""

import os
import logging
from pathlib import Path
import math

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from omegaconf import DictConfig, OmegaConf

from choreoai.encoders.text_encoder import TextEncoder
from choreoai.generator.denoising_transformer import DenoisingTransformer
from choreoai.generator.diffusion_scheduler import DDPMScheduler
from choreoai.torch_dataset import build_dataloader
from choreoai.evaluate import compute_fmd, extract_motion_features

logger = logging.getLogger(__name__)

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def compute_smoothness(motion: torch.Tensor) -> torch.Tensor:
    """Compute motion smoothness as the negative average acceleration norm."""
    # motion: (B, T, K, 3)
    accel = motion[:, 2:] - 2 * motion[:, 1:-1] + motion[:, :-2]
    norm = torch.linalg.norm(accel, dim=-1).mean()
    return -norm # higher is smoother

def train_one_epoch(
    model, scheduler, encoder, optimizer, loader, 
    scaler, device, epoch, cfg, rank
):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        
        poses = batch["poses"].to(device)
        mask = batch["mask_pose"].to(device)
        kpm = ~mask
        B = poses.shape[0]
        
        t = torch.randint(0, scheduler.num_timesteps, (B,), device=device)
        
        with autocast(enabled=cfg.training.mixed_precision):
            x_t, noise = scheduler.forward_diffusion(poses, t)
            with torch.no_grad():
                z = encoder.encode_texts(batch.get("text_raw", [""] * B), device=device)
            
            noise_pred = model(x_t, t, z, key_padding_mask=kpm)
            
            valid = mask.unsqueeze(-1).unsqueeze(-1).float()
            loss = nn.functional.mse_loss(noise_pred * valid, noise * valid)
            loss = loss / cfg.training.gradient_accumulation_steps

        scaler.scale(loss).backward()
        
        if (step + 1) % cfg.training.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        
        if rank == 0 and step % 10 == 0:
            logger.info(f"Epoch {epoch} | Step {step}/{len(loader)} | Loss {loss.item():.6f}")

    return total_loss / len(loader)

@hydra.main(config_path="../../configs", config_name="train_full", version_base=None)
def main(cfg: DictConfig):
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if rank == 0:
        logger.info(f"World size: {world_size}")
        Path(cfg.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Data
    loader = build_dataloader(
        manifest_path=cfg.data.manifest,
        batch_size=cfg.training.batch_size // world_size,
        num_workers=cfg.training.num_workers,
        max_seq_len=cfg.data.max_seq_len,
        num_joints=cfg.model.num_joints,
        shuffle=False, # Sampler handles shuffling
    )
    if world_size > 1:
        sampler = DistributedSampler(loader.dataset, num_replicas=world_size, rank=rank)
        loader = torch.utils.data.DataLoader(
            loader.dataset, 
            batch_size=loader.batch_size, 
            sampler=sampler,
            num_workers=loader.num_workers,
            collate_fn=loader.collate_fn
        )

    # Model
    model = DenoisingTransformer(
        num_joints=cfg.model.num_joints,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.gen_layers,
        latent_dim=cfg.model.latent_dim,
        max_seq_len=cfg.data.max_seq_len + 10,
        max_timesteps=cfg.diffusion.num_timesteps,
    ).to(device)

    if cfg.training.gradient_checkpointing:
        model.model.gradient_checkpointing_enable()

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    encoder = TextEncoder(latent_dim=cfg.model.latent_dim).to(device)
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False

    scheduler = DDPMScheduler(num_timesteps=cfg.diffusion.num_timesteps).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scaler = GradScaler(enabled=cfg.training.mixed_precision)

    best_loss = float("inf")
    patience = 0

    for epoch in range(cfg.training.epochs):
        if world_size > 1:
            loader.sampler.set_epoch(epoch)
        
        loss = train_one_epoch(model, scheduler, encoder, optimizer, loader, scaler, device, epoch, cfg, rank)
        
        if rank == 0:
            logger.info(f"Epoch {epoch} finished. Average Loss: {loss:.6f}")
            
            # Save periodic
            if (epoch + 1) % cfg.training.save_every == 0:
                torch.save(model.state_dict(), Path(cfg.training.checkpoint_dir) / f"ckpt_epoch_{epoch}.pt")
            
            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience = 0
                torch.save(model.state_dict(), Path(cfg.training.checkpoint_dir) / "best_model.pt")
            else:
                patience += 1
                if patience >= cfg.training.early_stopping_patience:
                    logger.info("Early stopping triggered.")
                    break

    cleanup_distributed()

if __name__ == "__main__":
    main()
