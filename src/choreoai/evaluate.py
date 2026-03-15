"""
evaluate.py — Evaluation metrics for motion generation.

Implements:

* **Fréchet Motion Distance (FMD)** — analogous to FID for images.
  Uses the feature statistics (mean + covariance) of the motion encoder
  to measure distributional distance between real and generated sequences.
* **Semantic Retrieval Accuracy** — top-K accuracy of cross-modal retrieval
  using cosine similarity in the shared latent space.
* **Average Joint Position Error (AJPE)** — simple per-frame, per-joint
  L2 error for supervised settings where ground-truth exists.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fréchet Motion Distance (FMD)
# ---------------------------------------------------------------------------


def _compute_feature_stats(features: Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of a feature matrix.

    Args:
        features: ``(N, D)`` feature tensor.

    Returns:
        Tuple ``(mean, cov)`` as numpy arrays.
    """
    arr = features.cpu().float().numpy()     # (N, D)
    mu = arr.mean(axis=0)                    # (D,)
    sigma = np.cov(arr, rowvar=False)        # (D, D)
    return mu, sigma


def _frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute the Fréchet distance between two Gaussian distributions.

    FD = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2 * √(Σ₁·Σ₂))

    Uses the scipy matrix square-root implementation.

    Args:
        mu1: Mean of distribution 1.
        sigma1: Covariance of distribution 1.
        mu2: Mean of distribution 2.
        sigma2: Covariance of distribution 2.
        eps: Small offset added to diagonal for numerical stability.

    Returns:
        Fréchet distance scalar.
    """
    from scipy import linalg  # type: ignore[import-untyped]

    diff = mu1 - mu2
    # Product of covariance matrices
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Numerical clean-up
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fd = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return fd


def compute_fmd(
    real_features: Tensor,
    gen_features: Tensor,
) -> float:
    """Fréchet Motion Distance between real and generated feature sets.

    Args:
        real_features: ``(N_real, D)`` features of real motion sequences.
        gen_features: ``(N_gen, D)`` features of generated sequences.

    Returns:
        Scalar FMD value (lower is better).
    """
    mu_r, sigma_r = _compute_feature_stats(real_features)
    mu_g, sigma_g = _compute_feature_stats(gen_features)
    return _frechet_distance(mu_r, sigma_r, mu_g, sigma_g)


def extract_motion_features(
    poses_list: list[Tensor],
    encoder: torch.nn.Module,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
) -> Tensor:
    """Extract motion encoder features for a list of pose tensors.

    Args:
        poses_list: List of ``(T, K, 3)`` tensors (variable length OK).
        encoder: Motion encoder returning ``(B, D)`` embeddings.
        device: Compute device.
        batch_size: Encoding batch size.

    Returns:
        ``(N, D)`` feature tensor.
    """
    encoder.eval()
    all_feats: list[Tensor] = []

    with torch.no_grad():
        for i in range(0, len(poses_list), batch_size):
            batch = poses_list[i : i + batch_size]
            # Pad to same length
            max_t = max(p.shape[0] for p in batch)
            padded: list[Tensor] = []
            masks: list[Tensor] = []
            for p in batch:
                T = p.shape[0]
                pad_len = max_t - T
                if pad_len > 0:
                    pad = torch.zeros(pad_len, *p.shape[1:], dtype=p.dtype)
                    p = torch.cat([p, pad], dim=0)
                mask = torch.zeros(max_t, dtype=torch.bool)
                mask[:T] = True
                padded.append(p)
                masks.append(mask)

            poses_t = torch.stack(padded).to(device)   # (B, T, K, 3)
            mask_t = torch.stack(masks).to(device)     # (B, T)
            kpm = ~mask_t

            feats = encoder(poses_t, key_padding_mask=kpm)
            all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0)  # (N, D)


# ---------------------------------------------------------------------------
# Semantic Retrieval Accuracy
# ---------------------------------------------------------------------------


def retrieval_accuracy(
    query_embeddings: Tensor,
    gallery_embeddings: Tensor,
    top_k: int | list[int] = 1,
) -> dict[str, float]:
    """Compute top-K retrieval accuracy.

    For each query, ranks gallery items by cosine similarity and checks
    whether the diagonal (ground-truth pair) falls within the top-K.
    Assumes ``query_embeddings[i]`` is paired with ``gallery_embeddings[i]``.

    Args:
        query_embeddings: ``(N, D)`` query embedding matrix.
        gallery_embeddings: ``(N, D)`` gallery embedding matrix.
        top_k: Single or list of K values.

    Returns:
        Dict ``{"R@1": float, "R@5": float, ...}`` with recall values.
    """
    if isinstance(top_k, int):
        top_k = [top_k]

    q = F.normalize(query_embeddings, dim=-1)
    g = F.normalize(gallery_embeddings, dim=-1)

    sim = q @ g.T  # (N, N) cosine similarity matrix
    N = sim.size(0)
    ground_truth = torch.arange(N, device=sim.device)

    results: dict[str, float] = {}
    for k in top_k:
        _, top_indices = sim.topk(k, dim=1)                        # (N, k)
        correct = (top_indices == ground_truth.unsqueeze(1)).any(dim=1)
        results[f"R@{k}"] = correct.float().mean().item()

    return results


# ---------------------------------------------------------------------------
# Average Joint Position Error
# ---------------------------------------------------------------------------


def average_joint_position_error(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
) -> float:
    """Compute mean per-joint Euclidean position error.

    Args:
        pred: ``(B, T, K, 3)`` predicted pose tensor.
        target: ``(B, T, K, 3)`` ground-truth pose tensor.
        mask: Optional ``(B, T)`` bool mask (``True`` = valid frame).
            Padded frames are excluded from the mean.

    Returns:
        Mean joint position error (scalar).
    """
    assert pred.shape == target.shape

    # Per-joint per-frame L2 error: (B, T, K)
    err = (pred - target).norm(dim=-1)

    if mask is not None:
        # Mask out padded frames
        valid = mask.unsqueeze(-1).float()  # (B, T, 1)
        err = err * valid
        total_valid = valid.sum() * pred.shape[2]  # B*T_valid*K
        return (err.sum() / total_valid.clamp(min=1)).item()

    return err.mean().item()
