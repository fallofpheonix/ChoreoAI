"""
preprocess.py — Pose preprocessing and normalisation utilities.

Provides functions to clean, normalise, and augment raw ``(T, K, 3)``
skeleton tensors before feeding them into the training pipeline.
"""

from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Core normalisation
# ---------------------------------------------------------------------------


def center_poses(poses: Tensor, hip_joint: int = 0) -> Tensor:
    """Translate pose sequences so that the hip joint is at the origin.

    Args:
        poses: ``(T, K, 3)`` skeleton tensor.
        hip_joint: Index of the root joint (default 0 = pelvis / hip).

    Returns:
        Centred pose tensor of the same shape.
    """
    root = poses[:, hip_joint : hip_joint + 1, :]  # (T, 1, 3)
    return poses - root


def scale_poses(poses: Tensor, target_scale: float = 1.0) -> Tensor:
    """Scale poses so that the mean bone length equals *target_scale*.

    Uses the L2 norm of all per-frame, per-joint displacements from the
    origin as a proxy for skeleton size.

    Args:
        poses: ``(T, K, 3)`` skeleton tensor.
        target_scale: Desired mean distance from root after scaling.

    Returns:
        Scaled pose tensor.
    """
    scale = poses.norm(dim=-1).mean()  # scalar
    if scale < 1e-6:
        return poses
    return poses * (target_scale / scale)


def normalize_poses(
    poses: Tensor,
    *,
    hip_joint: int = 0,
    target_scale: float = 1.0,
) -> Tensor:
    """Center + scale a pose sequence.

    Convenience wrapper around :func:`center_poses` and
    :func:`scale_poses`.

    Args:
        poses: ``(T, K, 3)`` skeleton tensor.
        hip_joint: Root joint index.
        target_scale: Target skeleton scale after normalisation.

    Returns:
        Normalised ``(T, K, 3)`` tensor.
    """
    poses = center_poses(poses, hip_joint=hip_joint)
    poses = scale_poses(poses, target_scale=target_scale)
    return poses


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def random_rotation_y(poses: Tensor, max_angle_deg: float = 30.0) -> Tensor:
    """Apply a random rotation around the Y-axis (vertical).

    Useful for data augmentation during training.

    Args:
        poses: ``(T, K, 3)`` skeleton tensor.
        max_angle_deg: Maximum rotation angle in degrees.

    Returns:
        Rotated pose tensor.
    """
    angle = torch.empty(1).uniform_(-max_angle_deg, max_angle_deg).item()
    theta = float(angle) * (3.14159265 / 180.0)
    cos_t = float(torch.tensor(theta).cos())
    sin_t = float(torch.tensor(theta).sin())

    # Rotation matrix around Y
    R = torch.tensor(
        [
            [cos_t, 0.0, sin_t],
            [0.0, 1.0, 0.0],
            [-sin_t, 0.0, cos_t],
        ],
        dtype=poses.dtype,
        device=poses.device,
    )  # (3, 3)

    return poses @ R.T  # (T, K, 3) × (3, 3) → (T, K, 3)


def temporal_jitter(poses: Tensor, max_shift: int = 5) -> Tensor:
    """Randomly shift the temporal start position of a sequence.

    Frames from the end are repeated to maintain fixed length.

    Args:
        poses: ``(T, K, 3)`` pose tensor.
        max_shift: Maximum number of frames to shift.

    Returns:
        Jittered ``(T, K, 3)`` tensor.
    """
    shift = int(torch.randint(0, max_shift + 1, (1,)).item())
    if shift == 0:
        return poses
    padding = poses[-1:].expand(shift, -1, -1)  # repeat last frame
    return torch.cat([poses[shift:], padding], dim=0)


def mirror_poses(poses: Tensor, mirror_x: bool = True) -> Tensor:
    """Flip poses along the X-axis (left-right mirror).

    Args:
        poses: ``(T, K, 3)`` skeleton tensor.
        mirror_x: Whether to apply the mirror.

    Returns:
        Mirrored (or unmodified) pose tensor.
    """
    if not mirror_x:
        return poses
    flipped = poses.clone()
    flipped[..., 0] = -flipped[..., 0]
    return flipped


# ---------------------------------------------------------------------------
# Velocity / acceleration features
# ---------------------------------------------------------------------------


def compute_velocity(poses: Tensor) -> Tensor:
    """Compute first-order finite differences along the temporal axis.

    Args:
        poses: ``(T, K, 3)`` pose tensor.

    Returns:
        Velocity tensor of shape ``(T-1, K, 3)``.
    """
    return poses[1:] - poses[:-1]


def compute_acceleration(poses: Tensor) -> Tensor:
    """Compute second-order finite differences (velocity of velocity).

    Args:
        poses: ``(T, K, 3)`` pose tensor.

    Returns:
        Acceleration tensor of shape ``(T-2, K, 3)``.
    """
    vel = compute_velocity(poses)
    return compute_velocity(vel)


def append_velocity(poses: Tensor) -> Tensor:
    """Append velocity features to joint coordinates.

    The first frame receives zero velocity.

    Args:
        poses: ``(T, K, 3)`` pose tensor.

    Returns:
        ``(T, K, 6)`` tensor with position + velocity concatenated on the
        last dimension.
    """
    vel = torch.zeros_like(poses)  # (T, K, 3)
    vel[1:] = compute_velocity(poses)
    return torch.cat([poses, vel], dim=-1)  # (T, K, 6)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def compute_mean_std(
    poses_list: list[Tensor],
) -> tuple[Tensor, Tensor]:
    """Compute per-joint-per-axis mean and standard deviation over a dataset.

    Args:
        poses_list: List of ``(T_i, K, 3)`` tensors.

    Returns:
        Tuple ``(mean, std)`` each of shape ``(K, 3)``.
    """
    all_poses = torch.cat(poses_list, dim=0)  # (sum_T, K, 3)
    mean = all_poses.mean(dim=0)              # (K, 3)
    std = all_poses.std(dim=0).clamp(min=1e-6)
    return mean, std


def standardize_poses(
    poses: Tensor,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    """Standardise poses using dataset-level mean and std.

    Args:
        poses: ``(T, K, 3)`` tensor.
        mean: ``(K, 3)`` mean tensor.
        std: ``(K, 3)`` standard deviation tensor.

    Returns:
        Standardised ``(T, K, 3)`` tensor.
    """
    return (poses - mean) / std


def destandardize_poses(
    poses: Tensor,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    """Invert :func:`standardize_poses`.

    Args:
        poses: ``(T, K, 3)`` standardised tensor.
        mean: ``(K, 3)`` mean tensor.
        std: ``(K, 3)`` standard deviation tensor.

    Returns:
        ``(T, K, 3)`` tensor in original scale.
    """
    return poses * std + mean
