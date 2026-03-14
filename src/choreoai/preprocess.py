from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np

from choreoai.dataset import DatasetIndex, load_pose_array


@dataclass(frozen=True)
class PreprocessConfig:
    smooth_window: int = 5
    center_joint: int = 0
    scale_epsilon: float = 1e-6


def interpolate_missing(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, dtype=np.float64, copy=True)
    frames, joints, dims = out.shape

    for joint_idx in range(joints):
        for dim_idx in range(dims):
            values = out[:, joint_idx, dim_idx]
            valid = np.isfinite(values)

            if valid.all():
                continue
            if not valid.any():
                values[:] = 0.0
                continue

            valid_idx = np.flatnonzero(valid)
            missing_idx = np.flatnonzero(~valid)
            values[missing_idx] = np.interp(missing_idx, valid_idx, values[valid_idx])

    return out


def smooth_pose_sequence(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.array(arr, copy=True)
    if window % 2 == 0:
        raise ValueError(f"smooth_window must be odd, got {window}")

    pad = window // 2
    padded = np.pad(arr, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    out = np.empty_like(arr, dtype=np.float64)

    for idx in range(arr.shape[0]):
        out[idx] = padded[idx : idx + window].mean(axis=0)

    return out


def normalize_pose_sequence(
    arr: np.ndarray,
    center_joint: int,
    scale_epsilon: float,
) -> np.ndarray:
    if center_joint < 0 or center_joint >= arr.shape[1]:
        raise IndexError(f"center_joint out of bounds: {center_joint}")

    centered = arr - arr[:, center_joint : center_joint + 1, :]
    radii = np.linalg.norm(centered, axis=2)
    mean_radius = float(np.mean(radii))
    scale = mean_radius if mean_radius > scale_epsilon else 1.0
    normalized = centered / scale
    return normalized.astype(np.float32, copy=False)


def preprocess_pose_sequence(arr: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    repaired = interpolate_missing(arr)
    smoothed = smooth_pose_sequence(repaired, config.smooth_window)
    return normalize_pose_sequence(smoothed, config.center_joint, config.scale_epsilon)


def preprocess_dataset(
    input_root: Path,
    output_root: Path,
    config: PreprocessConfig,
    force: bool = False,
) -> list[Path]:
    created: list[Path] = []

    for example in DatasetIndex(input_root).sequences():
        seq_dir = output_root / example.seq_id
        if seq_dir.exists() and not force:
            raise FileExistsError(f"sequence already exists: {seq_dir}")

        seq_dir.mkdir(parents=True, exist_ok=True)
        arr = load_pose_array(example.poses_path, allow_nonfinite=True)
        processed = preprocess_pose_sequence(arr, config)
        np.save(seq_dir / "poses.npy", processed)

        for maybe_path in (example.text_path, example.image_path, example.audio_path):
            if maybe_path is not None:
                shutil.copy2(maybe_path, seq_dir / maybe_path.name)

        created.append(seq_dir)

    return created
