"""
preprocess_np.py — NumPy-based pose preprocessing utilities.

Provides functions for interpolating missing values, smoothing,
and normalizing raw pose sequences stored as NumPy arrays.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from choreoai.dataset_index import DatasetIndex, load_pose_array


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for pose preprocessing."""

    smooth_window: int = 5
    center_joint: int = 0
    scale_epsilon: float = 1e-6


def interpolate_missing(arr: np.ndarray) -> np.ndarray:
    """Interpolate missing (NaN/Inf) values in a pose array.

    Args:
        arr: Pose array with shape (T, K, 3).

    Returns:
        Array with missing values interpolated.
    """
    out = np.array(arr, dtype=np.float64, copy=True)
    mask = np.isfinite(out)
    
    if mask.all():
        return out
    if not mask.any():
        return np.zeros_like(out)
        
    try:
        import pandas as pd
        frames = out.shape[0]
        flattened = out.reshape(frames, -1)
        df = pd.DataFrame(flattened)
        df.interpolate(method='linear', axis=0, limit_direction='both', inplace=True)
        df.fillna(0.0, inplace=True)
        return df.values.reshape(out.shape)
    except ImportError:
        frames = out.shape[0]
        out_flat = out.reshape(frames, -1)
        for i in range(out_flat.shape[1]):
            col = out_flat[:, i]
            valid = np.isfinite(col)
            if valid.all():
                continue
            if not valid.any():
                col[:] = 0.0
                continue
            valid_idx = np.flatnonzero(valid)
            missing_idx = np.flatnonzero(~valid)
            col[missing_idx] = np.interp(missing_idx, valid_idx, col[valid_idx])
            
        return out_flat.reshape(out.shape)


def smooth_pose_sequence(arr: np.ndarray, window: int) -> np.ndarray:
    """Apply moving average smoothing to a pose sequence.

    Args:
        arr: Pose array with shape (T, K, 3).
        window: Size of the smoothing window (must be odd).

    Returns:
        Smoothed pose array.

    Raises:
        ValueError: If window is even.
    """
    if window <= 1:
        return np.array(arr, copy=True)
    if window % 2 == 0:
        raise ValueError(f"smooth_window must be odd, got {window}")

    # Use a uniform kernel for moving average
    kernel = np.ones(window) / window
    
    # We want to smooth along the temporal axis (0) for each joint and dimension.
    # We can reshape the array to (T, K*3) and apply 1D convolution to each column.
    T, K, D = arr.shape
    flattened = arr.reshape(T, -1)
    
    # Padding corresponds to 'edge' mode in the original loop
    # We can manually pad and use 'valid' convolution mode or use scipy.ndimage.uniform_filter1d
    try:
        from scipy.ndimage import uniform_filter1d
        smoothed_flat = uniform_filter1d(flattened, size=window, axis=0, mode='nearest')
    except ImportError:
        # Fallback to numpy convolution if scipy is missing
        pad = window // 2
        padded = np.pad(flattened, ((pad, pad), (0, 0)), mode="edge")
        smoothed_flat = np.zeros_like(flattened)
        for i in range(flattened.shape[1]):
            smoothed_flat[:, i] = np.convolve(padded[:, i], kernel, mode='valid')
            
    return smoothed_flat.reshape(T, K, D)


def normalize_pose_sequence(
    arr: np.ndarray,
    center_joint: int,
    scale_epsilon: float,
) -> np.ndarray:
    """Center and scale a pose sequence.

    Args:
        arr: Pose array with shape (T, K, 3).
        center_joint: Index of the joint to use as the origin.
        scale_epsilon: Minimum scale value to avoid division by zero.

    Returns:
        Normalized pose array with dtype float32.

    Raises:
        IndexError: If center_joint is out of bounds.
    """
    if center_joint < 0 or center_joint >= arr.shape[1]:
        raise IndexError(f"center_joint out of bounds: {center_joint}")

    centered = arr - arr[:, center_joint : center_joint + 1, :]
    radii = np.linalg.norm(centered, axis=2)
    mean_radius = float(np.mean(radii))
    scale = mean_radius if mean_radius > scale_epsilon else 1.0
    normalized = centered / scale
    return normalized.astype(np.float32, copy=False)


def preprocess_pose_sequence(arr: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Apply full preprocessing pipeline to a pose sequence.

    Applies interpolation, smoothing, and normalization in sequence.

    Args:
        arr: Raw pose array with shape (T, K, 3).
        config: Preprocessing configuration.

    Returns:
        Preprocessed pose array.
    """
    repaired = interpolate_missing(arr)
    smoothed = smooth_pose_sequence(repaired, config.smooth_window)
    return normalize_pose_sequence(smoothed, config.center_joint, config.scale_epsilon)


def preprocess_dataset(
    input_root: Path,
    output_root: Path,
    config: PreprocessConfig,
    force: bool = False,
) -> list[Path]:
    """Preprocess all sequences in a dataset.

    Args:
        input_root: Root directory of the input dataset.
        output_root: Root directory for the output dataset.
        config: Preprocessing configuration.
        force: If True, overwrite existing output directories.

    Returns:
        List of paths to created sequence directories.

    Raises:
        FileExistsError: If an output directory exists and force is False.
    """
    created: list[Path] = []

    for example in DatasetIndex(input_root).sequences():
        seq_dir = output_root / example.seq_id
        if seq_dir.exists() and not force:
            raise FileExistsError(f"sequence already exists: {seq_dir}")
        elif seq_dir.exists():
            shutil.rmtree(seq_dir)

        seq_dir.mkdir(parents=True, exist_ok=True)
        arr = load_pose_array(example.poses_path, allow_nonfinite=True)
        processed = preprocess_pose_sequence(arr, config)
        np.save(seq_dir / "poses.npy", processed)

        for maybe_path in (example.text_path, example.image_path, example.audio_path):
            if maybe_path is not None:
                shutil.copy(maybe_path, seq_dir / maybe_path.name)

        created.append(seq_dir)

    return created
