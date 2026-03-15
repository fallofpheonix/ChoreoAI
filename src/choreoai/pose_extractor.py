"""
pose_extractor.py — Video-to-skeleton pose extraction pipeline.

Converts a mono-camera video into a sequence of ``(T, K, 3)`` normalised
skeleton tensors.  Two backends are supported:

* **MediaPipe** (default) — CPU-friendly, ships as ``mediapipe`` Python package.
* **Stub** — deterministic random poses used for unit testing when neither
  backend is installed.

The actual 3D estimation from a monocular video is an inherently ill-posed
problem.  MediaPipe Pose returns 33 world-space 3-D landmarks in metres;
this module selects a configurable 17-joint COCO-style subset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Joint mapping: MediaPipe (33 landmarks) → COCO 17-joint subset
# ---------------------------------------------------------------------------
# Indices into MediaPipe's 33-landmark pose:
# https://google.github.io/mediapipe/solutions/pose.html

_MEDIAPIPE_TO_COCO17: list[int] = [
    0,   # 0  nose
    2,   # 1  left_eye
    5,   # 2  right_eye
    7,   # 3  left_ear
    8,   # 4  right_ear
    11,  # 5  left_shoulder
    12,  # 6  right_shoulder
    13,  # 7  left_elbow
    14,  # 8  right_elbow
    15,  # 9  left_wrist
    16,  # 10 right_wrist
    23,  # 11 left_hip
    24,  # 12 right_hip
    25,  # 13 left_knee
    26,  # 14 right_knee
    27,  # 15 left_ankle
    28,  # 16 right_ankle
]

NUM_JOINTS = len(_MEDIAPIPE_TO_COCO17)  # 17


# ---------------------------------------------------------------------------
# MediaPipe backend
# ---------------------------------------------------------------------------


def _extract_with_mediapipe(
    video_path: Path,
    visibility_threshold: float = 0.5,
) -> np.ndarray:
    """Extract poses from *video_path* using MediaPipe Pose.

    Args:
        video_path: Path to an MP4 / AVI video file.
        visibility_threshold: Landmarks with lower visibility are zeroed.

    Returns:
        Float32 numpy array of shape ``(T, 17, 3)``.

    Raises:
        ImportError: If ``mediapipe`` or ``cv2`` are not installed.
        FileNotFoundError: If *video_path* does not exist.
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "MediaPipe backend requires `mediapipe` and `opencv-python`. "
            "Install them with: pip install mediapipe opencv-python"
        ) from exc

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    mp_pose = mp.solutions.pose  # type: ignore[attr-defined]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frames: list[np.ndarray] = []  # each element: (17, 3)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_world_landmarks is None:
                # No person detected — repeat last frame or use zeros
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((NUM_JOINTS, 3), dtype=np.float32))
            else:
                lm = results.pose_world_landmarks.landmark
                frame_joints = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
                for out_idx, mp_idx in enumerate(_MEDIAPIPE_TO_COCO17):
                    landmark = lm[mp_idx]
                    vis = getattr(landmark, "visibility", 1.0)
                    if vis >= visibility_threshold:
                        frame_joints[out_idx] = [
                            landmark.x,
                            landmark.y,
                            landmark.z,
                        ]
                frames.append(frame_joints)

    cap.release()
    if not frames:
        raise ValueError(f"No frames extracted from {video_path}")

    return np.stack(frames, axis=0)  # (T, 17, 3)


# ---------------------------------------------------------------------------
# Stub backend (testing / no-GPU)
# ---------------------------------------------------------------------------


def _extract_stub(
    num_frames: int = 60,
    seed: int = 0,
    *,
    num_joints: int = NUM_JOINTS,
) -> np.ndarray:
    """Generate deterministic random poses for testing.

    Args:
        num_frames: Number of frames to generate.
        seed: Random seed for reproducibility.
        num_joints: Number of skeleton joints.

    Returns:
        Float32 numpy array of shape ``(num_frames, num_joints, 3)``.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_frames, num_joints, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_poses(
    video_path: str | Path,
    *,
    backend: str = "mediapipe",
    visibility_threshold: float = 0.5,
    stub_frames: int = 60,
    stub_seed: int = 0,
) -> Tensor:
    """Extract a ``(T, 17, 3)`` skeleton tensor from *video_path*.

    Args:
        video_path: Path to the source video. Not used when
            *backend* is ``"stub"``.
        backend: One of ``"mediapipe"`` or ``"stub"``.
        visibility_threshold: MediaPipe visibility threshold (ignored for
            stub backend).
        stub_frames: Number of synthetic frames (stub only).
        stub_seed: RNG seed for the stub backend.

    Returns:
        Float32 tensor of shape ``(T, K, 3)``.
    """
    if backend == "stub":
        arr = _extract_stub(num_frames=stub_frames, seed=stub_seed)
    elif backend == "mediapipe":
        arr = _extract_with_mediapipe(
            Path(video_path),
            visibility_threshold=visibility_threshold,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Choose 'mediapipe' or 'stub'.")

    return torch.from_numpy(arr)


def save_poses(poses: Tensor | np.ndarray, out_path: str | Path) -> None:
    """Save a pose tensor to a ``.npy`` file.

    Args:
        poses: ``(T, K, 3)`` tensor or numpy array.
        out_path: Destination file path (e.g. ``sample_01/poses.npy``).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(poses, Tensor):
        arr = poses.cpu().numpy()
    else:
        arr = np.asarray(poses, dtype=np.float32)
    np.save(out_path, arr)
    logger.info("Saved poses %s → %s", arr.shape, out_path)
