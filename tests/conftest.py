"""Shared pytest fixtures for ChoreoAI tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_JOINTS = 17
SEQ_LEN = 30
LATENT_DIM = 64
BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with synthetic pose samples."""
    rng = np.random.default_rng(42)
    for i in range(6):
        sample_dir = tmp_path / f"sample_{i:03d}"
        sample_dir.mkdir()
        # Pose file: (T, K, 3)
        poses = rng.standard_normal((SEQ_LEN, NUM_JOINTS, 3)).astype(np.float32)
        np.save(sample_dir / "poses.npy", poses)

        # Text tokens: (L,) integer ids
        tokens = torch.randint(0, 50265, (32,))
        torch.save(tokens, sample_dir / "tokens.pt")

        # Image tensor: (3, 224, 224)
        image = torch.randn(3, 224, 224)
        torch.save(image, sample_dir / "image.pt")

        # Audio spectrogram: (80, 128)
        audio = torch.randn(80, 128)
        torch.save(audio, sample_dir / "audio.pt")

    return tmp_path


@pytest.fixture
def manifest_path(tmp_data_dir: Path) -> Path:
    """Write a manifest JSON for the temporary data directory."""
    entries = []
    for sample_dir in sorted(tmp_data_dir.glob("sample_*")):
        rel = sample_dir.relative_to(tmp_data_dir)
        entries.append(
            {
                "path": str(rel),
                "text": str(rel / "tokens.pt"),
                "image": str(rel / "image.pt"),
                "audio": str(rel / "audio.pt"),
            }
        )
    manifest = tmp_data_dir / "manifest.json"
    with manifest.open("w") as fh:
        json.dump(entries, fh)
    return manifest


@pytest.fixture
def random_poses() -> torch.Tensor:
    """Return a batch of random pose tensors (B, T, K, 3)."""
    return torch.randn(BATCH_SIZE, SEQ_LEN, NUM_JOINTS, 3)


@pytest.fixture
def random_pose_single() -> torch.Tensor:
    """Return a single random pose tensor (T, K, 3)."""
    return torch.randn(SEQ_LEN, NUM_JOINTS, 3)
