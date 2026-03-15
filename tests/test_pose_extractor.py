"""Tests for pose extraction utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from choreoai.pose_extractor import extract_poses, save_poses, NUM_JOINTS


class TestExtractPosesStub:
    def test_output_shape(self) -> None:
        poses = extract_poses("", backend="stub", stub_frames=60)
        assert poses.shape == (60, NUM_JOINTS, 3)

    def test_output_dtype(self) -> None:
        poses = extract_poses("", backend="stub")
        assert poses.dtype == torch.float32

    def test_deterministic_with_seed(self) -> None:
        p1 = extract_poses("", backend="stub", stub_seed=42)
        p2 = extract_poses("", backend="stub", stub_seed=42)
        assert torch.equal(p1, p2)

    def test_different_seeds_differ(self) -> None:
        p1 = extract_poses("", backend="stub", stub_seed=1)
        p2 = extract_poses("", backend="stub", stub_seed=2)
        assert not torch.equal(p1, p2)

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            extract_poses("video.mp4", backend="bad_backend")


class TestSavePoses:
    def test_saves_npy(self, tmp_path) -> None:
        poses = extract_poses("", backend="stub", stub_frames=20)
        out = tmp_path / "sample" / "poses.npy"
        save_poses(poses, out)
        assert out.exists()
        loaded = np.load(out)
        assert loaded.shape == (20, NUM_JOINTS, 3)

    def test_saves_numpy_array(self, tmp_path) -> None:
        arr = np.zeros((15, NUM_JOINTS, 3), dtype=np.float32)
        out = tmp_path / "poses.npy"
        save_poses(arr, out)
        loaded = np.load(out)
        assert loaded.shape == arr.shape
