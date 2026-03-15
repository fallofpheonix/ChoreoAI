"""Tests for preprocessing utilities."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from choreoai.preprocess import (
    center_poses,
    scale_poses,
    normalize_poses,
    random_rotation_y,
    temporal_jitter,
    mirror_poses,
    compute_velocity,
    compute_acceleration,
    append_velocity,
    compute_mean_std,
    standardize_poses,
    destandardize_poses,
)
from tests.conftest import NUM_JOINTS, SEQ_LEN


@pytest.fixture
def poses() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(SEQ_LEN, NUM_JOINTS, 3)


class TestCenterPoses:
    def test_shape_preserved(self, poses: Tensor) -> None:
        out = center_poses(poses)
        assert out.shape == poses.shape

    def test_hip_at_origin(self, poses: Tensor) -> None:
        out = center_poses(poses, hip_joint=0)
        # After centering, hip joint should be (approximately) at origin for each frame
        assert out[:, 0, :].abs().max().item() < 1e-5


class TestScalePoses:
    def test_shape_preserved(self, poses: Tensor) -> None:
        out = scale_poses(poses, target_scale=1.0)
        assert out.shape == poses.shape

    def test_zero_poses_safe(self) -> None:
        zero = torch.zeros(10, NUM_JOINTS, 3)
        out = scale_poses(zero)
        assert (out == 0).all()


class TestNormalizePoses:
    def test_shape(self, poses: Tensor) -> None:
        out = normalize_poses(poses)
        assert out.shape == poses.shape


class TestRandomRotationY:
    def test_shape_preserved(self, poses: Tensor) -> None:
        out = random_rotation_y(poses)
        assert out.shape == poses.shape

    def test_preserves_norm(self, poses: Tensor) -> None:
        """Rotation should preserve Euclidean distances."""
        out = random_rotation_y(poses, max_angle_deg=45.0)
        # Norms should be approximately equal (up to floating-point)
        orig_norms = poses.norm(dim=-1)
        out_norms = out.norm(dim=-1)
        assert torch.allclose(orig_norms, out_norms, atol=1e-5)


class TestTemporalJitter:
    def test_shape_preserved(self, poses: Tensor) -> None:
        out = temporal_jitter(poses, max_shift=5)
        assert out.shape == poses.shape

    def test_zero_shift_returns_same(self, poses: Tensor) -> None:
        # max_shift=0 must always return identical tensor
        out = temporal_jitter(poses, max_shift=0)
        assert torch.equal(out, poses)


class TestMirrorPoses:
    def test_x_flipped(self, poses: Tensor) -> None:
        out = mirror_poses(poses, mirror_x=True)
        assert torch.allclose(out[..., 0], -poses[..., 0])
        assert torch.allclose(out[..., 1], poses[..., 1])

    def test_no_mirror(self, poses: Tensor) -> None:
        out = mirror_poses(poses, mirror_x=False)
        assert torch.equal(out, poses)


class TestVelocityFeatures:
    def test_velocity_shape(self, poses: Tensor) -> None:
        vel = compute_velocity(poses)
        assert vel.shape == (SEQ_LEN - 1, NUM_JOINTS, 3)

    def test_acceleration_shape(self, poses: Tensor) -> None:
        acc = compute_acceleration(poses)
        assert acc.shape == (SEQ_LEN - 2, NUM_JOINTS, 3)

    def test_append_velocity_shape(self, poses: Tensor) -> None:
        out = append_velocity(poses)
        assert out.shape == (SEQ_LEN, NUM_JOINTS, 6)

    def test_first_frame_velocity_zero(self, poses: Tensor) -> None:
        out = append_velocity(poses)
        # First frame velocity (last 3 dims) should be zeros
        assert (out[0, :, 3:] == 0).all()


class TestStatistics:
    def test_compute_mean_std_shape(self, poses: Tensor) -> None:
        mean, std = compute_mean_std([poses, poses])
        assert mean.shape == (NUM_JOINTS, 3)
        assert std.shape == (NUM_JOINTS, 3)

    def test_std_positive(self, poses: Tensor) -> None:
        _, std = compute_mean_std([poses, poses + 1.0])
        assert (std > 0).all()

    def test_standardize_roundtrip(self, poses: Tensor) -> None:
        mean, std = compute_mean_std([poses])
        standardized = standardize_poses(poses, mean, std)
        recovered = destandardize_poses(standardized, mean, std)
        assert torch.allclose(recovered, poses, atol=1e-5)
