"""Tests for encoder modules."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from choreoai.encoders.motion_encoder import MotionEncoder, SinusoidalPositionalEncoding
from choreoai.encoders.audio_encoder import AudioEncoder

from tests.conftest import NUM_JOINTS, SEQ_LEN, LATENT_DIM, BATCH_SIZE


# ---------------------------------------------------------------------------
# Motion Encoder
# ---------------------------------------------------------------------------


class TestMotionEncoder:
    @pytest.fixture
    def encoder(self) -> MotionEncoder:
        return MotionEncoder(
            num_joints=NUM_JOINTS,
            d_model=64,
            nhead=4,
            num_layers=2,
            latent_dim=LATENT_DIM,
        )

    def test_output_shape(self, encoder: MotionEncoder, random_poses: Tensor) -> None:
        z = encoder(random_poses)
        assert z.shape == (BATCH_SIZE, LATENT_DIM)

    def test_output_dtype(self, encoder: MotionEncoder, random_poses: Tensor) -> None:
        z = encoder(random_poses)
        assert z.dtype == torch.float32

    def test_with_padding_mask(self, encoder: MotionEncoder, random_poses: Tensor) -> None:
        # Create a mask: last 5 frames are padded
        mask = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)
        mask[:, -5:] = True  # True = padded

        z = encoder(random_poses, key_padding_mask=mask)
        assert z.shape == (BATCH_SIZE, LATENT_DIM)

    def test_all_frames_masked_safe(self, encoder: MotionEncoder) -> None:
        """When all but one frame are masked, forward should not crash."""
        poses = torch.randn(2, 5, NUM_JOINTS, 3)
        mask = torch.ones(2, 5, dtype=torch.bool)
        mask[:, 0] = False  # only first frame is valid
        z = encoder(poses, key_padding_mask=mask)
        assert z.shape == (2, LATENT_DIM)
        assert torch.isfinite(z).all()

    def test_no_nan(self, encoder: MotionEncoder, random_poses: Tensor) -> None:
        z = encoder(random_poses)
        assert torch.isfinite(z).all()

    def test_gradients_flow(self, encoder: MotionEncoder, random_poses: Tensor) -> None:
        random_poses.requires_grad_(False)
        z = encoder(random_poses)
        loss = z.sum()
        loss.backward()
        # At least one parameter should have a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in encoder.parameters()
        )
        assert has_grad


class TestSinusoidalPositionalEncoding:
    def test_output_shape(self) -> None:
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=256)
        x = torch.randn(4, 32, 64)
        out = pe(x)
        assert out.shape == x.shape

    def test_deterministic(self) -> None:
        pe = SinusoidalPositionalEncoding(d_model=64)
        pe.eval()  # disable dropout for deterministic output
        x = torch.randn(2, 10, 64)
        assert torch.equal(pe(x.clone()), pe(x.clone()))


# ---------------------------------------------------------------------------
# Audio Encoder
# ---------------------------------------------------------------------------


class TestAudioEncoder:
    @pytest.fixture
    def encoder(self) -> AudioEncoder:
        return AudioEncoder(
            n_mels=32,
            d_model=64,
            nhead=4,
            num_layers=2,
            latent_dim=LATENT_DIM,
        )

    def test_output_shape(self, encoder: AudioEncoder) -> None:
        spec = torch.randn(BATCH_SIZE, 32, 50)  # (B, F, T_a)
        z = encoder(spec)
        assert z.shape == (BATCH_SIZE, LATENT_DIM)

    def test_no_nan(self, encoder: AudioEncoder) -> None:
        spec = torch.randn(2, 32, 50)
        z = encoder(spec)
        assert torch.isfinite(z).all()

    def test_variable_time_dim(self, encoder: AudioEncoder) -> None:
        """Encoder should accept variable temporal lengths."""
        for t in [30, 60, 100]:
            spec = torch.randn(2, 32, t)
            z = encoder(spec)
            assert z.shape == (2, LATENT_DIM)
