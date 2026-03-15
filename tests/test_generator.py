"""Tests for diffusion generator components."""

from __future__ import annotations

import pytest
import torch

from choreoai.generator.denoising_transformer import (
    DenoisingTransformer,
    SinusoidalTimestepEmbedding,
)
from choreoai.generator.diffusion_scheduler import DDPMScheduler
from tests.conftest import NUM_JOINTS, SEQ_LEN, LATENT_DIM, BATCH_SIZE


# ---------------------------------------------------------------------------
# SinusoidalTimestepEmbedding
# ---------------------------------------------------------------------------


class TestSinusoidalTimestepEmbedding:
    def test_output_shape(self) -> None:
        emb = SinusoidalTimestepEmbedding(d_model=64)
        t = torch.randint(0, 1000, (BATCH_SIZE,))
        out = emb(t)
        assert out.shape == (BATCH_SIZE, 64)

    def test_no_nan(self) -> None:
        emb = SinusoidalTimestepEmbedding(d_model=64)
        t = torch.randint(0, 1000, (BATCH_SIZE,))
        out = emb(t)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# DenoisingTransformer
# ---------------------------------------------------------------------------


class TestDenoisingTransformer:
    @pytest.fixture
    def model(self) -> DenoisingTransformer:
        return DenoisingTransformer(
            num_joints=NUM_JOINTS,
            d_model=64,
            nhead=4,
            num_layers=2,
            latent_dim=LATENT_DIM,
            max_seq_len=SEQ_LEN + 10,
            max_timesteps=1000,
        )

    def test_output_shape(self, model: DenoisingTransformer) -> None:
        noisy = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_JOINTS, 3)
        t = torch.randint(0, 1000, (BATCH_SIZE,))
        z = torch.randn(BATCH_SIZE, LATENT_DIM)
        out = model(noisy, t, z)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, NUM_JOINTS, 3)

    def test_no_nan(self, model: DenoisingTransformer) -> None:
        noisy = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_JOINTS, 3)
        t = torch.randint(0, 1000, (BATCH_SIZE,))
        z = torch.randn(BATCH_SIZE, LATENT_DIM)
        out = model(noisy, t, z)
        assert torch.isfinite(out).all()

    def test_with_padding_mask(self, model: DenoisingTransformer) -> None:
        noisy = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_JOINTS, 3)
        t = torch.randint(0, 1000, (BATCH_SIZE,))
        z = torch.randn(BATCH_SIZE, LATENT_DIM)
        mask = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)
        mask[:, -5:] = True  # last 5 frames padded
        out = model(noisy, t, z, key_padding_mask=mask)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, NUM_JOINTS, 3)

    def test_gradients_flow(self, model: DenoisingTransformer) -> None:
        noisy = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_JOINTS, 3)
        t = torch.randint(0, 1000, (BATCH_SIZE,))
        z = torch.randn(BATCH_SIZE, LATENT_DIM, requires_grad=True)
        out = model(noisy, t, z)
        out.sum().backward()
        assert z.grad is not None


# ---------------------------------------------------------------------------
# DDPMScheduler
# ---------------------------------------------------------------------------


class TestDDPMScheduler:
    @pytest.fixture
    def scheduler(self) -> DDPMScheduler:
        return DDPMScheduler(num_timesteps=10, beta_start=1e-4, beta_end=0.02)

    def test_buffer_shapes(self, scheduler: DDPMScheduler) -> None:
        assert scheduler.betas.shape == (10,)
        assert scheduler.alphas_cumprod.shape == (10,)
        assert scheduler.sqrt_alphas_cumprod.shape == (10,)

    def test_betas_in_range(self, scheduler: DDPMScheduler) -> None:
        assert (scheduler.betas >= 0).all()
        assert (scheduler.betas <= 1).all()

    def test_forward_diffusion_shape(self, scheduler: DDPMScheduler) -> None:
        x0 = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_JOINTS, 3)
        t = torch.randint(0, 10, (BATCH_SIZE,))
        x_t, noise = scheduler.forward_diffusion(x0, t)
        assert x_t.shape == x0.shape
        assert noise.shape == x0.shape

    def test_forward_diffusion_t0_close_to_clean(self, scheduler: DDPMScheduler) -> None:
        """At t=0 with very small beta, x_t should be close to x_0."""
        x0 = torch.randn(2, 5, NUM_JOINTS, 3)
        t = torch.zeros(2, dtype=torch.long)
        x_t, noise = scheduler.forward_diffusion(x0, t)
        # x_t ≈ sqrt(ᾱ_0) * x_0 + small noise
        assert torch.allclose(x_t, scheduler.sqrt_alphas_cumprod[0] * x0 + scheduler.sqrt_one_minus_alphas_cumprod[0] * noise, atol=1e-5)

    def test_reverse_step_shape(self, scheduler: DDPMScheduler) -> None:
        x_t = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_JOINTS, 3)
        noise_pred = torch.randn_like(x_t)
        t = torch.randint(1, 10, (BATCH_SIZE,))
        x_prev = scheduler.reverse_step(noise_pred, t, x_t)
        assert x_prev.shape == x_t.shape

    def test_cosine_schedule(self) -> None:
        sched = DDPMScheduler(num_timesteps=10, schedule="cosine")
        assert (sched.betas > 0).all()
        assert (sched.betas < 1).all()

    def test_invalid_schedule_raises(self) -> None:
        with pytest.raises(ValueError):
            DDPMScheduler(schedule="unknown")

    def test_sample_shape(self) -> None:
        """Full reverse sampling loop should produce correct shape."""
        scheduler = DDPMScheduler(num_timesteps=5)
        model = DenoisingTransformer(
            num_joints=NUM_JOINTS,
            d_model=32,
            nhead=4,
            num_layers=1,
            latent_dim=LATENT_DIM,
            max_seq_len=SEQ_LEN + 10,
            max_timesteps=5,
        )
        z = torch.randn(2, LATENT_DIM)
        shape = (2, SEQ_LEN, NUM_JOINTS, 3)
        output = scheduler.sample(model, shape, conditioning=z, device="cpu")
        assert output.shape == shape
