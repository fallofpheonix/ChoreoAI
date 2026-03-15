"""Tests for InfoNCE and multi-modal alignment losses."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from choreoai.losses import InfoNCELoss, infonce_loss, MultiModalAlignmentLoss
from tests.conftest import LATENT_DIM, BATCH_SIZE


# ---------------------------------------------------------------------------
# InfoNCELoss (module)
# ---------------------------------------------------------------------------


class TestInfoNCELoss:
    @pytest.fixture
    def loss_fn(self) -> InfoNCELoss:
        return InfoNCELoss(temperature=0.07)

    @pytest.fixture
    def embeddings(self) -> tuple[Tensor, Tensor]:
        torch.manual_seed(42)
        z1 = F.normalize(torch.randn(BATCH_SIZE, LATENT_DIM), dim=-1)
        z2 = F.normalize(torch.randn(BATCH_SIZE, LATENT_DIM), dim=-1)
        return z1, z2

    def test_scalar_output(self, loss_fn: InfoNCELoss, embeddings: tuple) -> None:
        z1, z2 = embeddings
        loss = loss_fn(z1, z2)
        assert loss.ndim == 0

    def test_non_negative(self, loss_fn: InfoNCELoss, embeddings: tuple) -> None:
        z1, z2 = embeddings
        loss = loss_fn(z1, z2)
        assert loss.item() >= 0.0

    def test_perfect_alignment_lower_than_random(self, loss_fn: InfoNCELoss) -> None:
        """Perfectly aligned pairs (z1 == z2) should give lower loss than random."""
        torch.manual_seed(7)
        z = F.normalize(torch.randn(BATCH_SIZE, LATENT_DIM), dim=-1)
        z_random = F.normalize(torch.randn(BATCH_SIZE, LATENT_DIM), dim=-1)

        loss_aligned = loss_fn(z, z)
        loss_random = loss_fn(z, z_random)
        assert loss_aligned.item() < loss_random.item()

    def test_symmetric(self, loss_fn: InfoNCELoss, embeddings: tuple) -> None:
        z1, z2 = embeddings
        loss_12 = loss_fn(z1, z2)
        loss_21 = loss_fn(z2, z1)
        assert abs(loss_12.item() - loss_21.item()) < 1e-5

    def test_shape_mismatch_raises(self, loss_fn: InfoNCELoss) -> None:
        z1 = torch.randn(4, LATENT_DIM)
        z2 = torch.randn(4, LATENT_DIM + 10)
        with pytest.raises(AssertionError):
            loss_fn(z1, z2)

    def test_gradients_flow(self, loss_fn: InfoNCELoss) -> None:
        z1 = torch.randn(BATCH_SIZE, LATENT_DIM, requires_grad=True)
        z2 = torch.randn(BATCH_SIZE, LATENT_DIM, requires_grad=True)
        loss = loss_fn(z1, z2)
        loss.backward()
        assert z1.grad is not None
        assert z2.grad is not None

    def test_learnable_temperature(self) -> None:
        loss_fn = InfoNCELoss(temperature=0.07, learn_temperature=True)
        assert any(p.requires_grad for p in loss_fn.parameters())

    def test_fixed_temperature(self) -> None:
        loss_fn = InfoNCELoss(temperature=0.07, learn_temperature=False)
        assert not any(p.requires_grad for p in loss_fn.parameters())


# ---------------------------------------------------------------------------
# infonce_loss (functional)
# ---------------------------------------------------------------------------


class TestInfoNCELossFunctional:
    def test_matches_module(self) -> None:
        torch.manual_seed(0)
        z1 = torch.randn(BATCH_SIZE, LATENT_DIM)
        z2 = torch.randn(BATCH_SIZE, LATENT_DIM)
        temperature = 0.07

        loss_module = InfoNCELoss(temperature=temperature, learn_temperature=False)
        loss_fn_val = infonce_loss(z1, z2, temperature=temperature)
        loss_mod_val = loss_module(z1, z2)

        assert abs(loss_fn_val.item() - loss_mod_val.item()) < 1e-5


# ---------------------------------------------------------------------------
# MultiModalAlignmentLoss
# ---------------------------------------------------------------------------


class TestMultiModalAlignmentLoss:
    @pytest.fixture
    def loss_fn(self) -> MultiModalAlignmentLoss:
        return MultiModalAlignmentLoss(temperature=0.07)

    def test_single_modality(self, loss_fn: MultiModalAlignmentLoss) -> None:
        motion_z = torch.randn(BATCH_SIZE, LATENT_DIM)
        text_z = torch.randn(BATCH_SIZE, LATENT_DIM)
        loss = loss_fn(motion_z, {"text": text_z, "image": None, "audio": None})
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_all_none_returns_zero(self, loss_fn: MultiModalAlignmentLoss) -> None:
        motion_z = torch.randn(BATCH_SIZE, LATENT_DIM)
        loss = loss_fn(motion_z, {"text": None, "image": None, "audio": None})
        assert loss.item() == 0.0

    def test_multiple_modalities(self, loss_fn: MultiModalAlignmentLoss) -> None:
        motion_z = torch.randn(BATCH_SIZE, LATENT_DIM)
        modalities = {
            "text": torch.randn(BATCH_SIZE, LATENT_DIM),
            "image": torch.randn(BATCH_SIZE, LATENT_DIM),
            "audio": None,
        }
        loss = loss_fn(motion_z, modalities)
        assert torch.isfinite(loss)

    def test_gradients_flow(self, loss_fn: MultiModalAlignmentLoss) -> None:
        motion_z = torch.randn(BATCH_SIZE, LATENT_DIM, requires_grad=True)
        text_z = torch.randn(BATCH_SIZE, LATENT_DIM, requires_grad=True)
        loss = loss_fn(motion_z, {"text": text_z})
        loss.backward()
        assert motion_z.grad is not None
