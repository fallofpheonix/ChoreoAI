"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from choreoai.evaluate import (
    compute_fmd,
    retrieval_accuracy,
    average_joint_position_error,
)
from tests.conftest import LATENT_DIM, BATCH_SIZE, NUM_JOINTS, SEQ_LEN


class TestRetrievalAccuracy:
    def test_perfect_retrieval(self) -> None:
        z = F.normalize(torch.randn(10, LATENT_DIM), dim=-1)
        result = retrieval_accuracy(z, z, top_k=[1, 5])
        assert result["R@1"] == pytest.approx(1.0)
        assert result["R@5"] == pytest.approx(1.0)

    def test_random_retrieval_below_perfect(self) -> None:
        torch.manual_seed(99)
        q = F.normalize(torch.randn(50, LATENT_DIM), dim=-1)
        g = F.normalize(torch.randn(50, LATENT_DIM), dim=-1)
        result = retrieval_accuracy(q, g, top_k=1)
        # Random retrieval should rarely get R@1=1.0
        assert result["R@1"] < 1.0

    def test_int_top_k(self) -> None:
        z = F.normalize(torch.randn(10, LATENT_DIM), dim=-1)
        result = retrieval_accuracy(z, z, top_k=3)
        assert "R@3" in result

    def test_output_in_range(self) -> None:
        q = torch.randn(20, LATENT_DIM)
        g = torch.randn(20, LATENT_DIM)
        result = retrieval_accuracy(q, g, top_k=[1, 5, 10])
        for v in result.values():
            assert 0.0 <= v <= 1.0


class TestAverageJointPositionError:
    def test_zero_error(self) -> None:
        poses = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_JOINTS, 3)
        err = average_joint_position_error(poses, poses)
        assert err == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_error(self) -> None:
        pred = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_JOINTS, 3)
        target = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_JOINTS, 3)
        err = average_joint_position_error(pred, target)
        assert err > 0.0

    def test_with_mask(self) -> None:
        pred = torch.ones(2, 10, NUM_JOINTS, 3)
        target = torch.zeros(2, 10, NUM_JOINTS, 3)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, :5] = True  # only first 5 frames valid

        err_masked = average_joint_position_error(pred, target, mask=mask)
        err_full = average_joint_position_error(pred, target)
        # Both should be sqrt(3) (L2 of [1,1,1] - [0,0,0])
        expected = 3.0 ** 0.5
        assert err_masked == pytest.approx(expected, abs=1e-5)
        assert err_full == pytest.approx(expected, abs=1e-5)


class TestComputeFMD:
    def test_same_distribution_near_zero(self) -> None:
        torch.manual_seed(0)
        feats = torch.randn(100, 32)
        fmd = compute_fmd(feats, feats)
        assert fmd < 1e-3

    def test_different_distributions_positive(self) -> None:
        real = torch.randn(100, 32)
        gen = torch.randn(100, 32) + 5.0  # shifted distribution
        fmd = compute_fmd(real, gen)
        assert fmd > 0.0
