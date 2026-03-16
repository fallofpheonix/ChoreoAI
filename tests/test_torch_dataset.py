"""Tests for ChoreoDataset and collate_fn."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from choreoai.torch_dataset import ChoreoDataset, collate_fn, build_dataloader

# re-use shared constants
from tests.conftest import NUM_JOINTS, SEQ_LEN
from tests.conftest import NUM_JOINTS, SEQ_LEN, BATCH_SIZE


# ---------------------------------------------------------------------------
# ChoreoDataset
# ---------------------------------------------------------------------------


class TestChoreoDataset:
    def test_len(self, manifest_path: Path) -> None:
        ds = ChoreoDataset(manifest_path, max_seq_len=SEQ_LEN, num_joints=NUM_JOINTS)
        assert len(ds) == 6

    def test_item_shapes(self, manifest_path: Path) -> None:
        ds = ChoreoDataset(manifest_path, max_seq_len=SEQ_LEN, num_joints=NUM_JOINTS)
        item = ds[0]

        assert item["poses"].shape == (SEQ_LEN, NUM_JOINTS, 3)
        assert item["mask_pose"].shape == (SEQ_LEN,)
        assert item["mask_pose"].dtype == torch.bool
        # All frames are valid (sample seq_len == max_seq_len)
        assert item["mask_pose"].all()

    def test_padding(self, tmp_path: Path) -> None:
        """Test that shorter sequences are padded correctly."""
        import json

        sample_dir = tmp_path / "short"
        sample_dir.mkdir()
        short_len = 10
        poses = np.zeros((short_len, NUM_JOINTS, 3), dtype=np.float32)
        np.save(sample_dir / "poses.npy", poses)

        manifest = tmp_path / "m.json"
        with manifest.open("w") as fh:
            json.dump([{"path": "short"}], fh)

        ds = ChoreoDataset(manifest, max_seq_len=SEQ_LEN, num_joints=NUM_JOINTS)
        item = ds[0]

        assert item["poses"].shape == (SEQ_LEN, NUM_JOINTS, 3)
        assert item["mask_pose"][:short_len].all()
        assert not item["mask_pose"][short_len:].any()
        # Padded region should be zeros
        assert (item["poses"][short_len:] == 0).all()

    def test_truncation(self, tmp_path: Path) -> None:
        """Test that longer sequences are truncated to max_seq_len."""
        import json

        sample_dir = tmp_path / "long"
        sample_dir.mkdir()
        long_len = SEQ_LEN + 20
        rng = np.random.default_rng(0)
        poses = rng.standard_normal((long_len, NUM_JOINTS, 3)).astype(np.float32)
        np.save(sample_dir / "poses.npy", poses)

        manifest = tmp_path / "m2.json"
        with manifest.open("w") as fh:
            json.dump([{"path": "long"}], fh)

        ds = ChoreoDataset(manifest, max_seq_len=SEQ_LEN, num_joints=NUM_JOINTS)
        item = ds[0]
        assert item["poses"].shape == (SEQ_LEN, NUM_JOINTS, 3)
        assert item["mask_pose"].all()

    def test_modality_loading(self, manifest_path: Path) -> None:
        ds = ChoreoDataset(
            manifest_path,
            max_seq_len=SEQ_LEN,
            num_joints=NUM_JOINTS,
            modalities=["text", "image", "audio"],
        )
        item = ds[0]
        assert item["text"] is not None
        assert item["image"] is not None
        assert item["audio"] is not None

    def test_no_modalities(self, manifest_path: Path) -> None:
        ds = ChoreoDataset(
            manifest_path,
            max_seq_len=SEQ_LEN,
            num_joints=NUM_JOINTS,
            modalities=[],
        )
        item = ds[0]
        assert item["text"] is None
        assert item["image"] is None
        assert item["audio"] is None

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ChoreoDataset(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------


class TestCollateFn:
    def test_basic_collation(self, manifest_path: Path) -> None:
        ds = ChoreoDataset(manifest_path, max_seq_len=SEQ_LEN, num_joints=NUM_JOINTS)
        samples = [ds[i] for i in range(4)]
        batch = collate_fn(samples)

        assert batch["poses"].shape == (4, SEQ_LEN, NUM_JOINTS, 3)
        assert batch["mask_pose"].shape == (4, SEQ_LEN)

    def test_none_modality_propagated(self, manifest_path: Path) -> None:
        """When all samples have None for a modality it stays None."""
        ds = ChoreoDataset(
            manifest_path,
            max_seq_len=SEQ_LEN,
            num_joints=NUM_JOINTS,
            modalities=[],  # disable all modalities
        )
        samples = [ds[i] for i in range(4)]
        batch = collate_fn(samples)
        assert batch["text"] is None

    def test_partial_none_filled_with_zeros(self, manifest_path: Path) -> None:
        """If only some samples have a modality, missing ones use zeros."""
        ds = ChoreoDataset(manifest_path, max_seq_len=SEQ_LEN, num_joints=NUM_JOINTS)
        samples = [ds[0], ds[1], ds[2], ds[3]]

        # Manually null out text in two samples
        samples[0]["text"] = None
        samples[2]["text"] = None

        batch = collate_fn(samples)
        # Should not be None (non-None values exist)
        assert batch["text"] is not None
        assert batch["text"].shape[0] == 4


# ---------------------------------------------------------------------------
# build_dataloader
# ---------------------------------------------------------------------------


class TestBuildDataloader:
    def test_dataloader_creation(self, manifest_path: Path) -> None:
        loader = build_dataloader(
            manifest_path,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            max_seq_len=SEQ_LEN,
            num_joints=NUM_JOINTS,
        )
        assert isinstance(loader, DataLoader)

    def test_dataloader_batch_shapes(self, manifest_path: Path) -> None:
        loader = build_dataloader(
            manifest_path,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            max_seq_len=SEQ_LEN,
            num_joints=NUM_JOINTS,
            modalities=["text"],
        )
        batch = next(iter(loader))
        assert batch["poses"].shape == (2, SEQ_LEN, NUM_JOINTS, 3)
        assert batch["mask_pose"].shape == (2, SEQ_LEN)
