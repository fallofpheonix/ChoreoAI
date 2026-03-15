"""
torch_dataset.py — PyTorch Dataset for ChoreoAI.

Loads (T, K, 3) pose arrays alongside optional modality tensors
(text tokens, image tensors, audio spectrograms) and returns batched
dictionaries ready for training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class ChoreoDataset(Dataset):
    """Dataset that loads pose sequences and optional paired modalities.

    Each sample directory (or entry in a manifest JSON) must contain:
      - ``poses.npy``: float32 array of shape ``(T, K, 3)``
      - (optional) ``tokens.pt``: pre-tokenised text tensor
      - (optional) ``image.pt``: image tensor ``(C, H, W)``
      - (optional) ``audio.pt``: audio tensor ``(F, T_a)``

    Args:
        manifest_path: Path to a JSON manifest file. Each entry is a dict
            with ``"path"`` (required) and optional fields
            ``"text"``, ``"image"``, ``"audio"`` pointing to relative files.
        max_seq_len: Fixed sequence length for padding / truncation.
        num_joints: Expected number of skeleton joints ``K``.
        modalities: Subset of ``{"text", "image", "audio"}`` to load.
            Defaults to all three when *None*.
        transform: Optional callable applied to the pose tensor.
    """

    MODALITIES = ("text", "image", "audio")

    def __init__(
        self,
        manifest_path: str | Path,
        max_seq_len: int = 120,
        num_joints: int = 17,
        modalities: list[str] | None = None,
        transform: Any | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.max_seq_len = max_seq_len
        self.num_joints = num_joints
        self.modalities = set(modalities) if modalities is not None else set(self.MODALITIES)
        self.transform = transform

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with self.manifest_path.open() as fh:
            self.entries: list[dict[str, Any]] = json.load(fh)

        self._base_dir = self.manifest_path.parent

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_poses(self, entry: dict[str, Any]) -> tuple[Tensor, Tensor]:
        """Load and pad a pose array to ``(max_seq_len, K, 3)``.

        Returns:
            poses: ``(max_seq_len, K, 3)`` float32 tensor.
            mask:  ``(max_seq_len,)`` bool tensor — ``True`` for real frames.
        """
        pose_file = self._base_dir / entry["path"] / "poses.npy"
        raw: np.ndarray = np.load(pose_file).astype(np.float32)  # (T, K, 3)

        T, K, _ = raw.shape
        assert K == self.num_joints, (
            f"Expected {self.num_joints} joints, got {K} in {pose_file}"
        )

        # Build mask before truncation/padding
        T_eff = min(T, self.max_seq_len)
        mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        mask[:T_eff] = True

        # Truncate or zero-pad
        if T >= self.max_seq_len:
            poses_np = raw[: self.max_seq_len]
        else:
            pad = np.zeros((self.max_seq_len - T, K, 3), dtype=np.float32)
            poses_np = np.concatenate([raw, pad], axis=0)

        poses = torch.from_numpy(poses_np)  # (max_seq_len, K, 3)
        return poses, mask

    def _load_modality(
        self, entry: dict[str, Any], key: str
    ) -> Tensor | None:
        """Load a modality tensor if present, else return None."""
        if key not in self.modalities:
            return None
        rel_path = entry.get(key)
        if rel_path is None:
            return None
        full_path = self._base_dir / rel_path
        if not full_path.exists():
            return None
        return torch.load(full_path, weights_only=True)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Tensor | None]:
        entry = self.entries[idx]
        poses, mask = self._load_poses(entry)

        if self.transform is not None:
            poses = self.transform(poses)

        return {
            "poses": poses,                                   # (T, K, 3)
            "mask_pose": mask,                                # (T,)
            "text": self._load_modality(entry, "text"),       # token ids
            "image": self._load_modality(entry, "image"),     # (C, H, W)
            "audio": self._load_modality(entry, "audio"),     # (F, T_a)
        }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_fn(
    batch: list[dict[str, Tensor | None]],
) -> dict[str, Tensor | None]:
    """Collate a list of samples into a batched dictionary.

    None values are propagated as ``None`` if *all* samples in the batch
    are missing that modality; otherwise missing entries are replaced with
    zero tensors matched to the shape of present entries.

    Args:
        batch: List of sample dictionaries from :class:`ChoreoDataset`.

    Returns:
        Batched dictionary with stacked tensors (or ``None``).
    """
    keys = list(batch[0].keys())
    result: dict[str, Tensor | None] = {}

    for key in keys:
        values = [sample[key] for sample in batch]
        non_none = [v for v in values if v is not None]

        if not non_none:
            result[key] = None
            continue

        # Fill missing entries with zeros
        ref_shape = non_none[0].shape
        ref_dtype = non_none[0].dtype
        filled = [
            v if v is not None else torch.zeros(ref_shape, dtype=ref_dtype)
            for v in values
        ]
        result[key] = torch.stack(filled, dim=0)

    return result


# ---------------------------------------------------------------------------
# Convenience DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    manifest_path: str | Path,
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    max_seq_len: int = 120,
    num_joints: int = 17,
    modalities: list[str] | None = None,
) -> DataLoader:
    """Build a :class:`~torch.utils.data.DataLoader` for ChoreoAI training.

    Args:
        manifest_path: Path to the JSON manifest.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset.
        num_workers: Parallel workers for loading.
        max_seq_len: Fixed temporal length after padding.
        num_joints: Expected joint count per frame.
        modalities: Modalities to load.

    Returns:
        Configured DataLoader.
    """
    dataset = ChoreoDataset(
        manifest_path=manifest_path,
        max_seq_len=max_seq_len,
        num_joints=num_joints,
        modalities=modalities,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
