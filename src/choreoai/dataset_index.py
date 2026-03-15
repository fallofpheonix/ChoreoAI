"""
dataset_index.py — Dataset indexing, validation, and staging utilities.

Provides utilities for managing a directory-based dataset layout where
each sequence is stored in its own subdirectory with poses.npy and
optional modality files.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


@dataclass(frozen=True)
class SequenceExample:
    """Represents a single sequence in the dataset."""

    seq_id: str
    poses_path: Path
    text_path: Path | None
    image_path: Path | None
    audio_path: Path | None


@dataclass(frozen=True)
class SequenceSummary:
    """Summary statistics for a single sequence."""

    seq_id: str
    frames: int
    joints: int
    dims: int
    dtype: str
    has_text: bool
    has_image: bool
    has_audio: bool


class DatasetIndex:
    """Index for iterating over sequences in a dataset directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def sequences(self) -> Iterator[SequenceExample]:
        """Iterate over all sequence directories in the dataset."""
        if not self.root.exists():
            raise FileNotFoundError(f"dataset root not found: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"dataset root is not a directory: {self.root}")

        for seq_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            poses = seq_dir / "poses.npy"
            text = seq_dir / "text_prompt.txt"
            image = seq_dir / "image_reference.png"
            audio = seq_dir / "audio.wav"

            yield SequenceExample(
                seq_id=seq_dir.name,
                poses_path=poses,
                text_path=text if text.exists() else None,
                image_path=image if image.exists() else None,
                audio_path=audio if audio.exists() else None,
            )


def load_pose_array(path: Path, allow_nonfinite: bool = False) -> np.ndarray:
    """Load and validate a pose array from an .npy file.

    Args:
        path: Path to the .npy file.
        allow_nonfinite: If True, allow NaN/Inf values in the array.

    Returns:
        Pose array with shape (T, K, 3).

    Raises:
        ValueError: If the array has invalid shape or dtype.
    """
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"invalid rank {arr.ndim}, expected 3")
    if arr.shape[-1] != 3:
        raise ValueError(f"invalid trailing dimension {arr.shape[-1]}, expected 3")
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError(f"invalid empty shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"invalid dtype {arr.dtype}, expected numeric")
    if not allow_nonfinite and not np.isfinite(arr).all():
        raise ValueError("pose array contains non-finite values")
    return arr


def validate_sequence(example: SequenceExample) -> list[str]:
    """Validate a single sequence, returning a list of error messages."""
    errors: list[str] = []
    if not example.poses_path.exists():
        errors.append(f"{example.seq_id}: missing poses.npy")
        return errors

    try:
        arr = load_pose_array(example.poses_path)
    except Exception as exc:  # noqa: BLE001 - capture numpy errors
        errors.append(f"{example.seq_id}: poses.npy unreadable ({exc})")
        return errors

    if arr.ndim != 3 or arr.shape[-1] != 3:
        errors.append(
            f"{example.seq_id}: poses.npy has invalid shape {arr.shape}, expected (T,K,3)"
        )

    return errors


def validate_dataset(root: Path) -> list[str]:
    """Validate all sequences in a dataset, returning all error messages."""
    index = DatasetIndex(root)
    errors: list[str] = []
    for example in index.sequences():
        errors.extend(validate_sequence(example))
    return errors


def iter_pose_sequences(root: Path) -> Iterable[np.ndarray]:
    """Iterate over all valid pose arrays in a dataset."""
    for example in DatasetIndex(root).sequences():
        if example.poses_path.exists():
            yield load_pose_array(example.poses_path)


def summarize_dataset(root: Path) -> list[SequenceSummary]:
    """Generate summary statistics for all sequences in a dataset."""
    summaries: list[SequenceSummary] = []
    for example in DatasetIndex(root).sequences():
        if not example.poses_path.exists():
            continue
        try:
            arr = load_pose_array(example.poses_path)
        except Exception:  # noqa: BLE001
            continue
        summaries.append(
            SequenceSummary(
                seq_id=example.seq_id,
                frames=int(arr.shape[0]),
                joints=int(arr.shape[1]),
                dims=int(arr.shape[2]),
                dtype=str(arr.dtype),
                has_text=example.text_path is not None,
                has_image=example.image_path is not None,
                has_audio=example.audio_path is not None,
            )
        )
    return summaries


def stage_pose_sequence(
    source_path: Path,
    dataset_root: Path,
    seq_id: str,
    text: str | None = None,
    force: bool = False,
) -> Path:
    """Stage a pose sequence from a source .npy file into the dataset layout.

    Args:
        source_path: Path to the source .npy file.
        dataset_root: Root directory of the dataset.
        seq_id: Identifier for the new sequence.
        text: Optional text prompt to store with the sequence.
        force: If True, overwrite existing sequence directory.

    Returns:
        Path to the created sequence directory.

    Raises:
        FileExistsError: If the sequence already exists and force is False.
    """
    arr = load_pose_array(source_path)
    seq_dir = dataset_root / seq_id
    poses_path = seq_dir / "poses.npy"

    if seq_dir.exists() and not force:
        raise FileExistsError(f"sequence already exists: {seq_dir}")
    elif seq_dir.exists():
        shutil.rmtree(seq_dir)

    seq_dir.mkdir(parents=True, exist_ok=True)
    np.save(poses_path, arr)

    if text is not None:
        (seq_dir / "text_prompt.txt").write_text(text.strip() + "\n", encoding="utf-8")

    return seq_dir


def bootstrap_dataset_from_raw(
    raw_root: Path,
    dataset_root: Path,
    force: bool = False,
) -> list[Path]:
    """Bootstrap a dataset from raw .npy files in a directory.

    Args:
        raw_root: Directory containing raw .npy pose files.
        dataset_root: Root directory for the output dataset.
        force: If True, overwrite existing sequence directories.

    Returns:
        List of paths to created sequence directories.

    Raises:
        FileNotFoundError: If the raw root does not exist.
        NotADirectoryError: If the raw root is not a directory.
        FileExistsError: If a sequence already exists and force is False.
    """
    if not raw_root.exists():
        raise FileNotFoundError(f"raw root not found: {raw_root}")
    if not raw_root.is_dir():
        raise NotADirectoryError(f"raw root is not a directory: {raw_root}")

    created: list[Path] = []
    for source in sorted(raw_root.glob("*.npy")):
        seq_id = source.stem
        seq_dir = dataset_root / seq_id
        if seq_dir.exists() and not force:
            raise FileExistsError(f"sequence already exists: {seq_dir}")
        elif seq_dir.exists():
            shutil.rmtree(seq_dir)
        seq_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, seq_dir / "poses.npy")
        created.append(seq_dir)
    return created
