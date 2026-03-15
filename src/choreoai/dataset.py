"""
dataset.py — Dataset schema definitions and manifest utilities.

Provides dataclass schemas for ChoreoAI samples and helpers to
scan data directories and emit JSON manifest files consumed by
:mod:`choreoai.torch_dataset`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data schemas
# ---------------------------------------------------------------------------


@dataclass
class ModalityPaths:
    """Relative file paths for optional paired modalities."""

    text: str | None = None    # path to tokens.pt
    image: str | None = None   # path to image.pt
    audio: str | None = None   # path to audio.pt


@dataclass
class SampleEntry:
    """Manifest entry describing a single ChoreoAI training sample.

    Attributes:
        path: Relative directory containing ``poses.npy``.
        text: Relative path to pre-tokenised text tensor (optional).
        image: Relative path to image tensor (optional).
        audio: Relative path to audio tensor (optional).
        metadata: Free-form dictionary for any extra annotations
                  (e.g. dancer id, piece name, duration in seconds).
    """

    path: str
    text: str | None = None
    image: str | None = None
    audio: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dict, omitting ``None`` modality fields."""
        d = asdict(self)
        for key in ("text", "image", "audio"):
            if d[key] is None:
                del d[key]
        return d


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def save_manifest(entries: list[SampleEntry], manifest_path: str | Path) -> None:
    """Serialise a list of :class:`SampleEntry` to a JSON manifest.

    Args:
        entries: Dataset entries to serialise.
        manifest_path: Destination JSON file path.
    """
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [e.to_dict() for e in entries]
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def load_manifest(manifest_path: str | Path) -> list[SampleEntry]:
    """Load a manifest JSON file into a list of :class:`SampleEntry`.

    Args:
        manifest_path: Path to the JSON manifest.

    Returns:
        List of dataset entries.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open() as fh:
        raw: list[dict[str, Any]] = json.load(fh)
    return [SampleEntry(**entry) for entry in raw]


def scan_data_directory(
    root: str | Path,
    *,
    require_text: bool = False,
    require_image: bool = False,
    require_audio: bool = False,
) -> list[SampleEntry]:
    """Walk *root* and auto-discover samples by the presence of ``poses.npy``.

    For each discovered sample directory the function checks for optional
    companion files ``tokens.pt``, ``image.pt``, and ``audio.pt``.

    Args:
        root: Top-level data directory.
        require_text: Skip samples without a text modality file.
        require_image: Skip samples without an image modality file.
        require_audio: Skip samples without an audio modality file.

    Returns:
        List of discovered :class:`SampleEntry` objects with paths relative
        to *root*.
    """
    root = Path(root)
    entries: list[SampleEntry] = []

    for pose_file in sorted(root.rglob("poses.npy")):
        sample_dir = pose_file.parent
        rel = sample_dir.relative_to(root)

        text_file = sample_dir / "tokens.pt"
        image_file = sample_dir / "image.pt"
        audio_file = sample_dir / "audio.pt"

        text_rel = str(rel / "tokens.pt") if text_file.exists() else None
        image_rel = str(rel / "image.pt") if image_file.exists() else None
        audio_rel = str(rel / "audio.pt") if audio_file.exists() else None

        # Apply optional filters
        if require_text and text_rel is None:
            continue
        if require_image and image_rel is None:
            continue
        if require_audio and audio_rel is None:
            continue

        entries.append(
            SampleEntry(
                path=str(rel),
                text=text_rel,
                image=image_rel,
                audio=audio_rel,
            )
        )

    return entries
