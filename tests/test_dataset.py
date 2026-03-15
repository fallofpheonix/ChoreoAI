"""Tests for dataset schema utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from choreoai.dataset import (
    SampleEntry,
    save_manifest,
    load_manifest,
    scan_data_directory,
)


class TestSampleEntry:
    def test_to_dict_omits_none_modalities(self) -> None:
        entry = SampleEntry(path="sample/001", text="sample/001/tokens.pt")
        d = entry.to_dict()
        assert "image" not in d
        assert "audio" not in d
        assert d["text"] == "sample/001/tokens.pt"

    def test_to_dict_includes_all_modalities(self) -> None:
        entry = SampleEntry(
            path="s",
            text="s/t.pt",
            image="s/i.pt",
            audio="s/a.pt",
        )
        d = entry.to_dict()
        assert all(k in d for k in ("path", "text", "image", "audio"))


class TestManifestIO:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        entries = [
            SampleEntry(path="a", text="a/t.pt"),
            SampleEntry(path="b", audio="b/a.pt"),
        ]
        manifest = tmp_path / "m.json"
        save_manifest(entries, manifest)

        assert manifest.exists()
        loaded = load_manifest(manifest)
        assert len(loaded) == 2
        assert loaded[0].path == "a"
        assert loaded[0].text == "a/t.pt"
        assert loaded[1].audio == "b/a.pt"

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nope.json")


class TestScanDataDirectory:
    def test_discovers_samples(self, tmp_data_dir: Path) -> None:
        entries = scan_data_directory(tmp_data_dir)
        assert len(entries) == 6

    def test_relative_paths(self, tmp_data_dir: Path) -> None:
        entries = scan_data_directory(tmp_data_dir)
        for e in entries:
            # Path should not start with /
            assert not Path(e.path).is_absolute()

    def test_require_text_filters(self, tmp_path: Path) -> None:
        """Samples without text should be excluded when require_text=True."""
        # Create two samples: one with text, one without
        (tmp_path / "with_text").mkdir()
        np.save(tmp_path / "with_text" / "poses.npy", np.zeros((10, 17, 3), dtype=np.float32))
        torch.save(torch.zeros(32), tmp_path / "with_text" / "tokens.pt")

        (tmp_path / "no_text").mkdir()
        np.save(tmp_path / "no_text" / "poses.npy", np.zeros((10, 17, 3), dtype=np.float32))

        entries = scan_data_directory(tmp_path, require_text=True)
        assert len(entries) == 1
        assert "with_text" in entries[0].path

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        entries = scan_data_directory(tmp_path)
        assert entries == []
