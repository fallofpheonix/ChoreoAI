from __future__ import annotations

from pathlib import Path

import numpy as np

from choreoai.services.dataset_service import DatasetService


def test_validate_reports_missing_poses(tmp_path: Path) -> None:
    (tmp_path / "broken_seq").mkdir()

    service = DatasetService()
    errors = service.validate(tmp_path)

    assert len(errors) == 1
    assert "missing poses.npy" in errors[0]


def test_stage_and_summarize_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "raw.npy"
    np.save(source, np.zeros((8, 17, 3), dtype=np.float32))

    dataset_root = tmp_path / "dataset"
    service = DatasetService()

    seq_dir = service.stage(
        source_path=source,
        dataset_root=dataset_root,
        seq_id="seq_001",
        text="short prompt",
        force=False,
    )

    summaries = service.summarize(dataset_root)

    assert seq_dir.exists()
    assert len(summaries) == 1
    assert summaries[0].frames == 8
