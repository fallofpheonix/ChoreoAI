"""Dataset management service wrapping indexing and preprocessing modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from choreoai.dataset_index import (
    bootstrap_dataset_from_raw,
    stage_pose_sequence,
    summarize_dataset,
    validate_dataset,
)
from choreoai.preprocess_np import PreprocessConfig, preprocess_dataset


@dataclass
class DatasetService:
    """Handles dataset validation, staging, summarization, and preprocessing."""

    def validate(self, root: Path) -> list[str]:
        return validate_dataset(root)

    def summarize(self, root: Path):
        return summarize_dataset(root)

    def stage(
        self,
        source_path: Path,
        dataset_root: Path,
        seq_id: str,
        text: str | None,
        force: bool,
    ) -> Path:
        return stage_pose_sequence(
            source_path=source_path,
            dataset_root=dataset_root,
            seq_id=seq_id,
            text=text,
            force=force,
        )

    def bootstrap(self, raw_root: Path, dataset_root: Path, force: bool) -> list[Path]:
        return bootstrap_dataset_from_raw(raw_root=raw_root, dataset_root=dataset_root, force=force)

    def preprocess(
        self,
        input_root: Path,
        output_root: Path,
        smooth_window: int,
        center_joint: int,
        force: bool,
    ) -> list[Path]:
        config = PreprocessConfig(smooth_window=smooth_window, center_joint=center_joint)
        return preprocess_dataset(
            input_root=input_root,
            output_root=output_root,
            config=config,
            force=force,
        )
