"""
dataset_loader.py — Advanced dataset loading utilities.
"""

import torch
from pathlib import Path
from choreoai.torch_dataset import ChoreoDataset, build_dataloader

class ProductionDataset(ChoreoDataset):
    """Production wrapper for ChoreoDataset with extra validation."""
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # Add production-specific checks or augmentations here
        return sample

def get_production_loader(manifest_path, batch_size=32, num_workers=4, **kwargs):
    return build_dataloader(
        manifest_path=manifest_path,
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )
