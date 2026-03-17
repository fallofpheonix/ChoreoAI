"""Generation orchestration service."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from choreoai.core.requests import MotionGenerationRequest
from choreoai.inference import generate_motion
from choreoai.utils.metrics import global_metrics


@dataclass
class GenerationService:
    """Coordinates validation, inference invocation, and basic telemetry."""

    def generate(self, request: MotionGenerationRequest) -> torch.Tensor:
        request.validate()

        start = time.time()
        device = request.device or ("cuda" if torch.cuda.is_available() else "cpu")
        motion = generate_motion(
            text=request.prompt,
            num_joints=request.num_joints,
            seq_len=request.seq_len,
            guidance_scale=request.guidance_scale,
            device=device,
        )
        global_metrics.record_inference(start)
        return motion

    def metrics_snapshot(self) -> dict[str, float | int | str]:
        return global_metrics.get_summary()
