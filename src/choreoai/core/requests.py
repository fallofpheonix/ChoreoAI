"""Core request DTOs for orchestration services."""

from __future__ import annotations

from dataclasses import dataclass

from choreoai.core.errors import ValidationError


@dataclass(frozen=True)
class MotionGenerationRequest:
    """Validated generation request used by the service layer."""

    prompt: str
    num_joints: int = 17
    seq_len: int = 120
    guidance_scale: float = 2.0
    device: str | None = None

    def validate(self) -> None:
        text = self.prompt.strip()
        if not text:
            raise ValidationError("prompt cannot be empty")
        if len(text) > 1000:
            raise ValidationError("prompt is too long")
        if self.num_joints <= 0:
            raise ValidationError("num_joints must be > 0")
        if self.seq_len <= 0:
            raise ValidationError("seq_len must be > 0")
        if self.guidance_scale < 1.0:
            raise ValidationError("guidance_scale must be >= 1.0")
