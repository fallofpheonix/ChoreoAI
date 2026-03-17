from __future__ import annotations

import pytest

from choreoai.core.errors import ValidationError
from choreoai.core.requests import MotionGenerationRequest


def test_generation_request_validation_rejects_blank_prompt() -> None:
    request = MotionGenerationRequest(prompt="   ")

    with pytest.raises(ValidationError):
        request.validate()


def test_generation_request_validation_accepts_normal_prompt() -> None:
    request = MotionGenerationRequest(prompt="a dancer moves slowly", guidance_scale=2.0)
    request.validate()
