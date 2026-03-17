"""Pydantic schemas for API boundary models."""

from pydantic import BaseModel


class GenerationRequestModel(BaseModel):
    prompt: str
    num_joints: int = 17
    seq_len: int = 120
    guidance_scale: float = 2.0


class GenerationResponseModel(BaseModel):
    status: str
    shape: list[int]
    motion: list[list[list[float]]]
