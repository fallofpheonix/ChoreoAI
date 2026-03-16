"""
api.py — FastAPI service for ChoreoAI generation.

Provides a REST API to trigger motion generation from text prompts.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from choreoai.inference import generate_motion

from choreoai.utils.metrics import global_metrics
import time

app = FastAPI(title="ChoreoAI API", version="0.1.0")

class GenerationRequest(BaseModel):
    prompt: str
    num_joints: int = 17
    seq_len: int = 120
    guidance_scale: float = 2.0

class GenerationResponse(BaseModel):
    status: str
    shape: list[int]
    motion: list[list[list[float]]]

@app.post("/generate_motion", response_model=GenerationResponse)
def generate(req: GenerationRequest):
    try:
        start_time = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        motion_tensor = generate_motion(
            text=req.prompt,
            num_joints=req.num_joints,
            seq_len=req.seq_len,
            guidance_scale=req.guidance_scale,
            device=device
        )
        
        global_metrics.record_inference(start_time)
        
        return GenerationResponse(
            status="success",
            shape=list(motion_tensor.shape),
            motion=motion_tensor.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    return global_metrics.get_summary()

@app.get("/health")
def health_check():
    return {"status": "ok"}
