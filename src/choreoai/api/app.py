"""FastAPI application wiring for ChoreoAI."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from choreoai.api.schemas import GenerationRequestModel, GenerationResponseModel
from choreoai.core.errors import ValidationError
from choreoai.core.requests import MotionGenerationRequest
from choreoai.services.generation_service import GenerationService


def create_app() -> FastAPI:
    app = FastAPI(title="ChoreoAI API", version="0.1.0")
    generation_service = GenerationService()

    @app.post("/generate_motion", response_model=GenerationResponseModel)
    def generate(payload: GenerationRequestModel) -> GenerationResponseModel:
        request = MotionGenerationRequest(
            prompt=payload.prompt,
            num_joints=payload.num_joints,
            seq_len=payload.seq_len,
            guidance_scale=payload.guidance_scale,
        )
        try:
            motion = generation_service.generate(request)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return GenerationResponseModel(
            status="success",
            shape=list(motion.shape),
            motion=motion.tolist(),
        )

    @app.get("/metrics")
    def get_metrics() -> dict[str, float | int | str]:
        return generation_service.metrics_snapshot()

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
