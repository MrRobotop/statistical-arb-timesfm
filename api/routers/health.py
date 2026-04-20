"""GET /health — liveness check."""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import HealthResponse
from pipeline.model.loader import TimesFMLoader

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loader = TimesFMLoader.get_instance()
    return HealthResponse(status="ok", model_loaded=loader.is_loaded())
