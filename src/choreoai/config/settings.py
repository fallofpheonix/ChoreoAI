"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppSettings:
    env: str = "development"
    log_level: str = "INFO"
    default_device: str = "cpu"


def load_settings() -> AppSettings:
    default_device = os.getenv("CHOREOAI_DEFAULT_DEVICE", "cpu").strip().lower()
    if default_device not in {"cpu", "cuda"}:
        default_device = "cpu"

    return AppSettings(
        env=os.getenv("CHOREOAI_ENV", "development").strip() or "development",
        log_level=os.getenv("CHOREOAI_LOG_LEVEL", "INFO").strip().upper() or "INFO",
        default_device=default_device,
    )
