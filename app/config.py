from __future__ import annotations

import os
from pathlib import Path


def _parse_csv_env(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


DATA_RAW_DIR = Path(os.getenv("DATA_RAW_DIR", "/app/data/raw"))
UPLOAD_MAX_BYTES = int(os.getenv("UPLOAD_MAX_BYTES", str(25 * 1024 * 1024)))

ADMIN_API_TOKEN = os.getenv("ADMIN_API_TOKEN", "")
CORS_ALLOW_ORIGINS = _parse_csv_env("CORS_ALLOW_ORIGINS", "*")
