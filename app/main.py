from __future__ import annotations

import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.db import init_db


def _run_autoprocess() -> None:
    import time
    time.sleep(5)  # let uvicorn finish starting up
    try:
        from app.services.autoprocess import run_autoprocess
        run_autoprocess()
    except Exception as exc:
        import sys
        print(f"[autoprocess] fatal error: {exc}", file=sys.stderr)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Evidentum API",
        version="0.1.0",
        description="API for retrieval and grounded answers over clinical guidelines.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def on_startup() -> None:
        init_db()
        t = threading.Thread(target=_run_autoprocess, daemon=True, name="autoprocess")
        t.start()

    app.include_router(api_router, prefix="/api", tags=["api"])

    return app


app = create_app()
