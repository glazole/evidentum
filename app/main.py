from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.db import init_db


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

    app.include_router(api_router, prefix="/api", tags=["api"])

    return app


app = create_app()