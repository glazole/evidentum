from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db import session_scope
from app.models import Document
from app.services.answer import answer_question, format_for_ui
from app.services.retriever import retrieve


router = APIRouter()


TranslateMode = Literal["off", "query", "query_and_hits", "dual_query"]


class HealthResponse(BaseModel):
    status: str


class SourceItemResponse(BaseModel):
    document_id: int
    source_id: str
    source_name: str
    title: str
    region: str | None = None
    year: int | None = None
    url: str | None = None
    chunk_count: int = 0


class SourcesResponse(BaseModel):
    items: list[SourceItemResponse]


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=6, ge=1, le=20)
    translate_mode: TranslateMode = "dual_query"
    document_id: int | None = None
    source_id: str | None = None
    region: str | None = None
    debug: bool = False


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=6, ge=1, le=20)
    translate_mode: TranslateMode = "dual_query"
    model: str = "alice"
    temperature: float = Field(default=0.2, ge=0.0, le=1.5)
    max_output_tokens: int = Field(default=1800, ge=128, le=4000)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/sources", response_model=SourcesResponse)
def list_sources() -> SourcesResponse:
    with session_scope() as session:
        documents = list(
            session.scalars(
                select(Document).order_by(Document.region.asc(), Document.id.asc())
            )
        )

        items: list[SourceItemResponse] = []
        for doc in documents:
            items.append(
                SourceItemResponse(
                    document_id=doc.id,
                    source_id=doc.source_id,
                    source_name=doc.source_name,
                    title=doc.title,
                    region=doc.region,
                    year=doc.year,
                    url=doc.url,
                    chunk_count=len(doc.chunks),
                )
            )

    return SourcesResponse(items=items)


@router.post("/retrieve")
def retrieve_route(payload: RetrieveRequest) -> dict[str, Any]:
    try:
        result = retrieve(
            payload.question,
            top_k=payload.top_k,
            document_id=payload.document_id,
            source_id=payload.source_id,
            region=payload.region,
            translate_mode=payload.translate_mode,
            debug=payload.debug,
        )
        return result.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc


@router.post("/answer")
def answer_route(payload: AnswerRequest) -> dict[str, Any]:
    try:
        result = answer_question(
            payload.question,
            top_k=payload.top_k,
            translate_mode=payload.translate_mode,
            model_family=payload.model,
            temperature=payload.temperature,
            max_output_tokens=payload.max_output_tokens,
            return_raw=False,
        )
        return format_for_ui(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {exc}") from exc