from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import distinct, select

from app.db import session_scope
from app.models import Chunk, Document
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
    specialty: str | None = None
    nosology_primary: str | None = None
    has_summary: bool = False


class SourcesResponse(BaseModel):
    items: list[SourceItemResponse]


class CatalogResponse(BaseModel):
    specialties: list[str]
    nosologies: list[str]
    source_names: list[str]


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=6, ge=1, le=20)
    translate_mode: TranslateMode = "dual_query"
    document_id: int | None = None
    source_id: str | None = None
    region: str | None = None
    specialty: str | None = None
    nosology: str | None = None
    per_source_k: int | None = None
    debug: bool = False


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=6, ge=1, le=20)
    translate_mode: TranslateMode = "dual_query"
    model: str = "alice"
    temperature: float = Field(default=0.2, ge=0.0, le=1.5)
    max_output_tokens: int = Field(default=1800, ge=128, le=4000)
    synthesize: bool = True
    specialty: str | None = None
    nosology: str | None = None
    per_source_k: int | None = None


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=8, ge=1, le=20)
    translate_mode: TranslateMode = "dual_query"
    model: str = "alice"
    temperature: float = Field(default=0.2, ge=0.0, le=1.5)
    per_source_k: int = Field(default=2, ge=1, le=5)
    specialty: str | None = None
    nosology: str | None = None


class DocumentSummaryResponse(BaseModel):
    document_id: int
    source_name: str
    year: int | None
    summary_ru: str | None
    version_delta_ru: str | None


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
                    specialty=getattr(doc, "specialty", None),
                    nosology_primary=getattr(doc, "nosology_primary", None),
                    has_summary=bool(getattr(doc, "summary_ru", None)),
                )
            )

    return SourcesResponse(items=items)


@router.get("/catalog", response_model=CatalogResponse)
def get_catalog() -> CatalogResponse:
    with session_scope() as session:
        specialties = [
            r[0] for r in session.execute(
                select(distinct(Chunk.specialty))
                .where(Chunk.specialty.is_not(None))
                .order_by(Chunk.specialty)
            ).all()
        ]
        nosologies = [
            r[0] for r in session.execute(
                select(distinct(Chunk.nosology))
                .where(Chunk.nosology.is_not(None))
                .order_by(Chunk.nosology)
            ).all()
        ]
        source_names = [
            r[0] for r in session.execute(
                select(distinct(Document.source_name))
                .order_by(Document.source_name)
            ).all()
        ]

    return CatalogResponse(
        specialties=specialties,
        nosologies=nosologies,
        source_names=source_names,
    )


@router.post("/retrieve")
def retrieve_route(payload: RetrieveRequest) -> dict[str, Any]:
    try:
        result = retrieve(
            payload.question,
            top_k=payload.top_k,
            document_id=payload.document_id,
            source_id=payload.source_id,
            region=payload.region,
            specialty=payload.specialty,
            nosology=payload.nosology,
            per_source_k=payload.per_source_k,
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
            synthesize=payload.synthesize,
            specialty=payload.specialty,
            nosology=payload.nosology,
            per_source_k=payload.per_source_k,
        )
        return format_for_ui(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {exc}") from exc


@router.post("/compare")
def compare_route(payload: CompareRequest) -> dict[str, Any]:
    """
    Retrieve top chunks across multiple guidelines and compare their positions
    using LLM-based structured synthesis.
    """
    try:
        retrieval = retrieve(
            payload.question,
            top_k=payload.top_k,
            translate_mode=payload.translate_mode,
            per_source_k=payload.per_source_k,
            specialty=payload.specialty,
            nosology=payload.nosology,
        )

        hits = retrieval.hits
        if not hits:
            return {"question": payload.question, "sources": [], "comparison": "Фрагменты не найдены."}

        # Group by source
        by_source: dict[str, list[Any]] = {}
        for h in hits:
            by_source.setdefault(h.source_id, []).append(h)

        # Build per-source position using the answerer
        from app.services.answer import YandexOpenAIAnswerer, _normalize_sources

        answerer = YandexOpenAIAnswerer()
        sources_list = _normalize_sources(hits)

        structured = answerer.synthesize_structured(
            question=payload.question,
            context_sources=sources_list,
            model_family=payload.model,
            temperature=payload.temperature,
        )

        source_rows = []
        for source_id, source_hits in by_source.items():
            first = source_hits[0]
            text_combined = "\n---\n".join(
                (h.translated_chunk_text or h.chunk_text or "").strip()[:600]
                for h in source_hits
            )
            source_rows.append({
                "source_id": source_id,
                "source_name": first.source_name,
                "title": first.title,
                "region": first.region,
                "year": first.year,
                "url": first.url,
                "chunk_count": len(source_hits),
                "combined_text": text_combined,
                "score": max(h.score for h in source_hits),
            })

        return {
            "question": payload.question,
            "effective_query": retrieval.effective_query,
            "sources": source_rows,
            "consensus": structured.get("consensus"),
            "disagreements": structured.get("disagreements") or [],
            "recommendation": structured.get("recommendation"),
            "citations": structured.get("citations") or [],
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Compare failed: {exc}") from exc


@router.get("/documents/{document_id}/summary", response_model=DocumentSummaryResponse)
def get_document_summary(document_id: int) -> DocumentSummaryResponse:
    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        return DocumentSummaryResponse(
            document_id=doc.id,
            source_name=doc.source_name,
            year=doc.year,
            summary_ru=getattr(doc, "summary_ru", None),
            version_delta_ru=getattr(doc, "version_delta_ru", None),
        )


def _generate_summary_task(document_id: int, model_family: str) -> None:
    try:
        from app.services.summarizer import generate_document_summary
        from app.services.llm import YandexLLMClient
        llm = YandexLLMClient(model_family=model_family)
        generate_document_summary(document_id, llm=llm)
    except Exception as exc:
        import sys
        print(f"[summarizer] error for document {document_id}: {exc}", file=sys.stderr)


@router.post("/documents/{document_id}/summary", response_model=DocumentSummaryResponse)
def generate_summary(
    document_id: int,
    background_tasks: BackgroundTasks,
    model: str = "alice",
) -> DocumentSummaryResponse:
    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    background_tasks.add_task(_generate_summary_task, document_id, model)

    with session_scope() as session:
        doc = session.get(Document, document_id)
        return DocumentSummaryResponse(
            document_id=doc.id,
            source_name=doc.source_name,
            year=doc.year,
            summary_ru=getattr(doc, "summary_ru", None),
            version_delta_ru=getattr(doc, "version_delta_ru", None),
        )
