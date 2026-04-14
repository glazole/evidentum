from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.db import session_scope
from app.models import Chunk, Document, Subscription
from app.services.retriever import RetrievalResult, retrieve


app = FastAPI(title="evidentum API", version="0.1.0")


TranslateMode = Literal["off", "query", "query_and_hits", "dual_query"]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=50)
    document_id: int | None = None
    source_id: str | None = None
    region: str | None = None
    translate_mode: TranslateMode | None = None
    debug: bool = False


class SourceItem(BaseModel):
    idx: int
    score: float
    distance: float
    source_name: str
    title: str
    url: str | None
    section_title: str | None
    chunk_id: int
    document_id: int
    excerpt: str


class AskResponse(BaseModel):
    query: str
    effective_query: str
    answer_markdown: str
    sources: list[SourceItem]
    created_at: datetime


def _excerpt(text: str, *, limit: int = 420) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= limit:
        return t
    return t[:limit].rstrip() + "…"


def _format_answer(result: RetrievalResult) -> tuple[str, list[SourceItem]]:
    sources: list[SourceItem] = []
    lines: list[str] = []

    if not result.hits:
        return (
            "Не нашёл релевантных фрагментов в базе. Попробуй переформулировать запрос или загрузить гайдлайны.",
            [],
        )

    lines.append("Ниже — наиболее релевантные фрагменты из гайдлайнов (с ссылками на источники):")
    lines.append("")

    for idx, hit in enumerate(result.hits, start=1):
        chunk_text = hit.translated_chunk_text or hit.chunk_text
        sources.append(
            SourceItem(
                idx=idx,
                score=float(hit.score),
                distance=float(hit.distance),
                source_name=str(hit.source_name),
                title=str(hit.title),
                url=hit.url,
                section_title=hit.section_title,
                chunk_id=int(hit.chunk_id),
                document_id=int(hit.document_id),
                excerpt=_excerpt(chunk_text),
            )
        )

        section = f" — {hit.section_title}" if hit.section_title else ""
        url_part = f" ([источник]({hit.url}))" if hit.url else ""
        lines.append(f"**[{idx}] {hit.source_name} — {hit.title}{section}**{url_part}")
        lines.append("")
        lines.append(f"> {_excerpt(chunk_text, limit=700)}")
        lines.append("")

    lines.append(
        "_Дисклеймер: ответ носит информационный характер и не является медицинской рекомендацией._"
    )

    return "\n".join(lines).strip(), sources


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
def api_ask(payload: AskRequest) -> AskResponse:
    result = retrieve(
        payload.query,
        top_k=payload.top_k,
        document_id=payload.document_id,
        source_id=payload.source_id,
        region=payload.region,
        translate_mode=(payload.translate_mode or None),  # type: ignore[arg-type]
        debug=payload.debug,
    )
    answer_md, sources = _format_answer(result)
    return AskResponse(
        query=result.query,
        effective_query=result.effective_query,
        answer_markdown=answer_md,
        sources=sources,
        created_at=datetime.utcnow(),
    )


class DocumentItem(BaseModel):
    document_id: int
    source_id: str
    source_name: str
    title: str
    region: str | None
    year: int | None
    url: str | None
    total_chunks: int
    embedded_chunks: int


@app.get("/api/documents", response_model=list[DocumentItem])
def api_documents() -> list[DocumentItem]:
    with session_scope() as session:
        docs = list(session.scalars(select(Document).order_by(Document.id.asc())))
        result: list[DocumentItem] = []
        for doc in docs:
            total_chunks = int(
                session.scalar(select(func.count(Chunk.id)).where(Chunk.document_id == doc.id)) or 0
            )
            embedded_chunks = int(
                session.scalar(
                    select(func.count(Chunk.id)).where(
                        Chunk.document_id == doc.id,
                        Chunk.embedding.is_not(None),
                    )
                )
                or 0
            )
            result.append(
                DocumentItem(
                    document_id=doc.id,
                    source_id=doc.source_id,
                    source_name=doc.source_name,
                    title=doc.title,
                    region=doc.region,
                    year=doc.year,
                    url=doc.url,
                    total_chunks=total_chunks,
                    embedded_chunks=embedded_chunks,
                )
            )
        return result


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(12, ge=1, le=50)
    region: str | None = None
    translate_mode: TranslateMode | None = None


class CompareRow(BaseModel):
    source_name: str
    title: str
    url: str | None
    excerpt: str
    score: float


class CompareResponse(BaseModel):
    query: str
    rows: list[CompareRow]


@app.post("/api/compare", response_model=CompareResponse)
def api_compare(payload: CompareRequest) -> CompareResponse:
    # For MVP: retrieve hits and group by source_name picking best hit per source.
    result = retrieve(
        payload.query,
        top_k=payload.top_k,
        region=payload.region,
        translate_mode=(payload.translate_mode or None),  # type: ignore[arg-type]
    )

    best_by_source: dict[str, CompareRow] = {}
    for hit in result.hits:
        source = str(hit.source_name)
        chunk_text = hit.translated_chunk_text or hit.chunk_text
        row = CompareRow(
            source_name=source,
            title=str(hit.title),
            url=hit.url,
            excerpt=_excerpt(chunk_text, limit=360),
            score=float(hit.score),
        )
        existing = best_by_source.get(source)
        if existing is None or row.score > existing.score:
            best_by_source[source] = row

    rows = sorted(best_by_source.values(), key=lambda r: r.score, reverse=True)
    return CompareResponse(query=result.query, rows=rows)


class SubscribeRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    topic: str = Field(..., min_length=1, max_length=512)


class SubscribeResponse(BaseModel):
    ok: bool
    subscription_id: int


@app.post("/api/subscribe", response_model=SubscribeResponse)
def api_subscribe(payload: SubscribeRequest) -> SubscribeResponse:
    email = payload.email.strip().lower()
    topic = payload.topic.strip()
    with session_scope() as session:
        existing = session.scalar(
            select(Subscription).where(Subscription.email == email, Subscription.topic == topic)
        )
        if existing is not None:
            return SubscribeResponse(ok=True, subscription_id=existing.id)

        sub = Subscription(email=email, topic=topic)
        session.add(sub)
        session.flush()
        return SubscribeResponse(ok=True, subscription_id=sub.id)