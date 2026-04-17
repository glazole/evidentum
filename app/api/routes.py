from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import delete as sa_delete
from sqlalchemy import distinct, func, select

DATA_RAW_DIR = Path("/app/data/raw")

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
    missing_embeddings: int = 0
    enriched_chunks: int = 0
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
    source_id: str | None = None
    document_id: int | None = None


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=12, ge=1, le=30)
    translate_mode: TranslateMode = "dual_query"
    model: str = "alice"
    temperature: float = Field(default=0.2, ge=0.0, le=1.5)
    min_score: float = Field(default=0.45, ge=0.0, le=1.0)
    max_chunks_per_source: int = Field(default=3, ge=1, le=6)
    specialty: str | None = None
    nosology: str | None = None
    nosology_filter: bool = Field(
        default=True,
        description=(
            "If True, filter out sources whose enriched chunks have nosology/specialty "
            "metadata that has no word overlap with the question. "
            "Sources with score >= nosology_bypass_score are always kept. "
            "Sources without enough enrichment data are also always kept."
        ),
    )
    nosology_bypass_score: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Sources with best chunk score >= this value bypass nosology filter.",
    )


# ── Nosology/specialty metadata filter ─────────────────────────────────────────

# Common Russian medical abbreviations → expanded form
_MEDICAL_ABBREVS: dict[str, str] = {
    "ба": "астма",
    "фп": "фибрилляция",
    "аг": "гипертония гипертензия",
    "хсн": "сердечная недостаточность",
    "хобл": "обструктивная лёгочная",
    "ибс": "ишемическая болезнь",
    "им": "инфаркт",
    "сд": "диабет",
    "хбп": "почечная болезнь",
    "дм": "диабет",
    "гкмп": "кардиомиопатия",
    "тэла": "эмболия",
    "чсс": "ритм",
    "инсульт": "инсульт цереброваскулярный",
    "бронхит": "бронхит лёгочный",
    "пнп": "нейропатия",
    "рак": "онкология опухоль",
    "рж": "желудок",
    "ркт": "колоректальный",
}

_STOP_WORDS: frozenset[str] = frozenset({
    "ли", "не", "и", "в", "с", "у", "по", "на", "при", "для", "к", "от",
    "что", "как", "надо", "нужно", "да", "нет", "это", "или", "но", "же",
    "был", "была", "быть", "есть", "чтобы", "если", "когда", "где",
    "который", "которые", "которого", "которой", "продолжать", "продолжение",
    "можно", "нельзя", "необходимо", "следует", "рекомендуется",
})


def _tokenize_medical(text: str) -> set[str]:
    """Tokenize medical text, expand abbreviations, strip ICD-10 codes.

    IMPORTANT: abbreviation expansion is applied BEFORE the length filter so that
    2-char abbreviations like 'БА', 'ФП', 'АГ' are expanded into meaningful terms
    ('астма', 'фибрилляция', 'гипертония') before short tokens are dropped.
    """
    import re as _re
    # Capture tokens >= 2 chars so 2-letter abbreviations like БА/ФП are preserved
    raw = {w.lower() for w in _re.findall(r"[а-яёa-z]+", text, _re.IGNORECASE)
           if len(w) >= 2}
    # Strip ICD-10 codes like J45, I48, C34 (single letter + 1-2 digits)
    raw = {w for w in raw if not _re.fullmatch(r"[a-z]\d{1,2}", w, _re.I)}

    expanded: set[str] = set()
    for token in raw:
        if token in _MEDICAL_ABBREVS:
            # Known abbreviation — replace with expanded terms (no length filter needed)
            for exp in _MEDICAL_ABBREVS[token].split():
                expanded.add(exp.lower())
        elif len(token) >= 3 and token not in _STOP_WORDS:
            # Regular word — apply length and stopword filter
            expanded.add(token)
    return expanded


def _nosology_matches_question(nosology: str, question_terms: set[str]) -> bool:
    """True if any significant nosology term overlaps with question terms.

    Uses exact token matching after abbreviation expansion.
    Deliberately avoids prefix/stem matching: "беременным" (question subject)
    must NOT match "Фибрилляция предсердий у беременных" (coincidental modifier
    in an unrelated guideline) — only the PRIMARY disease tokens matter.
    """
    return bool(_tokenize_medical(nosology) & question_terms)


def _apply_nosology_filter(
    relevant_sources: dict[str, list],
    question: str,
    bypass_score: float,
) -> dict[str, list]:
    """
    Secondary filter after score-based filtering.

    Rules (all must fail to exclude a source):
    1. best_score >= bypass_score  → always keep (high-confidence hit)
    2. enrichment_ratio < 0.5     → keep (not enough metadata to decide)
    3. ANY nosology/specialty in chunks overlaps with question → keep
    4. Otherwise → exclude
    """
    import sys as _sys
    question_terms = _tokenize_medical(question)
    kept: dict[str, list] = {}

    for src_id, src_hits in relevant_sources.items():
        best_score = max(h.score for h in src_hits)

        # Rule 1: bypass
        if best_score >= bypass_score:
            kept[src_id] = src_hits
            continue

        # Rule 2: insufficient enrichment
        meta_fields = [
            (getattr(h, "nosology", None) or "") + " " + (getattr(h, "specialty", None) or "")
            for h in src_hits
        ]
        enriched_count = sum(1 for m in meta_fields if m.strip())
        if enriched_count / len(src_hits) < 0.5:
            kept[src_id] = src_hits
            continue

        # Rule 3: metadata overlap
        combined_meta = " ".join(m for m in meta_fields if m.strip())
        if _nosology_matches_question(combined_meta, question_terms):
            kept[src_id] = src_hits
        else:
            # Report what was filtered
            label = getattr(src_hits[0], "source_name", src_id)
            nosologies = list({getattr(h, "nosology", "") for h in src_hits if getattr(h, "nosology", None)})
            print(
                f"[nosology_filter] EXCLUDED '{label}' (score={best_score:.2f}) "
                f"nosologies={nosologies} | q_terms={sorted(question_terms)[:8]}",
                file=_sys.stderr,
            )

    # Safety: never return empty (fall back to unfiltered)
    return kept if kept else relevant_sources


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

        # Fetch per-document chunk stats in bulk
        # func.count(col) counts non-NULL values
        stats_rows = session.execute(
            select(
                Chunk.document_id,
                func.count(Chunk.id).label("total"),
                func.count(Chunk.embedding).label("embedded"),
                func.count(Chunk.summary).label("enriched"),
            ).group_by(Chunk.document_id)
        ).mappings().all()

        stats: dict[int, dict] = {
            r["document_id"]: {
                "total": int(r["total"] or 0),
                "missing_emb": int(r["total"] or 0) - int(r["embedded"] or 0),
                "enriched": int(r["enriched"] or 0),
            }
            for r in stats_rows
        }

        items: list[SourceItemResponse] = []
        for doc in documents:
            s = stats.get(doc.id, {"total": 0, "missing_emb": 0, "enriched": 0})
            items.append(
                SourceItemResponse(
                    document_id=doc.id,
                    source_id=doc.source_id,
                    source_name=doc.source_name,
                    title=doc.title,
                    region=doc.region,
                    year=doc.year,
                    url=doc.url,
                    chunk_count=s["total"],
                    missing_embeddings=s["missing_emb"],
                    enriched_chunks=s["enriched"],
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
def answer_route(payload: AnswerRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    import time as _time
    t0 = _time.time()
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
            source_id=payload.source_id,
        )
        ui_data = format_for_ui(result)
        elapsed_ms = int((_time.time() - t0) * 1000)

        # Build context text for LLM-judge
        from app.services.evaluator import _build_context_text, run_judge, save_query_log
        context_text = _build_context_text(ui_data.get("sources") or [])
        log_id = save_query_log(
            question=payload.question,
            mode="answer",
            answer=(ui_data.get("answer") or "")[:4000],
            context_text=context_text,
            model=payload.model,
            elapsed_ms=elapsed_ms,
        )
        background_tasks.add_task(
            run_judge, log_id, payload.question,
            ui_data.get("answer") or "", context_text,
        )
        ui_data["log_id"] = log_id
        return ui_data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {exc}") from exc


@router.post("/compare")
def compare_route(payload: CompareRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """
    Semantic retrieval → group by source → drop sources below min_score →
    LLM synthesis only over truly relevant guidelines.
    """
    import time as _time
    t0 = _time.time()
    try:
        retrieval = retrieve(
            payload.question,
            top_k=payload.top_k,
            translate_mode=payload.translate_mode,
            specialty=payload.specialty,
            nosology=payload.nosology,
        )

        hits = retrieval.hits
        if not hits:
            return {"question": payload.question, "sources": [], "comparison": "Фрагменты не найдены."}

        # Group by source, keep top max_chunks_per_source per source
        by_source: dict[str, list[Any]] = {}
        for h in hits:
            bucket = by_source.setdefault(h.source_id, [])
            if len(bucket) < payload.max_chunks_per_source:
                bucket.append(h)

        # Filter out sources where best chunk score is below threshold
        relevant_sources = {
            src_id: src_hits
            for src_id, src_hits in by_source.items()
            if max(h.score for h in src_hits) >= payload.min_score
        }

        # Secondary nosology/specialty metadata filter
        if payload.nosology_filter and relevant_sources:
            relevant_sources = _apply_nosology_filter(
                relevant_sources,
                payload.question,
                payload.nosology_bypass_score,
            )

        if not relevant_sources:
            return {
                "question": payload.question,
                "sources": [],
                "comparison": (
                    f"Ни один источник не набрал достаточного сходства "
                    f"(порог score={payload.min_score}). "
                    "Попробуйте переформулировать вопрос."
                ),
            }

        # Flatten only relevant hits for synthesis
        relevant_hits = [h for hits_list in relevant_sources.values() for h in hits_list]

        from app.services.answer import YandexOpenAIAnswerer, _normalize_sources

        answerer = YandexOpenAIAnswerer()
        sources_list = _normalize_sources(relevant_hits)

        structured = answerer.synthesize_structured(
            question=payload.question,
            context_sources=sources_list,
            model_family=payload.model,
            temperature=payload.temperature,
        )

        # Build source_labels the same way synthesize_structured does
        def _source_label(hit: Any) -> str:
            return (
                getattr(hit, "source_name", None)
                or getattr(hit, "title", None)
                or getattr(hit, "source_id", None)
                or "—"
            )

        source_rows = []
        for source_id, source_hits in relevant_sources.items():
            first = source_hits[0]
            best_score = max(h.score for h in source_hits)
            label = _source_label(first)
            text_combined = "\n---\n".join(
                (h.translated_chunk_text or h.chunk_text or "").strip()[:600]
                for h in source_hits
            )
            fragments = [
                {
                    "index": i + 1,
                    "section_title": getattr(h, "section_title", None),
                    "summary": getattr(h, "summary", None),
                    "text": (h.translated_chunk_text or h.chunk_text or "").strip()[:800],
                    "score": round(h.score, 3),
                    "evidence_level": getattr(h, "evidence_level", None),
                }
                for i, h in enumerate(source_hits)
            ]
            source_rows.append({
                "source_id": source_id,
                "source_name": first.source_name,
                "label": label,
                "title": first.title,
                "region": first.region,
                "year": first.year,
                "url": first.url,
                "chunk_count": len(source_hits),
                "combined_text": text_combined,
                "fragments": fragments,
                "score": round(best_score, 3),
            })

        # Sort by score descending
        source_rows.sort(key=lambda r: r["score"], reverse=True)

        elapsed_ms = int((_time.time() - t0) * 1000)
        answer_text = structured.get("recommendation") or structured.get("consensus") or ""

        from app.services.evaluator import _build_context_from_compare_sources, run_judge, save_query_log
        context_text = _build_context_from_compare_sources(source_rows)
        log_id = save_query_log(
            question=payload.question,
            mode="compare",
            answer=answer_text[:4000],
            context_text=context_text,
            model=payload.model,
            elapsed_ms=elapsed_ms,
        )
        background_tasks.add_task(
            run_judge, log_id, payload.question, answer_text, context_text,
        )

        return {
            "question": payload.question,
            "effective_query": retrieval.effective_query,
            "sources": source_rows,
            "positions": structured.get("positions") or [],
            "consensus": structured.get("consensus"),
            "disagreements": structured.get("disagreements") or [],
            "recommendation": structured.get("recommendation"),
            "log_id": log_id,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Compare failed: {exc}") from exc


class TriggerResponse(BaseModel):
    document_id: int
    stage: str
    status: str
    message: str


@router.post("/documents/{document_id}/embed", response_model=TriggerResponse)
def trigger_embed(document_id: int, background_tasks: BackgroundTasks) -> TriggerResponse:
    """Force-trigger embedding for a specific document (runs in background)."""
    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    def _embed() -> None:
        from app.services.embedder import embed_chunks
        try:
            stats = embed_chunks(document_id=document_id)
            print(f"[trigger] embed doc {document_id}: {stats}", file=__import__("sys").stderr)
        except Exception as exc:
            print(f"[trigger] embed error doc {document_id}: {exc}", file=__import__("sys").stderr)

    background_tasks.add_task(_embed)
    return TriggerResponse(
        document_id=document_id,
        stage="embed",
        status="started",
        message="Построение эмбеддингов запущено в фоне. Обновите статус через 30–60 сек.",
    )


@router.post("/documents/{document_id}/enrich", response_model=TriggerResponse)
def trigger_enrich(document_id: int, background_tasks: BackgroundTasks) -> TriggerResponse:
    """Force-trigger LLM enrichment for a specific document (runs in background).

    Passes only_missing=False so it retries even chunks that previously failed
    (summary == '') and marks empty-text chunks with summary='' so they are
    counted as processed and not retried forever.
    """
    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    def _enrich() -> None:
        from app.services.enricher import enrich_chunks
        from app.services.embedder import embed_chunks
        from app.services.llm import get_llm_client
        try:
            llm = get_llm_client()
            # only_missing=False: retry all, including summary="" (error-marked) chunks
            stats = enrich_chunks(document_id, only_missing=False, llm=llm)
            print(f"[trigger] enrich doc {document_id}: {stats}", file=__import__("sys").stderr)
            embed_chunks(document_id=document_id, only_missing=False)
            print(f"[trigger] re-embed after enrich doc {document_id}", file=__import__("sys").stderr)
        except Exception as exc:
            print(f"[trigger] enrich error doc {document_id}: {exc}", file=__import__("sys").stderr)

    background_tasks.add_task(_enrich)
    return TriggerResponse(
        document_id=document_id,
        stage="enrich",
        status="started",
        message="LLM-обогащение запущено в фоне (все фрагменты, включая ошибочные). Обновите статус через несколько минут.",
    )


@router.post("/documents/{document_id}/title", response_model=TriggerResponse)
def trigger_title(document_id: int, background_tasks: BackgroundTasks) -> TriggerResponse:
    """Force-regenerate the display title for a document (runs in background)."""
    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    def _title() -> None:
        from app.services.autoprocess import _generate_title
        try:
            # Temporarily bypass boring-title guard so user can force regeneration
            from app.db import session_scope as _ss
            from app.models import Document as _Doc
            with _ss() as s:
                d = s.get(_Doc, document_id)
                if d:
                    d.source_name = d.source_id  # reset to slug so _generate_title runs
            _generate_title(document_id)
        except Exception as exc:
            print(f"[trigger] title error doc {document_id}: {exc}", file=__import__("sys").stderr)

    background_tasks.add_task(_title)
    return TriggerResponse(
        document_id=document_id,
        stage="title",
        status="started",
        message="Генерация названия запущена в фоне. Обновите статус через ~15 сек.",
    )


class DeleteDocumentResponse(BaseModel):
    document_id: int
    source_id: str
    chunks_deleted: int
    file_deleted: bool
    message: str


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(document_id: int) -> DeleteDocumentResponse:
    """
    Delete a document and all its chunks from the database.
    Also removes the source file from data/raw/ if it exists.
    """
    import sys as _sys

    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

        source_id = doc.source_id
        file_path = doc.file_path

        # Count chunks before deletion
        chunk_count = session.scalar(
            select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
        ) or 0

        # Delete chunks first (FK constraint)
        session.execute(sa_delete(Chunk).where(Chunk.document_id == document_id))
        session.delete(doc)

    # Try to remove source file from disk
    file_deleted = False
    if file_path:
        fp = Path(file_path)
        if not fp.is_absolute():
            fp = Path("/app") / fp
        try:
            if fp.exists():
                fp.unlink()
                file_deleted = True
                print(f"[delete] removed file {fp}", file=_sys.stderr)
        except Exception as exc:
            print(f"[delete] could not remove file {fp}: {exc}", file=_sys.stderr)

    print(
        f"[delete] doc {document_id} ({source_id}): {chunk_count} chunks deleted, "
        f"file_deleted={file_deleted}",
        file=_sys.stderr,
    )

    return DeleteDocumentResponse(
        document_id=document_id,
        source_id=source_id,
        chunks_deleted=chunk_count,
        file_deleted=file_deleted,
        message=(
            f"Документ «{source_id}» удалён: {chunk_count} фрагментов, "
            f"файл {'удалён' if file_deleted else 'не найден или уже удалён'}."
        ),
    )


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


class UploadResponse(BaseModel):
    filename: str
    status: str          # "ingested" | "already_exists"
    document_id: int | None = None
    message: str


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> UploadResponse:
    """
    Upload a PDF guideline file.
    The file is saved to data/raw/, then:
      - ingested + embedded synchronously (available for search in ~seconds)
      - LLM-enriched asynchronously in background (quality improves over minutes)
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".html", ".htm"}:
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только файлы PDF и HTML.",
        )

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_RAW_DIR / (file.filename or "upload.pdf")

    if dest.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Файл «{file.filename}» уже существует на сервере.",
        )

    # Save file
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    # Process in background (ingest + embed sync, enrich async inside)
    def _process() -> None:
        from app.services.autoprocess import process_single_file
        process_single_file(str(dest))

    background_tasks.add_task(_process)

    return UploadResponse(
        filename=file.filename or dest.name,
        status="ingested",
        document_id=None,
        message=(
            "Файл принят. Идексация запущена в фоне: "
            "эмбеддинги готовы через ~30 сек, LLM-обогащение — через несколько минут. "
            "Статус появится в блоке «Статус индексации» внизу страницы."
        ),
    )


class FeedbackRequest(BaseModel):
    feedback: int  # 1 = 👍, -1 = 👎


@router.post("/feedback/{log_id}")
def submit_feedback(log_id: int, payload: FeedbackRequest) -> dict[str, Any]:
    """Record user thumbs-up / thumbs-down for a query log entry."""
    if payload.feedback not in (1, -1):
        raise HTTPException(status_code=400, detail="feedback must be 1 or -1")
    from app.models import QueryLog
    with session_scope() as session:
        log = session.get(QueryLog, log_id)
        if log is None:
            raise HTTPException(status_code=404, detail=f"log_id {log_id} not found")
        log.user_feedback = payload.feedback
    return {"log_id": log_id, "feedback": payload.feedback, "status": "saved"}


@router.get("/metrics")
def get_metrics(limit: int = 50) -> dict[str, Any]:
    """Return aggregated LLM-judge scores and user feedback statistics."""
    from app.models import QueryLog
    from sqlalchemy import case, cast, Float

    with session_scope() as session:
        total = session.scalar(select(func.count(QueryLog.id))) or 0
        if total == 0:
            return {
                "total_queries": 0,
                "by_mode": {},
                "avg_scores": {},
                "feedback": {"thumbs_up": 0, "thumbs_down": 0, "not_rated": 0},
                "recent": [],
            }

        # Per-mode counts
        mode_rows = session.execute(
            select(QueryLog.mode, func.count(QueryLog.id).label("n"))
            .group_by(QueryLog.mode)
        ).all()
        by_mode = {r.mode: r.n for r in mode_rows}

        # Average scores (only rows where judge ran)
        # Note: PostgreSQL round(float8, n) does not exist — cast to numeric first
        from sqlalchemy import Numeric
        scored = session.execute(
            select(
                func.round(cast(func.avg(QueryLog.score_faithfulness), Numeric), 2).label("faithfulness"),
                func.round(cast(func.avg(QueryLog.score_relevance), Numeric), 2).label("relevance"),
                func.round(cast(func.avg(QueryLog.score_completeness), Numeric), 2).label("completeness"),
                func.round(cast(func.avg(QueryLog.score_consistency), Numeric), 2).label("consistency"),
                func.count(QueryLog.score_faithfulness).label("judged_count"),
            )
            .where(QueryLog.score_faithfulness.is_not(None))
        ).one()

        avg_scores = {
            "faithfulness": scored.faithfulness,
            "relevance": scored.relevance,
            "completeness": scored.completeness,
            "consistency": scored.consistency,
            "judged_count": scored.judged_count,
        }

        # User feedback distribution
        fb_up = session.scalar(
            select(func.count(QueryLog.id)).where(QueryLog.user_feedback == 1)
        ) or 0
        fb_down = session.scalar(
            select(func.count(QueryLog.id)).where(QueryLog.user_feedback == -1)
        ) or 0
        feedback = {
            "thumbs_up": fb_up,
            "thumbs_down": fb_down,
            "not_rated": total - fb_up - fb_down,
        }

        # Recent queries
        rows = session.execute(
            select(QueryLog)
            .order_by(QueryLog.created_at.desc())
            .limit(limit)
        ).scalars().all()

        recent = [
            {
                "id": r.id,
                "question": r.question[:120],
                "mode": r.mode,
                "model": r.model,
                "elapsed_ms": r.elapsed_ms,
                "score_faithfulness": r.score_faithfulness,
                "score_relevance": r.score_relevance,
                "score_completeness": r.score_completeness,
                "score_consistency": r.score_consistency,
                "judge_reasoning": r.judge_reasoning,
                "user_feedback": r.user_feedback,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]

    return {
        "total_queries": total,
        "by_mode": by_mode,
        "avg_scores": avg_scores,
        "feedback": feedback,
        "recent": recent,
    }


def _generate_summary_task(document_id: int, model_family: str) -> None:
    try:
        from app.services.summarizer import generate_document_summary
        from app.services.llm import get_llm_client
        llm = get_llm_client(model_family=model_family)
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
