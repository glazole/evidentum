from __future__ import annotations

import re
import sys
import threading
from pathlib import Path

from sqlalchemy import func, select

DATA_RAW_DIR = Path("/app/data/raw")


def _fmt_source_id(source_id: str) -> str:
    """Turn 'kr_rf_af_2024' into something like 'КР РФ ФП 2024'."""
    return source_id.replace("_", " ").upper()


def _norm_slug(value: str) -> str:
    """Compare LLM output to source_id without spaces/underscore noise."""
    return re.sub(r"[^a-zа-яё0-9]+", "", (value or "").lower())


def _is_boring_title(title: str | None) -> bool:
    """Return True if the title extracted from PDF is uninformative."""
    import re as _re
    if not title:
        return True
    t = title.strip().lower()
    boring_exact = {"оглавление", "содержание", "circulation", "untitled", "title"}
    if t in boring_exact or len(t) < 6:
        return True
    # File slugs / source_id-like strings: no spaces, only alphanumeric + hyphens/underscores/dots
    if _re.fullmatch(r"[a-z0-9а-яёa-z\-_\.]+", t):
        return True
    return False


def _llm_title_unacceptable(candidate: str | None, source_id: str) -> bool:
    """True if the model echoed the slug or returned another non-human label."""
    if not candidate or not str(candidate).strip():
        return True
    c = str(candidate).strip()
    if _norm_slug(c) == _norm_slug(source_id):
        return True
    return _is_boring_title(c)


def _build_title_context(session, document_id: int, doc) -> str:
    """Rich text for naming: PDF title, year, document summary, chunks from several positions."""
    from app.models import Chunk

    parts: list[str] = []
    if (doc.title or "").strip():
        parts.append(f"Заголовок/первая строка из PDF: {doc.title.strip()}")
    if doc.year:
        parts.append(f"Год (эвристика): {doc.year}")
    if (doc.region or "").strip():
        parts.append(f"Регион: {doc.region}")
    if (doc.nosology_primary or "").strip():
        parts.append(f"Нозология (метаданные): {doc.nosology_primary}")
    if (doc.summary_ru or "").strip():
        parts.append(f"Резюме документа (если есть):\n{(doc.summary_ru or '')[:2800]}")

    n_chunks = session.scalar(
        select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
    ) or 0
    if n_chunks == 0 and (doc.raw_text or "").strip():
        parts.append(f"Начало текста документа:\n{(doc.raw_text or '')[:3500]}")
        return "\n\n".join(parts)

    raw_indices = [0, 1, 2, n_chunks // 6, n_chunks // 3, n_chunks // 2, n_chunks - 1]
    idx_set: set[int] = set()
    for i in raw_indices:
        if n_chunks <= 0:
            break
        idx_set.add(max(0, min(n_chunks - 1, i)))
    indices = sorted(idx_set)

    chunks = list(
        session.scalars(
            select(Chunk)
            .where(Chunk.document_id == document_id, Chunk.chunk_index.in_(indices))
            .order_by(Chunk.chunk_index)
        ).all()
    )

    frag_parts: list[str] = []
    for ch in chunks:
        sec = (ch.section_title or "").strip()
        summ = (ch.summary or "").strip()
        body = summ if summ else (ch.chunk_text or "")
        body = body.strip()
        if not body:
            continue
        head = f"--- Фрагмент #{ch.chunk_index}"
        if sec:
            head += f" ({sec})"
        frag_parts.append(f"{head}\n{body[:900]}")

    if frag_parts:
        parts.append("Фрагменты из разных частей документа (после обогащения — с краткими резюме):\n" + "\n\n".join(frag_parts))

    return "\n\n".join(parts)


def _pick_fallback_display_name(doc) -> str:
    """When LLM returns only a slug, use PDF title, first body line, or spaced source_id."""
    t = (doc.title or "").strip()
    if t and not _is_boring_title(t):
        return t[:256]
    raw = (doc.raw_text or "").strip()
    if raw:
        # First substantial line from body (skip very short / TOC-like)
        for line in raw.splitlines():
            s = line.strip()
            if len(s) >= 24 and s.lower() not in {"оглавление", "содержание"}:
                return s[:256]
    return _fmt_source_id(doc.source_id)[:256]


def _generate_title(document_id: int, *, force: bool = False) -> None:
    """Use LLM to generate a human-readable display title for a document.

    When *force* is False, skips documents whose source_name already looks
    informative (slug-only filenames, generic TOC lines, etc.). Upload and the
    «Title» API use force=True so typical «Download (1).pdf» stems with spaces
    still get a proper guideline-style name.
    """
    try:
        from app.db import session_scope
        from app.models import Document
        from app.services.llm import get_llm_client

        with session_scope() as s:
            doc = s.get(Document, document_id)
            if doc is None:
                return
            if not force and not _is_boring_title(doc.source_name):
                return
            context = _build_title_context(s, document_id, doc)

        llm = get_llm_client()
        sid = doc.source_id
        base_user = (
            f"Технический идентификатор в системе (НЕ используй его как ответ): {sid!r}\n\n"
            f"{context}\n\n"
            "Придумай ОДНУ строку — отображаемое название источника для списка клинических гайдлайнов на русском.\n"
            "Формат по возможности: «организация или тип документа, год (если он явно есть в тексте выше): нозология».\n"
            "Требования:\n"
            "- ответ не должен совпадать с техническим идентификатором и не должен быть только латиницей/цифрами/slug;\n"
            "- в ответе должны быть обычные русские слова (тема, орган, диагноз, «клинические рекомендации» и т.д.);\n"
            "- не выдумывай год или организацию, если их нет в материале выше;\n"
            "- тогда опиши тему нейтрально по смыслу текста (например: «Клинические рекомендации: …»).\n"
            "Только одна строка названия, без кавычек и пояснений."
        )
        title = llm.complete(
            [{"role": "user", "content": base_user}],
            max_tokens=120,
            temperature=0.15,
        ).strip().strip('"').strip("'")

        if _llm_title_unacceptable(title, sid):
            retry_user = (
                f"Технический идентификатор (запрещено повторять как ответ): {sid!r}\n\n"
                f"{context[:12000]}\n\n"
                "Предыдущая попытка вернула бесполезный ярлык. Сформулируй заново: короткое (до 120 символов) "
                "читаемое название на русском для каталога гайдлайнов. Опирайся на «Заголовок/первая строка из PDF» "
                "и фрагменты. Нельзя отвечать slug'ом или только номером документа. Одна строка, без кавычек."
            )
            title = llm.complete(
                [{"role": "user", "content": retry_user}],
                max_tokens=120,
                temperature=0.25,
            ).strip().strip('"').strip("'")

        if _llm_title_unacceptable(title, sid):
            with session_scope() as s:
                doc2 = s.get(Document, document_id)
                if doc2 is None:
                    return
                fallback = _pick_fallback_display_name(doc2)
                if fallback:
                    doc2.source_name = fallback
                    print(
                        f"[autoprocess] title fallback (no good LLM line) doc {document_id}: {fallback!r}",
                        file=sys.stderr,
                    )
                return

        with session_scope() as s:
            doc3 = s.get(Document, document_id)
            if doc3:
                doc3.source_name = title[:256]
        print(f"[autoprocess] generated title for doc {document_id}: {title!r}", file=sys.stderr)
    except Exception as exc:
        print(f"[autoprocess] title gen error doc {document_id}: {exc}", file=sys.stderr)


def process_single_file(file_path: str) -> dict:
    """
    Ingest, embed, generate title and start background enrichment for one file.
    Returns {"document_id": int, "status": "ingested"|"already_exists", "filename": str}.
    Called immediately after a user uploads a file via the API.
    """
    from app.services.ingest import ingest_file
    from app.services.embedder import embed_chunks

    result = ingest_file(file_path)
    document_id: int | None = result.get("document_id")

    if not document_id:
        return {"filename": Path(file_path).name, "status": "already_exists", "document_id": None}

    # Synchronous: embed right away so search is available quickly
    try:
        embed_chunks(document_id=document_id)
    except Exception as exc:
        print(f"[autoprocess] embed error for {file_path}: {exc}", file=sys.stderr)

    # Synchronous: generate readable title (fast single LLM call)
    _generate_title(document_id, force=True)

    # Async: enrichment is slow — run in background thread
    def _enrich() -> None:
        try:
            from app.services.enricher import enrich_chunks
            from app.services.embedder import embed_chunks as _embed
            from app.services.llm import get_llm_client
            llm = get_llm_client()
            enrich_chunks(document_id, llm=llm)
            _embed(document_id=document_id, only_missing=False)
            print(f"[autoprocess] enrich+re-embed done for doc {document_id}", file=sys.stderr)
        except Exception as exc:
            print(f"[autoprocess] enrich error for doc {document_id}: {exc}", file=sys.stderr)

    t = threading.Thread(target=_enrich, daemon=True, name=f"enrich-upload-{document_id}")
    t.start()

    return {"filename": Path(file_path).name, "status": "ingested", "document_id": document_id}


def run_autoprocess(data_dir: Path = DATA_RAW_DIR) -> None:
    """
    1. Ingest any new PDF/HTML files in data_dir that are not yet in the DB.
    2. Embed any documents that have chunks without embeddings.
    3. Enrich unenriched chunks in the background.
    4. Generate better titles for documents with uninformative names.
    """
    print("[autoprocess] starting…", file=sys.stderr)

    # ── 1. Ingest new files ───────────────────────────────────────────────────
    if data_dir.exists():
        from app.services.ingest import ingest_file

        for f in sorted(data_dir.iterdir()):
            if f.suffix.lower() not in {".pdf", ".html", ".htm"}:
                continue
            try:
                result = ingest_file(str(f))
                status = "ingested" if result.get("inserted") else "already in DB"
                print(f"[autoprocess] {f.name}: {status}", file=sys.stderr)
            except Exception as exc:
                print(f"[autoprocess] ingest error {f.name}: {exc}", file=sys.stderr)
    else:
        print(f"[autoprocess] {data_dir} not found, skipping ingest", file=sys.stderr)

    # ── 2. Embed missing ──────────────────────────────────────────────────────
    from app.services.embedder import embed_chunks, list_documents

    for doc in list_documents():
        if doc["missing_embeddings"] > 0:
            print(
                f"[autoprocess] embedding doc {doc['document_id']} "
                f"({doc['missing_embeddings']} missing)…",
                file=sys.stderr,
            )
            try:
                stats = embed_chunks(document_id=doc["document_id"])
                print(f"[autoprocess] embedded doc {doc['document_id']}: {stats}", file=sys.stderr)
            except Exception as exc:
                print(f"[autoprocess] embed error doc {doc['document_id']}: {exc}", file=sys.stderr)

    # ── 3. Generate titles for boring names ───────────────────────────────────
    for doc in list_documents():
        from app.db import session_scope
        from app.models import Document

        with session_scope() as s:
            d = s.get(Document, doc["document_id"])
            if d and _is_boring_title(d.source_name):
                _generate_title(doc["document_id"])

    # ── 4. Enrich unenriched chunks (background thread) ───────────────────────
    def _enrich_background() -> None:
        try:
            from sqlalchemy import select, func
            from app.db import session_scope
            from app.models import Chunk
            from app.services.enricher import enrich_chunks
            from app.services.llm import get_llm_client

            with session_scope() as s:
                rows = s.execute(
                    select(Chunk.document_id, func.count(Chunk.id).label("n"))
                    .where(Chunk.summary.is_(None))
                    .group_by(Chunk.document_id)
                ).all()
                doc_ids = [r.document_id for r in rows]

            if not doc_ids:
                print("[autoprocess] all chunks already enriched", file=sys.stderr)
                return

            llm = get_llm_client()
            for doc_id in doc_ids:
                print(f"[autoprocess] enriching doc {doc_id}…", file=sys.stderr)
                try:
                    stats = enrich_chunks(doc_id, llm=llm)
                    print(f"[autoprocess] enriched doc {doc_id}: {stats}", file=sys.stderr)

                    # Re-embed after enrichment (summary now available)
                    from app.services.embedder import embed_chunks as _embed
                    _embed(document_id=doc_id, only_missing=False)
                    print(f"[autoprocess] re-embedded doc {doc_id} after enrichment", file=sys.stderr)
                except Exception as exc:
                    print(f"[autoprocess] enrich error doc {doc_id}: {exc}", file=sys.stderr)

        except Exception as exc:
            print(f"[autoprocess] enrich background error: {exc}", file=sys.stderr)

    t = threading.Thread(target=_enrich_background, daemon=True, name="enrich-bg")
    t.start()
    print("[autoprocess] embed done; enrich running in background", file=sys.stderr)
