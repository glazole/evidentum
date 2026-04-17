from __future__ import annotations

import sys
import threading
from pathlib import Path

DATA_RAW_DIR = Path("/app/data/raw")


def _fmt_source_id(source_id: str) -> str:
    """Turn 'kr_rf_af_2024' into something like 'КР РФ ФП 2024'."""
    return source_id.replace("_", " ").upper()


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


def _generate_title(document_id: int) -> None:
    """Use LLM to generate a human-readable display title for a document."""
    try:
        from sqlalchemy import select
        from app.db import session_scope
        from app.models import Chunk, Document
        from app.services.llm import YandexLLMClient

        with session_scope() as s:
            doc = s.get(Document, document_id)
            if doc is None:
                return
            if not _is_boring_title(doc.source_name):
                return

            samples = s.scalars(
                select(Chunk.chunk_text)
                .where(Chunk.document_id == document_id)
                .order_by(Chunk.chunk_index)
                .limit(5)
            ).all()
            context = "\n\n".join(samples[:3])[:2000]

        llm = YandexLLMClient()
        prompt = (
            f"source_id: {doc.source_id}\n\n"
            f"Первые фрагменты документа:\n{context}\n\n"
            "Определи точное название медицинского гайдлайна (организация, год, нозология) "
            "и верни одну строку — читаемое название на русском языке, например: "
            "'ESC 2024: Фибрилляция предсердий' или 'КР МЗ РФ 2024: Фибрилляция предсердий'. "
            "Только название, без пояснений."
        )
        title = llm.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.1,
        ).strip().strip('"').strip("'")

        if title:
            with session_scope() as s:
                doc = s.get(Document, document_id)
                if doc:
                    doc.source_name = title[:256]
            print(f"[autoprocess] generated title for doc {document_id}: {title!r}", file=sys.stderr)
    except Exception as exc:
        print(f"[autoprocess] title gen error doc {document_id}: {exc}", file=sys.stderr)


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
        from sqlalchemy import select
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
            from app.services.llm import YandexLLMClient

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

            llm = YandexLLMClient()
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
