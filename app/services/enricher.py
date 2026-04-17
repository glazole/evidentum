from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select

from app.db import session_scope
from app.models import Chunk, Document
from app.services.llm import BaseLLMClient, get_llm_client


ENRICH_SYSTEM_PROMPT = (
    "Ты медицинский редактор. Отвечай ТОЛЬКО валидным JSON без пояснений и без маркдаун-разметки."
)

ENRICH_USER_TEMPLATE = """\
Это фрагмент из раздела "{section_title}" гайдлайна "{source_name}" ({year}).

Верни строгий JSON (без других слов) с ключами:
  "summary"       — 2-3 предложения на русском, клиническое резюме фрагмента (сохраняй медицинские термины точно),
  "nosology"      — одно основное заболевание или синдром (можно с кодом МКБ-10, например "Фибрилляция предсердий I48"),
  "specialty"     — одна врачебная специальность (например "кардиология", "неврология"),
  "topic"         — одно из: therapy | diagnosis | prevention | follow-up | general,
  "evidence_level"— если явно упоминается уровень доказательности (1A, 1B, 2A, 2B, 3, 4, 5) — укажи строкой, иначе null.

Текст фрагмента:
{chunk_text}
"""


@dataclass
class EnrichStats:
    document_id: int
    selected: int = 0
    enriched: int = 0
    skipped: int = 0
    errors: int = 0


def _build_prompt(chunk: Chunk, source_name: str, year: Optional[int]) -> str:
    section = chunk.section_title or "Основной текст"
    year_str = str(year) if year else "н/д"
    return ENRICH_USER_TEMPLATE.format(
        section_title=section,
        source_name=source_name,
        year=year_str,
        chunk_text=(chunk.chunk_text or "")[:3000],
    )


def enrich_chunks(
    document_id: int,
    *,
    only_missing: bool = True,
    batch_size: int = 4,
    sleep_between_batches: float = 0.5,
    llm: BaseLLMClient | None = None,
    debug: bool = False,
) -> EnrichStats:
    """
    Generate per-chunk summary + metadata fields via YandexGPT.
    Writes summary, nosology, specialty, topic, evidence_level to each Chunk.
    """
    if llm is None:
        llm = get_llm_client(debug=debug)

    stats = EnrichStats(document_id=document_id)

    # Fetch document metadata and chunk IDs in a short-lived transaction.
    with session_scope() as session:
        document = session.scalar(select(Document).where(Document.id == document_id))
        if document is None:
            raise ValueError(f"Document id={document_id} not found.")

        stmt = select(Chunk.id).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
        if only_missing:
            stmt = stmt.where(Chunk.summary.is_(None))

        chunk_ids: list[int] = list(session.scalars(stmt))
        source_name = document.source_name
        year = document.year

    stats.selected = len(chunk_ids)
    total = len(chunk_ids)
    print(f"[enrich] document_id={document_id}, chunks to process: {total}", flush=True)

    if not chunk_ids:
        return stats

    # Process each batch in its own short transaction so the DB connection
    # is never idle for more than one LLM round-trip (avoids TCP timeouts).
    for start in range(0, total, batch_size):
        batch_ids = chunk_ids[start : start + batch_size]

        with session_scope() as session:
            batch = list(session.scalars(select(Chunk).where(Chunk.id.in_(batch_ids)).order_by(Chunk.chunk_index)))

            for chunk in batch:
                idx = start + batch.index(chunk) + 1
                print(f"[enrich] {idx}/{total} chunk_id={chunk.id} ...", flush=True)
                text = (chunk.chunk_text or "").strip()
                if not text:
                    stats.skipped += 1
                    print(f"[enrich] {idx}/{total} chunk_id={chunk.id} SKIPPED (empty)", flush=True)
                    continue
                try:
                    prompt = _build_prompt(chunk, source_name, year)
                    data = llm.complete_json(prompt, system_prompt=ENRICH_SYSTEM_PROMPT)

                    chunk.summary = str(data.get("summary") or "").strip() or None
                    chunk.nosology = str(data.get("nosology") or "").strip()[:256] or None
                    chunk.specialty = str(data.get("specialty") or "").strip()[:128] or None
                    chunk.topic = str(data.get("topic") or "").strip()[:128] or None
                    ev = data.get("evidence_level")
                    chunk.evidence_level = str(ev).strip()[:32] if ev else None

                    stats.enriched += 1
                    print(f"[enrich] {idx}/{total} chunk_id={chunk.id} OK", flush=True)
                except Exception as exc:
                    stats.errors += 1
                    print(f"[enrich] {idx}/{total} chunk_id={chunk.id} ERROR: {exc}", flush=True)
                    # Mark as attempted so this chunk is not retried forever on next run.
                    # Empty string is falsy → LLM synthesis skips it gracefully.
                    chunk.summary = ""

        # commit happens on session_scope exit; sleep outside the transaction
        if start + batch_size < total:
            time.sleep(sleep_between_batches)

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enrich chunks with LLM-generated summary and metadata")
    parser.add_argument("--document-id", type=int, required=True, help="Document ID to enrich.")
    parser.add_argument(
        "--reenrich",
        action="store_true",
        help="Re-enrich chunks that already have summary.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Chunks per loop iteration.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between batches.")
    parser.add_argument("--debug", action="store_true", help="Print LLM raw responses.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    llm = get_llm_client(debug=args.debug)
    stats = enrich_chunks(
        document_id=args.document_id,
        only_missing=not args.reenrich,
        batch_size=args.batch_size,
        sleep_between_batches=args.sleep,
        llm=llm,
        debug=args.debug,
    )
    print(
        {
            "document_id": stats.document_id,
            "selected": stats.selected,
            "enriched": stats.enriched,
            "skipped": stats.skipped,
            "errors": stats.errors,
        }
    )


if __name__ == "__main__":
    main()
