from __future__ import annotations

import argparse
from typing import Any, Optional

from sqlalchemy import select

from app.db import session_scope
from app.models import Chunk, Document
from app.services.llm import BaseLLMClient, get_llm_client


SUMMARIZE_SYSTEM_PROMPT = (
    "Ты медицинский редактор. Составляй краткие клинические резюме на русском языке. "
    "Отвечай только запрошенным текстом без преамбул."
)

SUMMARIZE_USER_TEMPLATE = """\
Ниже — краткие резюме разделов гайдлайна "{source_name}" ({year}).

{section_summaries}

Составь TL;DR этого гайдлайна: 8–12 чётких буллетов с ключевыми клиническими положениями.
Включай пункт только если соответствующая рекомендация явно сформулирована в приведённых резюме.
Не выводи рекомендацию, если она лишь подразумевается или следует из описательного контекста.
Используй формат:
• [положение из гайдлайна]
"""

DELTA_USER_TEMPLATE = """\
Ниже — эвристическое сравнение двух версий гайдлайна "{source_name}" на основе их резюме.
Это не полное сравнение исходных документов: сравниваются только сформированные резюме.

Старая версия ({old_year}):
{old_summary}

Новая версия ({new_year}):
{new_summary}

Опиши ключевые изменения в 5–8 буллетах. Если изменений мало — напиши "Существенных изменений не выявлено".
Не делай выводов об изменениях, если они явно не следуют из резюме.
Формат:
• [изменение]
"""


def _collect_section_summaries(document_id: int, session: Any) -> str:
    chunks = list(
        session.scalars(
            select(Chunk)
            .where(Chunk.document_id == document_id, Chunk.summary.is_not(None))
            .order_by(Chunk.chunk_index)
        )
    )
    if not chunks:
        return ""

    parts: list[str] = []
    current_section: Optional[str] = None
    for chunk in chunks:
        if chunk.section_title and chunk.section_title != current_section:
            current_section = chunk.section_title
            parts.append(f"### {current_section}")
        parts.append(f"- {chunk.summary}")
    return "\n".join(parts)


def generate_document_summary(
    document_id: int,
    *,
    llm: BaseLLMClient | None = None,
    debug: bool = False,
) -> Document:
    """
    Generate summary_ru (TL;DR) and version_delta_ru for a document.
    Saves results to DB and returns the updated Document.
    """
    if llm is None:
        llm = get_llm_client(debug=debug)

    with session_scope() as session:
        doc = session.scalar(select(Document).where(Document.id == document_id))
        if doc is None:
            raise ValueError(f"Document id={document_id} not found.")

        section_summaries = _collect_section_summaries(document_id, session)
        if not section_summaries:
            # Fallback: use raw_text snippet if no chunk summaries yet
            raw = (doc.raw_text or "")[:8000]
            section_summaries = raw

        year_str = str(doc.year) if doc.year else "н/д"
        prompt = SUMMARIZE_USER_TEMPLATE.format(
            source_name=doc.source_name,
            year=year_str,
            section_summaries=section_summaries[:12000],
        )

        try:
            summary_ru = llm.complete(
                [{"role": "system", "text": SUMMARIZE_SYSTEM_PROMPT}, {"role": "user", "text": prompt}]
            )
            doc.summary_ru = summary_ru
        except Exception as exc:
            if debug:
                print(f"[summarizer] summary failed: {exc}")

        # Delta vs previous version
        if doc.previous_document_id is not None:
            prev = session.scalar(select(Document).where(Document.id == doc.previous_document_id))
            if prev is not None and prev.summary_ru and doc.summary_ru:
                delta_prompt = DELTA_USER_TEMPLATE.format(
                    source_name=doc.source_name,
                    old_year=str(prev.year) if prev.year else "н/д",
                    old_summary=prev.summary_ru[:4000],
                    new_year=year_str,
                    new_summary=doc.summary_ru[:4000],
                )
                try:
                    delta = llm.complete(
                        [
                            {"role": "system", "text": SUMMARIZE_SYSTEM_PROMPT},
                            {"role": "user", "text": delta_prompt},
                        ]
                    )
                    doc.version_delta_ru = delta
                except Exception as exc:
                    if debug:
                        print(f"[summarizer] delta failed: {exc}")

        session.flush()
        session.expunge(doc)
        return doc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TL;DR summary (and version delta) for a document")
    parser.add_argument("--document-id", type=int, required=True, help="Document ID to summarize.")
    parser.add_argument("--debug", action="store_true", help="Debug LLM calls.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    llm = get_llm_client(debug=args.debug)
    doc = generate_document_summary(args.document_id, llm=llm, debug=args.debug)
    print(f"document_id={doc.id}")
    print(f"summary_ru={doc.summary_ru[:300] if doc.summary_ru else None}")
    print(f"version_delta_ru={doc.version_delta_ru[:200] if doc.version_delta_ru else None}")


if __name__ == "__main__":
    main()
