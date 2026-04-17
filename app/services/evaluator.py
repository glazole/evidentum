"""LLM-as-judge evaluation service.

Runs asynchronously after each user query (via FastAPI BackgroundTasks).
Scores the answer on four criteria (1-5) and persists results to query_logs.
"""
from __future__ import annotations

import sys
from typing import Any

from app.services.llm import get_llm_client


JUDGE_SYSTEM_PROMPT = (
    "Ты эксперт по оценке качества ответов медицинской ИИ-системы, "
    "работающей с клиническими рекомендациями. "
    "Отвечай ТОЛЬКО валидным JSON без пояснений и без маркдаун-разметки."
)

JUDGE_USER_TEMPLATE = """\
Оцени качество ответа медицинской ИИ-системы по клиническим рекомендациям.

Вопрос пользователя:
{question}

Контекст (фрагменты из гайдлайнов, на основе которых строился ответ):
{context}

Ответ системы:
{answer}

Верни JSON с оценками от 1 до 5 по каждому критерию:
{{
  "faithfulness": <int 1-5>,
  "relevance": <int 1-5>,
  "completeness": <int 1-5>,
  "consistency": <int 1-5>,
  "reasoning": "<1-2 предложения: краткое обоснование оценок>"
}}

Критерии:
- faithfulness (верность): ответ основан только на предоставленном контексте, без домысливания
- relevance (релевантность): ответ отвечает именно на заданный вопрос
- completeness (полнота): учтены все важные аспекты из контекста
- consistency (согласованность): нет внутренних противоречий в ответе
"""


def _build_context_text(sources: list[dict[str, Any]]) -> str:
    """Build a compact text representation of retrieved chunks for the judge."""
    parts: list[str] = []
    for src in sources[:6]:
        name = src.get("source_name") or src.get("title") or src.get("source_id") or "?"
        text = (
            src.get("translated_chunk_text")
            or src.get("chunk_text")
            or ""
        ).strip()
        summary = (src.get("summary") or "").strip()
        section = src.get("section_title") or ""
        part = f"[{name}]"
        if section:
            part += f" {section}"
        if summary:
            part += f"\nРезюме: {summary}"
        if text:
            part += f"\n{text[:400]}"
        parts.append(part)
    return "\n\n".join(parts)


def _build_context_from_compare_sources(sources: list[dict[str, Any]]) -> str:
    """Build context from compare-mode source_rows."""
    parts: list[str] = []
    for src in sources[:5]:
        label = src.get("label") or src.get("source_name") or "?"
        frags = src.get("fragments") or []
        texts = [
            (f.get("summary") or f.get("text") or "")[:300]
            for f in frags[:2]
        ]
        parts.append(f"[{label}]\n" + "\n".join(texts))
    return "\n\n".join(parts)


def run_judge(log_id: int, question: str, answer: str, context_text: str) -> None:
    """
    Call the LLM judge and persist scores to query_logs.
    Designed to run in a FastAPI BackgroundTask — never raises.
    """
    try:
        llm = get_llm_client()
        prompt = JUDGE_USER_TEMPLATE.format(
            question=question[:500],
            answer=answer[:2000],
            context=context_text[:3000],
        )
        scores = llm.complete_json(
            prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=400,
        )

        from sqlalchemy import select
        from app.db import session_scope
        from app.models import QueryLog

        with session_scope() as session:
            log = session.get(QueryLog, log_id)
            if log is None:
                return
            log.score_faithfulness = _clamp(scores.get("faithfulness"))
            log.score_relevance = _clamp(scores.get("relevance"))
            log.score_completeness = _clamp(scores.get("completeness"))
            log.score_consistency = _clamp(scores.get("consistency"))
            log.judge_reasoning = str(scores.get("reasoning") or "")[:512]

        print(f"[evaluator] log_id={log_id} scored: {scores}", file=sys.stderr)

    except Exception as exc:
        print(f"[evaluator] judge failed for log_id={log_id}: {exc}", file=sys.stderr)


def _clamp(value: Any) -> float | None:
    """Convert LLM score to float in [1.0, 5.0] or None."""
    try:
        v = float(value)
        return max(1.0, min(5.0, v))
    except (TypeError, ValueError):
        return None


def save_query_log(
    *,
    question: str,
    mode: str,
    answer: str,
    context_text: str,
    model: str | None,
    elapsed_ms: int | None,
) -> int:
    """
    Persist a query to query_logs and return the new log_id.
    Called synchronously inside the API route (fast — just a DB insert).
    """
    from app.db import session_scope
    from app.models import QueryLog

    with session_scope() as session:
        log = QueryLog(
            question=question[:1000],
            mode=mode,
            answer=answer[:4000],
            context_text=context_text[:4000],
            model=model,
            elapsed_ms=elapsed_ms,
        )
        session.add(log)
        session.flush()
        log_id = log.id
    return log_id
