from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

try:
    from app.services.retriever import RetrievalResult, retrieve
except Exception:  # pragma: no cover
    retrieve = None  # type: ignore
    RetrievalResult = None  # type: ignore

from app.services.llm import BaseLLMClient, get_llm_client


DEFAULT_MODEL_FAMILY = os.getenv("ANSWER_MODEL_FAMILY", "alice")
DEFAULT_TEMPERATURE = float(os.getenv("ANSWER_TEMPERATURE", "0.2"))
DEFAULT_TOP_K = int(os.getenv("ANSWER_TOP_K", "6"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("ANSWER_MAX_OUTPUT_TOKENS", "1800"))
DEFAULT_TRANSLATE_MODE = os.getenv("ANSWER_TRANSLATE_MODE", "dual_query")


@dataclass
class SourceItem:
    index: int
    source_id: str | None
    source_name: str | None  # human-readable document name (from Document.source_name)
    title: str | None
    section_title: str | None
    url: str | None
    score: float | None
    chunk_text: str
    translated_chunk_text: str | None = None
    summary: str | None = None
    nosology: str | None = None
    specialty: str | None = None
    evidence_level: str | None = None


@dataclass
class AnswerResult:
    question: str
    effective_query: str | None
    model: str
    answer: str
    disclaimer: str
    sources: list[SourceItem]
    raw_completion: dict[str, Any] | None = None
    # Structured synthesis fields (filled when synthesize=True)
    consensus: str | None = None
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    recommendation: str | None = None


class YandexOpenAIAnswerer:
    """
    LLM-based answerer for clinical guideline questions.

    Provider-agnostic: delegates all LLM calls to a BaseLLMClient instance.
    By default creates a client via get_llm_client() which respects LLM_PROVIDER
    env var (yandex | openai | anthropic). Pass ``llm=`` to override.
    """

    def __init__(
        self,
        *,
        default_model_family: str = DEFAULT_MODEL_FAMILY,
        llm: BaseLLMClient | None = None,
    ) -> None:
        self.default_model_family = default_model_family
        self.llm = llm if llm is not None else get_llm_client()

    def build_model_uri(self, model_family: str | None = None) -> str:
        """Return provider-canonical model name/URI for the given alias."""
        return self.llm.resolve_model(model_family or self.default_model_family)

    def generate(
        self,
        *,
        question: str,
        context_sources: list[SourceItem],
        model_family: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        return_raw: bool = False,
    ) -> tuple[str, str, dict[str, Any] | None]:
        model_uri = self.build_model_uri(model_family)

        system_prompt = _load_system_prompt()

        context_parts: list[str] = []
        for src in context_sources:
            snippet = (src.translated_chunk_text or src.chunk_text or "").strip()
            if len(snippet) > 1800:
                snippet = snippet[:1800].rstrip() + " ..."
            summary_line = f"summary={src.summary}\n" if src.summary else ""
            evidence_line = f"evidence_level={src.evidence_level}\n" if src.evidence_level else ""
            context_parts.append(
                "\n".join(
                    filter(None, [
                        f"[{src.index}] source_id={src.source_id or '-'}",
                        f"title={src.title or '-'}",
                        f"section={src.section_title or '-'}",
                        f"url={src.url or '-'}",
                        f"score={src.score if src.score is not None else '-'}",
                        summary_line.rstrip() or None,
                        evidence_line.rstrip() or None,
                        f"fragment={snippet}",
                    ])
                )
            )

        user_prompt = (
            f"Вопрос пользователя: {question}\n\n"
            "Ниже контекст из retriever:\n\n"
            + "\n\n".join(context_parts)
        )

        answer = self.llm.complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model_family,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        return answer, model_uri, None

    def synthesize_structured(
        self,
        *,
        question: str,
        context_sources: list[SourceItem],
        model_family: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> dict[str, Any]:
        """
        Generate a structured Консенсус/Расхождения/Рекомендация/Источники response.
        Returns a JSON dict. Falls back to empty dict on failure.
        """
        model_uri = self.build_model_uri(model_family)

        system_prompt = (
            "Ты клинический ассистент. Отвечай ТОЛЬКО на русском языке. "
            "Сохраняй латинские МНН-названия препаратов. "
            "Используй ТОЛЬКО информацию из предоставленных фрагментов. "
            "Если данных мало — скажи явно.\n\n"
            "ВАЖНО: во всех полях используй ПОЛНОЕ НАЗВАНИЕ гайдлайна из заголовка "
            "'=== ГАЙДЛАЙН: ... ===' — НЕ номера фрагментов.\n\n"
            "Верни строгий JSON без маркдаун-разметки:\n"
            '{"positions": [{"source": "Точное название гайдлайна из заголовка", '
            '"text": "2-4 предложения: позиция этого гайдлайна по заданному вопросу"}], '
            '"consensus": "Что рекомендуют все или большинство гайдлайнов, или null если консенсуса нет", '
            '"disagreements": [{"sources": ["Название 1", "Название 2"], "text": "Суть расхождения"}], '
            '"recommendation": "Практическая заметка для врача — ТОЛЬКО на основе явно совпадающих фрагментов из разных источников; если данных недостаточно или источники расходятся — напиши об этом прямо (2-4 предложения)"}'
        )

        # Group context by guideline (source_id), build name map
        grouped: dict[str, list[SourceItem]] = {}
        source_labels: dict[str, str] = {}
        for src in context_sources:
            key = src.source_id or "unknown"
            grouped.setdefault(key, []).append(src)
            if key not in source_labels:
                # Prefer human-readable source_name over PDF title
                source_labels[key] = src.source_name or src.title or src.source_id or key

        # Preamble: explicit mapping so LLM knows names
        name_map_lines = [
            f"  {label}" for label in source_labels.values()
        ]
        context_parts: list[str] = [
            "Гайдлайны в контексте:\n" + "\n".join(name_map_lines)
        ]

        for source_id, sources in grouped.items():
            label = source_labels[source_id]
            context_parts.append(f"=== ГАЙДЛАЙН: {label} ===")
            for src in sources:
                text = (src.translated_chunk_text or src.chunk_text or "").strip()
                summary = src.summary or ""
                section = src.section_title or ""
                evidence = f" [Уровень: {src.evidence_level}]" if src.evidence_level else ""
                context_parts.append(
                    f"[{src.index}] Раздел: {section}{evidence}\n"
                    + (f"Резюме: {summary}\n" if summary else "")
                    + f"Текст: {text[:1500]}"
                )

        user_prompt = (
            f"Вопрос врача: {question}\n\n"
            "Фрагменты из клинических рекомендаций:\n\n"
            + "\n\n".join(context_parts)
            + "\n\nВерни JSON-ответ строго по схеме. "
            "В поле sources пиши название гайдлайна из заголовка '=== ГАЙДЛАЙН: ... ===', не номер фрагмента."
        )

        try:
            return self.llm.complete_json(
                user_prompt,
                system_prompt=system_prompt,
                model=model_family,
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
        except Exception:
            return {}


def _load_system_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "answer_system.txt")
    try:
        with open(prompt_path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return (
            "Ты медицинский ассистент для MVP по клиническим рекомендациям. "
            "Отвечай только на основе переданных фрагментов. "
            "Не выдумывай факты. Пиши по-русски. "
            "Сначала опирайся на российские рекомендации, зарубежные — для сравнения. "
            "После утверждений ставь ссылки [1],[2]. "
            "В конце добавь дисклеймер: это не медицинская рекомендация."
        )


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_sources(hits: Iterable[Any]) -> list[SourceItem]:
    sources: list[SourceItem] = []
    for i, hit in enumerate(hits, start=1):
        sources.append(
            SourceItem(
                index=i,
                source_id=_safe_get(hit, "source_id"),
                source_name=_safe_get(hit, "source_name"),
                title=_safe_get(hit, "title"),
                section_title=_safe_get(hit, "section_title"),
                url=_safe_get(hit, "url"),
                score=_safe_get(hit, "score"),
                chunk_text=_safe_get(hit, "chunk_text", "") or "",
                translated_chunk_text=_safe_get(hit, "translated_chunk_text"),
                summary=_safe_get(hit, "summary"),
                nosology=_safe_get(hit, "nosology"),
                specialty=_safe_get(hit, "specialty"),
                evidence_level=_safe_get(hit, "evidence_level"),
            )
        )
    return sources


def _render_structured_markdown(structured: dict[str, Any]) -> str:
    """Render Consensus/Disagreements/Recommendation/Sources as Markdown."""
    lines: list[str] = []

    consensus = structured.get("consensus")
    if consensus:
        lines += ["## ✅ Консенсус", str(consensus), ""]

    disagreements = structured.get("disagreements") or []
    if disagreements:
        lines.append("## ⚡ Расхождения")
        for item in disagreements:
            sources = ", ".join(str(s) for s in (item.get("sources") or []))
            text = str(item.get("text") or "")
            lines.append(f"**{sources}:** {text}" if sources else text)
        lines.append("")

    recommendation = structured.get("recommendation")
    if recommendation:
        lines += ["## 💡 Итоговая рекомендация", str(recommendation), ""]

    citations = structured.get("citations") or []
    if citations:
        lines.append("## 📎 Источники")
        for c in citations:
            idx = c.get("idx", "")
            source_name = c.get("source_name", "")
            section = c.get("section") or ""
            url = c.get("url")
            line = f"**[{idx}] {source_name}**"
            if section:
                line += f" — {section}"
            if url:
                line += f" ([ссылка]({url}))"
            lines.append(line)
        lines.append("")

    lines.append("_Не является медицинской рекомендацией, требуется подтверждение врачом._")
    return "\n".join(lines).strip()


def answer_question(
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    translate_mode: str = DEFAULT_TRANSLATE_MODE,
    model_family: str = DEFAULT_MODEL_FAMILY,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    return_raw: bool = False,
    synthesize: bool = True,
    per_source_k: int | None = None,
    specialty: str | None = None,
    nosology: str | None = None,
    source_id: str | None = None,
    document_id: int | None = None,
) -> AnswerResult:
    if retrieve is None:
        raise RuntimeError("app.services.retriever.retrieve is unavailable.")

    retrieval_result = retrieve(
        question,
        top_k=top_k,
        translate_mode=translate_mode,
        per_source_k=per_source_k,
        specialty=specialty,
        nosology=nosology,
        source_id=source_id,
        document_id=document_id,
    )

    hits = _safe_get(retrieval_result, "hits", []) or []
    sources = _normalize_sources(hits)

    if not sources:
        return AnswerResult(
            question=question,
            effective_query=_safe_get(retrieval_result, "effective_query"),
            model=YandexOpenAIAnswerer().build_model_uri(model_family),
            answer="Недостаточно информации в найденных источниках для ответа на вопрос.",
            disclaimer="Не является медицинской рекомендацией, требуется подтверждение врачом.",
            sources=[],
        )

    answerer = YandexOpenAIAnswerer()

    if synthesize:
        structured = answerer.synthesize_structured(
            question=question,
            context_sources=sources,
            model_family=model_family,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        if structured:
            answer_text = _render_structured_markdown(structured)
            return AnswerResult(
                question=question,
                effective_query=_safe_get(retrieval_result, "effective_query"),
                model=answerer.build_model_uri(model_family),
                answer=answer_text,
                disclaimer="Не является медицинской рекомендацией, требуется подтверждение врачом.",
                sources=sources,
                consensus=structured.get("consensus"),
                disagreements=structured.get("disagreements") or [],
                recommendation=structured.get("recommendation"),
            )

    # Fallback: plain grounded answer
    answer_text, model_uri, raw_payload = answerer.generate(
        question=question,
        context_sources=sources,
        model_family=model_family,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        return_raw=return_raw,
    )

    return AnswerResult(
        question=question,
        effective_query=_safe_get(retrieval_result, "effective_query"),
        model=model_uri,
        answer=answer_text,
        disclaimer="Не является медицинской рекомендацией, требуется подтверждение врачом.",
        sources=sources,
        raw_completion=raw_payload,
    )


def format_for_console(result: AnswerResult) -> str:
    parts = [
        f"question={result.question}",
        f"effective_query={result.effective_query}",
        f"model={result.model}",
        "",
        result.answer,
        "",
        f"Дисклеймер: {result.disclaimer}",
        "",
        "Источники:",
    ]
    for src in result.sources:
        parts.append(
            f"[{src.index}] {src.title or '-'} | section={src.section_title or '-'} | source_id={src.source_id or '-'} | url={src.url or '-'}"
        )
    return "\n".join(parts)


def format_for_ui(result: AnswerResult) -> dict[str, Any]:
    return {
        "answer": result.answer,
        "question": result.question,
        "effective_query": result.effective_query,
        "model": result.model,
        "disclaimer": result.disclaimer,
        "consensus": result.consensus,
        "disagreements": result.disagreements,
        "recommendation": result.recommendation,
        "sources": [asdict(item) for item in result.sources],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate grounded answer via Yandex OpenAI-compatible API")
    parser.add_argument("question", nargs="?", help="Question for testing from console")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--translate-mode", default=DEFAULT_TRANSLATE_MODE)
    parser.add_argument("--model", default=DEFAULT_MODEL_FAMILY)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--no-synthesize", action="store_true", help="Use plain grounded answer instead of structured synthesis")
    parser.add_argument("--json", action="store_true", help="Return UI-friendly JSON")
    parser.add_argument("--raw", action="store_true", help="Include raw OpenAI-compatible response")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    question = args.question or input("Введите вопрос: ").strip()
    if not question:
        raise SystemExit("Question is empty")

    result = answer_question(
        question,
        top_k=args.top_k,
        translate_mode=args.translate_mode,
        model_family=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        return_raw=args.raw,
        synthesize=not args.no_synthesize,
    )

    if args.json:
        payload = format_for_ui(result)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(format_for_console(result))


if __name__ == "__main__":
    main()
