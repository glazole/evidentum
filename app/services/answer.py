from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from openai import OpenAI

try:
    from app.services.retriever import retrieve
except Exception:  # pragma: no cover
    retrieve = None  # type: ignore


YC_OPENAI_BASE_URL = os.getenv("YC_OPENAI_BASE_URL", "https://llm.api.cloud.yandex.net/v1")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")

DEFAULT_MODEL_FAMILY = os.getenv("ANSWER_MODEL_FAMILY", "alice")
DEFAULT_TEMPERATURE = float(os.getenv("ANSWER_TEMPERATURE", "0.2"))
DEFAULT_TOP_K = int(os.getenv("ANSWER_TOP_K", "6"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("ANSWER_MAX_OUTPUT_TOKENS", "1800"))
DEFAULT_TRANSLATE_MODE = os.getenv("ANSWER_TRANSLATE_MODE", "dual_query")

MODEL_ALIASES = {
    "yandex": "yandexgpt-5.1",
    "yandexgpt": "yandexgpt-5.1",
    "pro": "yandexgpt-5.1",
    "alice": "aliceai-llm",
    "aliceai": "aliceai-llm",
}


@dataclass
class SourceItem:
    index: int
    source_id: str | None
    title: str | None
    section_title: str | None
    url: str | None
    score: float | None
    chunk_text: str
    translated_chunk_text: str | None = None


@dataclass
class AnswerResult:
    question: str
    effective_query: str | None
    model: str
    answer: str
    disclaimer: str
    sources: list[SourceItem]
    raw_completion: dict[str, Any] | None = None


class YandexOpenAIAnswerer:
    def __init__(
        self,
        *,
        folder_id: str = YANDEX_FOLDER_ID,
        api_key: str = YANDEX_API_KEY,
        base_url: str = YC_OPENAI_BASE_URL,
        default_model_family: str = DEFAULT_MODEL_FAMILY,
    ) -> None:
        if not folder_id:
            raise ValueError("YANDEX_FOLDER_ID is empty")
        if not api_key:
            raise ValueError("YANDEX_API_KEY is empty")

        self.folder_id = folder_id
        self.base_url = base_url.rstrip("/")
        self.default_model_family = default_model_family
        self.client = OpenAI(base_url=self.base_url, api_key=api_key)

    def build_model_uri(self, model_family: str | None = None) -> str:
        family = (model_family or self.default_model_family or "alice").strip().lower()
        model_id = MODEL_ALIASES.get(family, family)
        return f"gpt://{self.folder_id}/{model_id}/latest"

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

        system_prompt = (
            "Ты медицинский ассистент для MVP по клиническим рекомендациям. "
            "Отвечай только на основе переданных фрагментов. "
            "Не выдумывай факты и не добавляй рекомендации, которых нет в источниках. "
            "Пиши по-русски. "
            "Структура ответа: 1) Краткий вывод; 2) Что говорят источники; "
            "3) Различия между источниками; 4) Когда данных недостаточно. "
            "После каждого утверждения ставь ссылки вида [1], [2]. "
            "Если источников недостаточно, прямо так и напиши. "
            "В конце добавь короткий дисклеймер: это не медицинская рекомендация и требуется подтверждение врачом."
        )

        context_parts: list[str] = []
        for src in context_sources:
            snippet = (src.translated_chunk_text or src.chunk_text or "").strip()
            if len(snippet) > 1800:
                snippet = snippet[:1800].rstrip() + " ..."
            context_parts.append(
                "\n".join(
                    [
                        f"[{src.index}] source_id={src.source_id or '-'}",
                        f"title={src.title or '-'}",
                        f"section={src.section_title or '-'}",
                        f"url={src.url or '-'}",
                        f"score={src.score if src.score is not None else '-'}",
                        f"fragment={snippet}",
                    ]
                )
            )

        user_prompt = (
            f"Вопрос пользователя: {question}\n\n"
            "Ниже контекст из retriever:\n\n"
            + "\n\n".join(context_parts)
        )

        completion = self.client.chat.completions.create(
            model=model_uri,
            temperature=temperature,
            max_tokens=max_output_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = completion.choices[0].message.content or ""
        raw_payload = completion.model_dump() if return_raw and hasattr(completion, "model_dump") else None
        return answer.strip(), model_uri, raw_payload


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
                title=_safe_get(hit, "title"),
                section_title=_safe_get(hit, "section_title"),
                url=_safe_get(hit, "url"),
                score=_safe_get(hit, "score"),
                chunk_text=_safe_get(hit, "chunk_text", "") or "",
                translated_chunk_text=_safe_get(hit, "translated_chunk_text"),
            )
        )
    return sources



def answer_question(
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    translate_mode: str = DEFAULT_TRANSLATE_MODE,
    model_family: str = DEFAULT_MODEL_FAMILY,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    return_raw: bool = False,
) -> AnswerResult:
    if retrieve is None:
        raise RuntimeError(
            "app.services.retriever.retrieve is unavailable. "
            "Place this file inside your project and make sure retriever.py is importable."
        )

    retrieval_result = retrieve(
        question,
        top_k=top_k,
        translate_mode=translate_mode,
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
            raw_completion=None,
        )

    answerer = YandexOpenAIAnswerer()
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
        "sources": [asdict(item) for item in result.sources],
    }



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate grounded answer via Yandex OpenAI-compatible API")
    parser.add_argument("question", nargs="?", help="Question for testing from console")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--translate-mode", default=DEFAULT_TRANSLATE_MODE)
    parser.add_argument("--model", default=DEFAULT_MODEL_FAMILY, help="alice | yandex | full model id")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
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
    )

    if args.json:
        payload = format_for_ui(result)
        if args.raw:
            payload["raw_completion"] = result.raw_completion
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(format_for_console(result))
    if args.raw and result.raw_completion is not None:
        print("\nRAW COMPLETION:\n")
        print(json.dumps(result.raw_completion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
