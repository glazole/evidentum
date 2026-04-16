from __future__ import annotations

import os
from typing import Any

import gradio as gr
import requests


API_BASE_URL = os.getenv("UI_API_BASE_URL", "http://localhost:8000/api")
REQUEST_TIMEOUT = int(os.getenv("UI_REQUEST_TIMEOUT", "180"))


def api_get(path: str) -> dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def load_sources() -> list[dict[str, Any]]:
    try:
        data = api_get("/sources")
        return data.get("items", [])
    except Exception:
        return []


def build_source_choices() -> list[tuple[str, str]]:
    items = load_sources()
    choices: list[tuple[str, str]] = [("Все источники", "")]
    for item in items:
        label = f"{item.get('source_name')} ({item.get('region') or '-'}, {item.get('year') or '-'})"
        choices.append((label, item.get("source_id", "")))
    return choices


def build_doc_choices() -> list[tuple[str, int]]:
    items = load_sources()
    choices: list[tuple[str, int]] = []
    for item in items:
        label = f"{item.get('source_name')} ({item.get('year') or '-'})"
        choices.append((label, item["document_id"]))
    return choices


def format_sources_table(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "Источники не найдены."

    lines = []
    for src in sources:
        idx = src.get("index", "-")
        title = src.get("title") or "-"
        section = src.get("section_title") or "-"
        source_id = src.get("source_id") or "-"
        score = src.get("score")
        url = src.get("url") or "-"
        evidence = src.get("evidence_level")
        extra = f" [уровень доказательности: {evidence}]" if evidence else ""
        lines.append(
            "\n".join([
                f"[{idx}] {title}{extra}",
                f"source_id: {source_id}",
                f"section: {section}",
                f"score: {score}",
                f"url: {url}",
            ])
        )
    return "\n\n---\n\n".join(lines)


def format_retrieval_preview(answer_payload: dict[str, Any]) -> str:
    sources = answer_payload.get("sources", []) or []
    if not sources:
        return "Нет найденных фрагментов."

    blocks: list[str] = []
    for src in sources:
        idx = src.get("index", "-")
        title = src.get("title") or "-"
        section = src.get("section_title") or "-"
        summary = src.get("summary") or ""
        snippet = (src.get("translated_chunk_text") or src.get("chunk_text") or "").strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200].rstrip() + " ..."
        lines = [f"[{idx}] {title}", f"section: {section}", ""]
        if summary:
            lines += [f"Резюме: {summary}", ""]
        lines.append(snippet)
        blocks.append("\n".join(lines))

    return "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(blocks)


def ask_question(
    question: str,
    source_id: str,
    top_k: int,
    translate_mode: str,
    model: str,
    temperature: float,
    synthesize: bool,
) -> tuple[str, str, str, str]:
    question = (question or "").strip()
    if not question:
        return "Введите вопрос.", "", "", ""

    payload: dict[str, Any] = {
        "question": question,
        "top_k": int(top_k),
        "translate_mode": translate_mode,
        "model": model,
        "temperature": float(temperature),
        "synthesize": synthesize,
    }
    if source_id:
        payload["source_id"] = source_id

    try:
        answer_data = api_post("/answer", payload)
    except Exception as exc:
        return f"Ошибка обращения к API: {exc}", "", "", ""

    answer_text = answer_data.get("answer", "")
    disclaimer = answer_data.get("disclaimer", "")
    effective_query = answer_data.get("effective_query", "") or ""
    sources_text = format_sources_table(answer_data.get("sources", []))
    preview_text = format_retrieval_preview(answer_data)

    meta_lines = [
        f"effective_query: {effective_query}",
        f"model: {answer_data.get('model', '-')}",
        f"disclaimer: {disclaimer}",
    ]
    meta_text = "\n".join(meta_lines)

    return answer_text, meta_text, sources_text, preview_text


def compare_guidelines(
    question: str,
    top_k: int,
    translate_mode: str,
    model: str,
    temperature: float,
) -> tuple[str, str]:
    question = (question or "").strip()
    if not question:
        return "Введите вопрос.", ""

    payload: dict[str, Any] = {
        "question": question,
        "top_k": int(top_k),
        "per_source_k": 2,
        "translate_mode": translate_mode,
        "model": model,
        "temperature": float(temperature),
    }

    try:
        data = api_post("/compare", payload)
    except Exception as exc:
        return f"Ошибка: {exc}", ""

    consensus = data.get("consensus")
    disagreements = data.get("disagreements") or []
    recommendation = data.get("recommendation")
    sources = data.get("sources") or []

    md_parts: list[str] = []

    if consensus:
        md_parts += [f"## ✅ Консенсус\n{consensus}", ""]

    if disagreements:
        md_parts.append("## ⚡ Расхождения")
        for item in disagreements:
            srcs = ", ".join(str(s) for s in (item.get("sources") or []))
            text = item.get("text") or ""
            md_parts.append(f"**{srcs}:** {text}" if srcs else text)
        md_parts.append("")

    if recommendation:
        md_parts += [f"## 💡 Итоговая рекомендация\n{recommendation}", ""]

    md_parts.append("_Не является медицинской рекомендацией._")
    analysis_md = "\n".join(md_parts)

    sources_lines = []
    for s in sources:
        name = s.get("source_name") or s.get("source_id") or "-"
        year = s.get("year")
        region = s.get("region") or "-"
        url = s.get("url")
        count = s.get("chunk_count", 0)
        line = f"**{name}** ({year or '-'}) | region: {region} | {count} фрагм."
        if url:
            line += f" | [ссылка]({url})"
        sources_lines.append(line)
    sources_md = "\n".join(sources_lines)

    return analysis_md, sources_md


def load_document_summary(document_id: int | None) -> tuple[str, str]:
    if document_id is None:
        return "Выберите документ.", ""
    try:
        data = api_get(f"/documents/{document_id}/summary")
        summary = data.get("summary_ru") or "_Резюме ещё не сформировано. Нажмите «Сформировать»._"
        delta = data.get("version_delta_ru") or ""
        return summary, delta
    except Exception as exc:
        return f"Ошибка: {exc}", ""


def generate_document_summary(document_id: int | None, model: str) -> tuple[str, str]:
    if document_id is None:
        return "Выберите документ.", ""
    try:
        api_post(f"/documents/{document_id}/summary", {"model": model})
        return "_Генерация запущена в фоне. Обновите через несколько секунд._", ""
    except Exception as exc:
        return f"Ошибка: {exc}", ""


# ── Build UI ──────────────────────────────────────────────────────────────────

_source_choices = build_source_choices()
_doc_choices = build_doc_choices()

with gr.Blocks(title="Evidentum") as demo:
    gr.Markdown(
        """
# Evidentum
Поиск по клиническим рекомендациям и генерация структурированного ответа.
        """.strip()
    )

    with gr.Tabs():

        # ── Tab 1: Ask ────────────────────────────────────────────────────────
        with gr.Tab("Вопрос-ответ"):
            with gr.Row():
                with gr.Column(scale=2):
                    question = gr.Textbox(
                        label="Вопрос",
                        lines=4,
                        placeholder="Например: дозировка апиксабана при ФП и почечной недостаточности",
                    )

                    source_dropdown = gr.Dropdown(
                        choices=_source_choices,
                        value="",
                        label="Источник",
                    )

                    with gr.Row():
                        model = gr.Dropdown(
                            choices=["alice", "yandex"],
                            value="alice",
                            label="Модель",
                        )
                        temperature = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.2, step=0.1, label="Temperature"
                        )

                    with gr.Row():
                        top_k = gr.Slider(minimum=3, maximum=15, value=6, step=1, label="Top-K")
                        translate_mode = gr.Dropdown(
                            choices=["off", "query", "query_and_hits", "dual_query"],
                            value="dual_query",
                            label="Режим перевода",
                        )

                    synthesize_cb = gr.Checkbox(
                        value=True, label="Структурированный синтез (Консенсус/Расхождения)"
                    )

                    ask_btn = gr.Button("Спросить", variant="primary")

                with gr.Column(scale=3):
                    answer_box = gr.Markdown(label="Ответ")
                    meta_box = gr.Textbox(label="Метаданные", lines=4)
                    sources_box = gr.Markdown(label="Источники")
                    preview_box = gr.Textbox(label="Найденные фрагменты", lines=20)

            ask_btn.click(
                fn=ask_question,
                inputs=[question, source_dropdown, top_k, translate_mode, model, temperature, synthesize_cb],
                outputs=[answer_box, meta_box, sources_box, preview_box],
            )

        # ── Tab 2: Compare ────────────────────────────────────────────────────
        with gr.Tab("Сравнение гайдлайнов"):
            with gr.Row():
                with gr.Column(scale=2):
                    cmp_question = gr.Textbox(
                        label="Вопрос для сравнения",
                        lines=3,
                        placeholder="Например: антикоагулянтная терапия при ФП",
                    )
                    with gr.Row():
                        cmp_model = gr.Dropdown(
                            choices=["alice", "yandex"], value="alice", label="Модель"
                        )
                        cmp_temperature = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.2, step=0.1, label="Temperature"
                        )
                    with gr.Row():
                        cmp_top_k = gr.Slider(minimum=3, maximum=20, value=8, step=1, label="Top-K")
                        cmp_translate_mode = gr.Dropdown(
                            choices=["off", "query", "query_and_hits", "dual_query"],
                            value="dual_query",
                            label="Режим перевода",
                        )
                    cmp_btn = gr.Button("Сравнить", variant="primary")

                with gr.Column(scale=3):
                    cmp_analysis = gr.Markdown(label="Анализ")
                    cmp_sources = gr.Markdown(label="Найденные источники")

            cmp_btn.click(
                fn=compare_guidelines,
                inputs=[cmp_question, cmp_top_k, cmp_translate_mode, cmp_model, cmp_temperature],
                outputs=[cmp_analysis, cmp_sources],
            )

        # ── Tab 3: Document Summary ───────────────────────────────────────────
        with gr.Tab("Резюме гайдлайна"):
            with gr.Row():
                with gr.Column(scale=1):
                    doc_dropdown = gr.Dropdown(
                        choices=_doc_choices,
                        label="Документ",
                    )
                    sum_model = gr.Dropdown(
                        choices=["alice", "yandex"], value="alice", label="Модель"
                    )
                    with gr.Row():
                        load_sum_btn = gr.Button("Загрузить", variant="secondary")
                        gen_sum_btn = gr.Button("Сформировать", variant="primary")

                with gr.Column(scale=3):
                    summary_box = gr.Markdown(label="TL;DR гайдлайна")
                    delta_box = gr.Markdown(label="Изменения vs предыдущая версия")

            load_sum_btn.click(
                fn=load_document_summary,
                inputs=[doc_dropdown],
                outputs=[summary_box, delta_box],
            )
            gen_sum_btn.click(
                fn=generate_document_summary,
                inputs=[doc_dropdown, sum_model],
                outputs=[summary_box, delta_box],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
