from __future__ import annotations

import json
import os
from typing import Any

import gradio as gr
import requests


API_BASE_URL = os.getenv("UI_API_BASE_URL", "http://localhost:8000/api")
REQUEST_TIMEOUT = int(os.getenv("UI_REQUEST_TIMEOUT", "180"))


def api_get(path: str) -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE_URL}{path}",
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}{path}",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
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
        label = f"{item.get('source_id')} | {item.get('region') or '-'} | {item.get('title')}"
        choices.append((label, item.get("source_id", "")))
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
        lines.append(
            "\n".join(
                [
                    f"[{idx}] {title}",
                    f"source_id: {source_id}",
                    f"section: {section}",
                    f"score: {score}",
                    f"url: {url}",
                ]
            )
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
        snippet = (src.get("translated_chunk_text") or src.get("chunk_text") or "").strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200].rstrip() + " ..."
        blocks.append(
            "\n".join(
                [
                    f"[{idx}] {title}",
                    f"section: {section}",
                    "",
                    snippet,
                ]
            )
        )

    return "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(blocks)


def ask_question(
    question: str,
    source_id: str,
    top_k: int,
    translate_mode: str,
    model: str,
    temperature: float,
) -> tuple[str, str, str, str]:
    question = (question or "").strip()
    if not question:
        return "Введите вопрос.", "", "", ""

    payload = {
        "question": question,
        "top_k": int(top_k),
        "translate_mode": translate_mode,
        "model": model,
        "temperature": float(temperature),
    }

    try:
        answer_data = api_post("/answer", payload)
    except Exception as exc:
        message = f"Ошибка обращения к API: {exc}"
        return message, "", "", ""

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


with gr.Blocks(title="Evidentum UI") as demo:
    gr.Markdown(
        """
# Evidentum
Тестовый интерфейс для поиска по клиническим рекомендациям и генерации ответа.
        """.strip()
    )

    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(
                label="Вопрос",
                lines=4,
                placeholder="Например: дозировка апиксабана при ФП и почечной недостаточности",
            )

            source_dropdown = gr.Dropdown(
                choices=build_source_choices(),
                value="",
                label="Источник",
            )

            with gr.Row():
                top_k = gr.Slider(
                    minimum=3,
                    maximum=10,
                    value=6,
                    step=1,
                    label="Top-K",
                )
                translate_mode = gr.Dropdown(
                    choices=["off", "query", "query_and_hits", "dual_query"],
                    value="dual_query",
                    label="Режим перевода",
                )

            with gr.Row():
                model = gr.Dropdown(
                    choices=["alice", "yandex"],
                    value="alice",
                    label="Модель",
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.2,
                    step=0.1,
                    label="Temperature",
                )

            ask_btn = gr.Button("Спросить", variant="primary")

        with gr.Column(scale=3):
            answer_box = gr.Markdown(label="Ответ")
            meta_box = gr.Textbox(label="Метаданные", lines=4)
            sources_box = gr.Markdown(label="Источники")
            preview_box = gr.Textbox(label="Найденные фрагменты", lines=20)

    ask_btn.click(
        fn=ask_question,
        inputs=[question, source_dropdown, top_k, translate_mode, model, temperature],
        outputs=[answer_box, meta_box, sources_box, preview_box],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)