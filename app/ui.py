from __future__ import annotations

import os
from typing import Any

import gradio as gr
import requests


API_BASE_URL = os.getenv("UI_API_BASE_URL", "http://localhost:8000/api")
REQUEST_TIMEOUT = int(os.getenv("UI_REQUEST_TIMEOUT", "180"))


# ── API helpers ───────────────────────────────────────────────────────────────

def api_get(path: str) -> dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sources() -> list[dict[str, Any]]:
    try:
        return api_get("/sources").get("items", [])
    except Exception:
        return []


def _display_name(item: dict[str, Any]) -> str:
    """Build a human-readable label for a guideline."""
    name = (item.get("source_name") or "").strip()
    year = item.get("year")
    region = (item.get("region") or "").upper()

    # If name looks uninformative, fall back to formatted source_id
    boring = {"оглавление", "содержание", "circulation", "untitled", "title", ""}
    if name.lower() in boring or len(name) < 6:
        name = item.get("source_id", "").replace("_", " ").upper()

    parts = [name]
    if year and str(year) not in name:
        parts.append(str(year))
    if region and region not in name.upper():
        parts.append(f"[{region}]")
    return " ".join(parts)


def build_source_choices(items: list[dict[str, Any]]) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = [("Все источники", "")]
    for item in items:
        choices.append((_display_name(item), item.get("source_id", "")))
    return choices


def build_status_markdown(items: list[dict[str, Any]]) -> str:
    """Return a warning markdown string if any documents have issues."""
    warnings: list[str] = []
    for item in items:
        name = _display_name(item)
        total = item.get("chunk_count", 0)
        missing_emb = item.get("missing_embeddings", 0)
        enriched = item.get("enriched_chunks", 0)

        if missing_emb > 0:
            warnings.append(
                f"⚠️ **{name}**: нет эмбеддингов для {missing_emb}/{total} фрагментов — поиск по этому источнику недоступен"
            )
        elif total > 0 and enriched < total:
            pct = int(enriched / total * 100)
            warnings.append(
                f"ℹ️ **{name}**: обогащено {enriched}/{total} фрагментов ({pct}%) — LLM-резюме частично отсутствуют"
            )

    if not warnings:
        return ""
    return "\n\n".join(["### ⚡ Статус индексации"] + warnings)


# ── Formatting ────────────────────────────────────────────────────────────────

def format_sources_table(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "Источники не найдены."
    lines = []
    for src in sources:
        idx = src.get("index", "-")
        title = src.get("title") or "-"
        section = src.get("section_title") or "-"
        score = src.get("score")
        url = src.get("url") or "-"
        evidence = src.get("evidence_level")
        extra = f" [уровень: {evidence}]" if evidence else ""
        lines.append("\n".join([
            f"[{idx}] {title}{extra}",
            f"section: {section}",
            f"score: {score} | url: {url}",
        ]))
    return "\n\n---\n\n".join(lines)


def format_retrieval_preview(sources: list[dict[str, Any]]) -> str:
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


# ── Business logic ────────────────────────────────────────────────────────────

def search(
    question: str,
    source_id: str,
    mode: str,
    top_k: int,
    translate_mode: str,
    model: str,
    temperature: float,
) -> tuple[str, str, str, str]:
    """
    Unified search function for both Answer and Compare modes.
    Returns: (main_md, detail_md, meta_text, preview_text)
    """
    question = (question or "").strip()
    if not question:
        return "Введите вопрос.", "", "", ""

    if mode == "Сравнение":
        return _compare(question, top_k, translate_mode, model, temperature)
    else:
        return _answer(question, source_id, top_k, translate_mode, model, temperature)


def _answer(
    question: str,
    source_id: str,
    top_k: int,
    translate_mode: str,
    model: str,
    temperature: float,
) -> tuple[str, str, str, str]:
    payload: dict[str, Any] = {
        "question": question,
        "top_k": int(top_k),
        "translate_mode": translate_mode,
        "model": model,
        "temperature": float(temperature),
        "synthesize": True,
    }
    if source_id:
        payload["source_id"] = source_id

    try:
        data = api_post("/answer", payload)
    except Exception as exc:
        return f"Ошибка API: {exc}", "", "", ""

    answer_text = data.get("answer", "")
    disclaimer = data.get("disclaimer", "")
    effective_query = data.get("effective_query") or ""
    sources_text = format_sources_table(data.get("sources", []))
    preview_text = format_retrieval_preview(data.get("sources", []))

    meta = "\n".join([
        f"effective_query: {effective_query}",
        f"model: {data.get('model', '-')}",
        f"disclaimer: {disclaimer}",
    ])
    return answer_text, sources_text, meta, preview_text


def _compare(
    question: str,
    top_k: int,
    translate_mode: str,
    model: str,
    temperature: float,
) -> tuple[str, str, str, str]:
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
        return f"Ошибка API: {exc}", "", "", ""

    sources = data.get("sources") or []

    # Per-source positions
    pos_parts: list[str] = []
    for s in sources:
        name = s.get("source_name") or s.get("source_id") or "-"
        year = s.get("year") or "-"
        region = (s.get("region") or "-").upper()
        url = s.get("url")
        text = (s.get("combined_text") or "").strip()
        if len(text) > 800:
            text = text[:800].rstrip() + " …"
        header = f"### {name} ({region}, {year})"
        if url:
            header += f" — [источник]({url})"
        pos_parts.append(f"{header}\n\n{text}" if text else header)
    positions_md = "\n\n---\n\n".join(pos_parts) if pos_parts else "Фрагменты не найдены."

    # Synthesis
    consensus = data.get("consensus")
    disagreements = data.get("disagreements") or []
    recommendation = data.get("recommendation")

    syn_parts: list[str] = []
    if consensus:
        syn_parts += [f"## ✅ Консенсус\n{consensus}", ""]
    if disagreements:
        syn_parts.append("## ⚡ Расхождения")
        for item in disagreements:
            srcs = ", ".join(str(s) for s in (item.get("sources") or []))
            text = item.get("text") or ""
            syn_parts.append(f"**{srcs}:** {text}" if srcs else text)
        syn_parts.append("")
    elif consensus:
        syn_parts += ["## ⚡ Расхождения\n_Существенных расхождений не выявлено._", ""]
    if recommendation:
        syn_parts += [f"## 💡 Итоговая рекомендация\n{recommendation}", ""]
    syn_parts.append("_Не является медицинской рекомендацией._")
    synthesis_md = "\n".join(syn_parts)

    effective_query = data.get("effective_query") or ""
    meta = f"effective_query: {effective_query}" if effective_query else ""

    return positions_md, synthesis_md, meta, ""


# ── Build UI ──────────────────────────────────────────────────────────────────

_items = load_sources()
_source_choices = build_source_choices(_items)
_status_md = build_status_markdown(_items)

with gr.Blocks(title="Evidentum") as demo:
    gr.Markdown(
        """
# Evidentum
Поиск по клиническим рекомендациям и генерация структурированного ответа.
        """.strip()
    )

    # ── Status warning ────────────────────────────────────────────────────────
    if _status_md:
        gr.Markdown(_status_md)

    with gr.Tabs():

        # ── Tab 1: Search (Ask + Compare merged) ─────────────────────────────
        with gr.Tab("Поиск"):
            with gr.Row():
                with gr.Column(scale=2):
                    question = gr.Textbox(
                        label="Вопрос",
                        lines=4,
                        placeholder="Например: дозировка апиксабана при ФП и почечной недостаточности",
                    )

                    mode = gr.Radio(
                        choices=["Ответ", "Сравнение"],
                        value="Ответ",
                        label="Режим",
                    )

                    source_dropdown = gr.Dropdown(
                        choices=_source_choices,
                        value="",
                        label="Источник (только в режиме «Ответ»)",
                    )

                    with gr.Row():
                        model = gr.Dropdown(
                            choices=["alice", "yandex"],
                            value="alice",
                            label="Модель",
                        )
                        temperature = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.2, step=0.1,
                            label="Temperature",
                        )

                    with gr.Row():
                        top_k = gr.Slider(
                            minimum=3, maximum=15, value=6, step=1, label="Top-K"
                        )
                        translate_mode = gr.Dropdown(
                            choices=["off", "query", "query_and_hits", "dual_query"],
                            value="dual_query",
                            label="Режим перевода",
                        )

                    search_btn = gr.Button("Найти", variant="primary")

                with gr.Column(scale=3):
                    main_md = gr.Markdown(
                        label="Ответ / Позиции источников",
                    )
                    detail_md = gr.Markdown(
                        label="Источники / Консенсус и расхождения",
                    )
                    meta_box = gr.Textbox(label="Метаданные", lines=3)
                    preview_box = gr.Textbox(label="Найденные фрагменты", lines=15)

            search_btn.click(
                fn=search,
                inputs=[question, source_dropdown, mode, top_k, translate_mode, model, temperature],
                outputs=[main_md, detail_md, meta_box, preview_box],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
