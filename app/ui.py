from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import gradio as gr
import requests


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


@dataclass
class AskUIResult:
    answer_markdown: str
    sources_markdown: str
    effective_query: str


def _post_json(path: str, payload: dict[str, Any], *, timeout_s: int = 60) -> dict[str, Any]:
    url = f"{API_BASE_URL}{path}"
    resp = requests.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _format_sources_md(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "Источники: —"
    lines: list[str] = ["### Источники"]
    for item in sources:
        idx = item.get("idx")
        source_name = item.get("source_name") or "Источник"
        title = item.get("title") or ""
        url = item.get("url")
        section = item.get("section_title")
        score = item.get("score")
        head = f"**[{idx}] {source_name} — {title}**"
        if section:
            head += f"  \n_{section}_"
        if url:
            head += f"  \n{url}"
        if score is not None:
            head += f"  \nscore={score:.3f}"
        lines.append(head)
    return "\n\n".join(lines).strip()


def ask_chat(
    message: str,
    history: list[dict[str, str]] | None,
    top_k: int,
    region: str,
    translate_mode: str,
) -> tuple[list[dict[str, str]], str, str, str]:
    history = history or []
    msg = (message or "").strip()
    if not msg:
        return history, "", "", ""

    payload: dict[str, Any] = {
        "query": msg,
        "top_k": int(top_k),
    }
    if region.strip():
        payload["region"] = region.strip()
    if translate_mode:
        payload["translate_mode"] = translate_mode

    try:
        data = _post_json("/api/ask", payload)
        answer = str(data.get("answer_markdown") or "").strip()
        effective_query = str(data.get("effective_query") or msg).strip()
        sources = data.get("sources") or []
        sources_md = _format_sources_md(list(sources))
    except Exception as exc:
        answer = f"Ошибка обращения к API: {exc}"
        sources_md = ""
        effective_query = msg

    history = history + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": answer},
    ]
    return history, "", sources_md, effective_query


def compare_for_query(query: str, region: str, translate_mode: str) -> tuple[list[list[Any]], str]:
    q = (query or "").strip()
    if not q:
        return [], "Сравнение: сначала задай вопрос в чате."

    payload: dict[str, Any] = {"query": q}
    if region.strip():
        payload["region"] = region.strip()
    if translate_mode:
        payload["translate_mode"] = translate_mode

    try:
        data = _post_json("/api/compare", payload)
        rows = data.get("rows") or []
        table_rows: list[list[Any]] = []
        for r in rows:
            table_rows.append(
                [
                    r.get("source_name"),
                    r.get("title"),
                    r.get("url"),
                    r.get("excerpt"),
                    r.get("score"),
                ]
            )
        return table_rows, "Сравнение сформировано по лучшим совпадениям из разных источников."
    except Exception as exc:
        return [], f"Ошибка сравнения: {exc}"


def subscribe(email: str, topic: str) -> str:
    e = (email or "").strip()
    t = (topic or "").strip()
    if not e or not t:
        return "Укажи email и тему/запрос."
    try:
        data = _post_json("/api/subscribe", {"email": e, "topic": t})
        sub_id = data.get("subscription_id")
        return f"Подписка сохранена (id={sub_id})."
    except Exception as exc:
        return f"Ошибка подписки: {exc}"


CSS = """
.container { max-width: 1100px; margin: 0 auto; }
.gradio-container { max-width: 1100px !important; }
"""


with gr.Blocks(css=CSS, title="evidentum") as demo:
    gr.Markdown(
        "## evidentum\n"
        "Поиск по международным клиническим гайдлайнам с объяснимостью (ссылки на источники).\n\n"
        "_Дисклеймер: сервис носит информационный характер и не является медицинской рекомендацией. "
        "При принятии решений используйте официальные источники._"
    )

    with gr.Row():
        top_k = gr.Slider(1, 20, value=5, step=1, label="Top-K фрагментов")
        region = gr.Textbox(value="", label="Фильтр region (опционально)", placeholder="RU / EU / US")
        translate_mode = gr.Dropdown(
            choices=["off", "query", "query_and_hits", "dual_query"],
            value=os.getenv("RETRIEVER_TRANSLATE_MODE", "off"),
            label="Перевод",
        )

    with gr.Tabs():
        with gr.Tab("Чат"):
            chatbot = gr.Chatbot(height=420, type="messages")
            msg = gr.Textbox(label="Вопрос", placeholder="Например: фибрилляция предсердий — нужна ли антикоагулянтная терапия?")

            with gr.Row():
                send_btn = gr.Button("Спросить", variant="primary")
                clear_btn = gr.Button("Очистить")

            with gr.Accordion("Источники", open=False):
                sources_md = gr.Markdown("Источники: —")

            # hidden state used as 'context' for comparison / subscribe
            effective_query = gr.Textbox(visible=False)

            send_btn.click(
                ask_chat,
                inputs=[msg, chatbot, top_k, region, translate_mode],
                outputs=[chatbot, msg, sources_md, effective_query],
            )
            msg.submit(
                ask_chat,
                inputs=[msg, chatbot, top_k, region, translate_mode],
                outputs=[chatbot, msg, sources_md, effective_query],
            )
            clear_btn.click(
                lambda: ([], "", "Источники: —", ""),
                outputs=[chatbot, msg, sources_md, effective_query],
            )

        with gr.Tab("Сравнение"):
            gr.Markdown("Сравнительный анализ по разным источникам для последнего запроса.")
            compare_btn = gr.Button("Сформировать сравнение")
            compare_status = gr.Markdown("")
            compare_table = gr.Dataframe(
                headers=["Источник", "Документ", "URL", "Фрагмент", "Score"],
                interactive=False,
                wrap=True,
            )
            compare_btn.click(
                compare_for_query,
                inputs=[effective_query, region, translate_mode],
                outputs=[compare_table, compare_status],
            )

        with gr.Tab("Подписка"):
            gr.Markdown("Подписаться на обновления по теме/запросу.")
            email = gr.Textbox(label="Email")
            topic = gr.Textbox(label="Тема/запрос", placeholder="Можно оставить последний запрос из чата")
            fill_btn = gr.Button("Подставить последний запрос")
            sub_btn = gr.Button("Подписаться", variant="primary")
            sub_status = gr.Markdown("")

            fill_btn.click(lambda q: q, inputs=[effective_query], outputs=[topic])
            sub_btn.click(subscribe, inputs=[email, topic], outputs=[sub_status])


def main() -> None:
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    root_path = (os.getenv("GRADIO_ROOT_PATH", "") or "").strip()
    if not root_path or root_path == "/":
        root_path = None
    demo.launch(server_name=server_name, server_port=server_port, root_path=root_path)


if __name__ == "__main__":
    main()

