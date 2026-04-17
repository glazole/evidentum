"""Streamlit UI for Evidentum — clinical guidelines RAG system."""
from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv("UI_API_BASE_URL", "http://localhost:8000/api")
REQUEST_TIMEOUT = int(os.getenv("UI_REQUEST_TIMEOUT", "180"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Evidentum",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Compact header */
    .block-container { padding-top: 1.5rem; }
    /* Source cards */
    .src-card {
        background: #f4f6f9;
        border-left: 4px solid #3d7bbf;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .src-card-warn { border-left-color: #f0ad4e; }
    .src-card-ok   { border-left-color: #5cb85c; }
    /* Mute Streamlit default footer */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── API helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=20, show_spinner=False)
def load_sources() -> list[dict]:
    try:
        r = requests.get(f"{API_BASE_URL}/sources", timeout=10)
        r.raise_for_status()
        return r.json().get("items", [])
    except Exception:
        return []


def api_answer(
    question: str,
    *,
    source_id: str | None,
    top_k: int,
    translate_mode: str,
    model: str,
    temperature: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": question,
        "top_k": top_k,
        "translate_mode": translate_mode,
        "model": model,
        "temperature": temperature,
        "synthesize": False,  # Plain grounded answer — no consensus/disagreements
    }
    if source_id:
        payload["source_id"] = source_id
    r = requests.post(f"{API_BASE_URL}/answer", json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def api_compare(
    question: str,
    *,
    top_k: int,
    translate_mode: str,
    model: str,
    temperature: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": question,
        "top_k": top_k,
        "translate_mode": translate_mode,
        "model": model,
        "temperature": temperature,
    }
    r = requests.post(f"{API_BASE_URL}/compare", json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


# ── Display-name helper ───────────────────────────────────────────────────────

def display_name(item: dict) -> str:
    name = (item.get("source_name") or item.get("title") or "").strip()
    year = item.get("year")
    region = (item.get("region") or "").upper()
    boring = {"оглавление", "содержание", "circulation", "untitled", "title", ""}
    if not name or name.lower() in boring or len(name) < 6:
        name = (item.get("source_id") or "").replace("_", " ").upper()
    parts = [name]
    if year and str(year) not in name:
        parts.append(str(year))
    if region and region not in name.upper():
        parts.append(f"[{region}]")
    return " ".join(parts).strip() or "—"


# ── Status section ────────────────────────────────────────────────────────────

def render_status() -> None:
    items = load_sources()
    if not items:
        st.warning("Не удалось загрузить список источников. API недоступен?")
        return

    any_issues = False
    any_missing_emb = False
    any_not_enriched = False

    cols = st.columns(max(len(items), 1))
    for col, item in zip(cols, items):
        total = item.get("chunk_count", 0)
        missing_emb = item.get("missing_embeddings", 0)
        enriched = item.get("enriched_chunks", 0)
        embedded = total - missing_emb
        name = display_name(item)

        if missing_emb > 0:
            icon, delta_color = "🔴", "inverse"
            status_val = f"{embedded}/{total}"
            label_suffix = "эмбеддингов"
            any_issues = True
            any_missing_emb = True
        elif enriched < total:
            icon, delta_color = "🟡", "off"
            status_val = f"{enriched}/{total}"
            label_suffix = "обогащено"
            any_issues = True
            any_not_enriched = True
        else:
            icon, delta_color = "🟢", "normal"
            status_val = str(total)
            label_suffix = "фрагментов"

        col.metric(
            label=f"{icon} {name[:28]}",
            value=status_val,
            delta=label_suffix,
            delta_color=delta_color,
            help=f"source_id: {item.get('source_id')}\n"
                 f"Эмбеддинги: {embedded}/{total}\n"
                 f"Обогащено: {enriched}/{total}",
        )

    c1, c2 = st.columns([6, 1])
    if any_missing_emb:
        c1.error(
            "🔴 Часть документов ещё не проиндексирована — поиск по ним недоступен. "
            "Индексация выполняется в фоне, обновите статус через несколько минут.",
            icon="🔴",
        )
    elif any_not_enriched:
        c1.warning(
            "🟡 Обогащение фрагментов ещё не завершено. "
            "Поиск работает, но **качество ответов и сравнений ниже**: "
            "система использует сырой текст вместо LLM-дистиллята (summary). "
            "Обогащение выполняется автоматически в фоне.",
        )
    else:
        c1.success("Все документы полностью проиндексированы и обогащены.", icon="✅")

    if c2.button("↻ Обновить", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── Result renderers ──────────────────────────────────────────────────────────

def render_answer_result(data: dict) -> None:
    answer = (data.get("answer") or "").strip()
    sources = data.get("sources") or []
    disclaimer = data.get("disclaimer") or ""
    effective_query = data.get("effective_query") or ""

    if not answer:
        st.warning("Ответ не получен. Попробуйте переформулировать вопрос.")
        return

    st.markdown(answer)

    if disclaimer:
        st.caption(f"⚕️ {disclaimer}")
    if effective_query:
        st.caption(f"🔍 Поисковый запрос: `{effective_query}`")

    if sources:
        with st.expander(f"📎 Найденные фрагменты ({len(sources)})", expanded=False):
            for src in sources:
                idx = src.get("index", "-")
                source_name = src.get("source_name") or src.get("title") or "-"
                section = src.get("section_title") or "-"
                score = src.get("score")
                summary = (src.get("summary") or "").strip()
                text = (
                    src.get("translated_chunk_text") or src.get("chunk_text") or ""
                ).strip()
                score_str = f" · score={score:.3f}" if score is not None else ""
                st.markdown(f"**[{idx}] {source_name}**{score_str}  \n*{section}*")
                if summary:
                    st.markdown(f"> {summary}")
                if text:
                    st.markdown(text[:1000] + (" …" if len(text) > 1000 else ""))
                st.divider()


def render_compare_result(data: dict) -> None:
    sources = data.get("sources") or []
    positions_list = data.get("positions") or []
    consensus = data.get("consensus")
    disagreements = data.get("disagreements") or []
    recommendation = data.get("recommendation")
    effective_query = data.get("effective_query") or ""

    if effective_query:
        st.caption(f"🔍 Поисковый запрос: `{effective_query}`")

    if not sources:
        st.warning(
            "Релевантные источники не найдены. "
            "Попробуйте переформулировать вопрос или снизить порог схожести."
        )
        return

    # Build positions lookup: label → text (LLM-generated per-source answer)
    positions_by_label: dict[str, str] = {}
    for p in positions_list:
        src_key = (p.get("source") or "").strip()
        if src_key:
            positions_by_label[src_key] = (p.get("text") or "").strip()

    # ── Per-source blocks ────────────────────────────────────────────────────
    st.subheader(f"Позиции источников — {len(sources)} гайдлайн(а/ов)")

    for s in sources:
        score = s.get("score")
        label = s.get("label") or display_name(s)
        url = s.get("url")
        score_str = f"  `score {score:.2f}`" if score is not None else ""
        header = f"**{label}**{score_str}"

        with st.expander(header, expanded=True):
            if url:
                st.caption(f"📄 [Перейти к источнику]({url})")

            # LLM position on this specific question
            position_text = positions_by_label.get(label, "")
            if position_text:
                st.markdown(position_text)
            else:
                # Fallback: show combined raw text
                combined = (s.get("combined_text") or "").strip()
                if combined:
                    st.markdown(combined[:900] + (" …" if len(combined) > 900 else ""))

            # Numbered fragment list
            fragments = s.get("fragments") or []
            if fragments:
                st.markdown("---")
                st.markdown("**Фрагменты:**")
                for frag in fragments:
                    idx = frag.get("index", "-")
                    section = frag.get("section_title") or ""
                    frag_summary = (frag.get("summary") or "").strip()
                    text = (frag.get("text") or "").strip()
                    ev = frag.get("evidence_level")
                    ev_str = f" · УД: {ev}" if ev else ""
                    section_str = f" · *{section}*" if section else ""
                    st.markdown(f"**{idx}.{section_str}{ev_str}**")
                    if frag_summary:
                        st.markdown(f"> {frag_summary}")
                    if text:
                        with st.expander("Полный текст фрагмента", expanded=False):
                            st.markdown(text[:700] + (" …" if len(text) > 700 else ""))

    # ── Comparison table ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Сравнение позиций")

    for s in sources:
        label = s.get("label") or display_name(s)
        pos_text = positions_by_label.get(label) or (s.get("combined_text") or "")[:300]
        col_name, col_pos = st.columns([1, 2], gap="medium")
        with col_name:
            score = s.get("score")
            score_str = f"\n\n`score {score:.2f}`" if score is not None else ""
            st.markdown(f"**{label}**{score_str}")
        with col_pos:
            st.markdown(pos_text or "_нет данных_")
        st.divider()

    # ── Synthesis ─────────────────────────────────────────────────────────────
    st.subheader("Анализ")

    if consensus:
        st.success(f"**✅ Консенсус**\n\n{consensus}")
    else:
        st.caption("Единый консенсус между источниками не выявлен.")

    if disagreements:
        st.subheader("⚡ Расхождения")
        for item in disagreements:
            src_list = item.get("sources") or []
            text = (item.get("text") or "").strip()
            with st.container(border=True):
                # Each source on its own line as a badge
                for src_name in src_list:
                    st.markdown(f"🔹 **{src_name}**")
                if text:
                    st.markdown(text)
    elif consensus:
        st.info("Существенных расхождений между источниками не выявлено.")

    if recommendation:
        st.info(f"**💡 Итоговая рекомендация**\n\n{recommendation}")

    st.caption("_Информация носит справочный характер и не заменяет врачебное решение._")


# ── Main layout ───────────────────────────────────────────────────────────────

st.title("🏥 Evidentum")
st.caption("Поиск по клиническим рекомендациям с генерацией структурированного ответа")

# Status (collapsed by default once all green)
items_now = load_sources()
has_issues_now = any(
    item.get("missing_embeddings", 0) > 0
    or item.get("enriched_chunks", 0) < item.get("chunk_count", 0)
    for item in items_now
)
with st.expander("📊 Статус индексации", expanded=has_issues_now):
    render_status()

st.divider()

# ── Search form ───────────────────────────────────────────────────────────────
items = load_sources()
source_options: dict[str, str | None] = {"Все источники": None}
for item in items:
    source_options[display_name(item)] = item.get("source_id")

left, right = st.columns([2, 3], gap="large")

with left:
    question = st.text_area(
        "Вопрос",
        height=130,
        placeholder=(
            "Например: длительность тройной антикоагулянтной терапии "
            "при сочетании ФП и инфаркта миокарда"
        ),
        key="question_input",
    )

    mode = st.radio(
        "Режим поиска",
        options=["Ответ", "Сравнение гайдлайнов"],
        horizontal=True,
        help=(
            "**Ответ** — единый структурированный ответ с консенсусом.  \n"
            "**Сравнение** — позиции каждого гайдлайна + анализ расхождений."
        ),
    )

    source_label = st.selectbox(
        "Источник",
        options=list(source_options.keys()),
        disabled=(mode == "Сравнение гайдлайнов"),
        help="В режиме «Сравнение» поиск идёт по всем источникам автоматически",
    )
    source_id = source_options[source_label]

    col_m, col_t = st.columns(2)
    with col_m:
        model = st.selectbox(
            "Модель",
            options=["alice", "yandex"],
            help="alice = AliceAI-LLM (быстрее), yandex = YandexGPT 5.1 (точнее)",
        )
    with col_t:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, step=0.05)

    col_k, col_tr = st.columns(2)
    with col_k:
        top_k = st.slider("Top-K фрагментов", 3, 15, 6, step=1)
    with col_tr:
        translate_mode = st.selectbox(
            "Перевод запроса",
            options=["dual_query", "off", "query", "query_and_hits"],
            help=(
                "dual_query — поиск и по русскому, и по переведённому запросу (рекомендуется)"
            ),
        )

    search_btn = st.button("🔍 Найти", type="primary", use_container_width=True)

# ── Results panel ─────────────────────────────────────────────────────────────
with right:
    if search_btn:
        q = (question or "").strip()
        if not q:
            st.error("Введите вопрос перед поиском.")
        else:
            t0 = time.time()
            with st.spinner("Поиск и генерация ответа…"):
                try:
                    if mode == "Ответ":
                        data = api_answer(
                            q,
                            source_id=source_id,
                            top_k=top_k,
                            translate_mode=translate_mode,
                            model=model,
                            temperature=temperature,
                        )
                        st.session_state["result"] = ("answer", data)
                    else:
                        data = api_compare(
                            q,
                            top_k=top_k,
                            translate_mode=translate_mode,
                            model=model,
                            temperature=temperature,
                        )
                        st.session_state["result"] = ("compare", data)
                    st.session_state["elapsed"] = time.time() - t0
                except requests.HTTPError as exc:
                    st.error(f"Ошибка API ({exc.response.status_code}): {exc.response.text[:300]}")
                    st.session_state.pop("result", None)
                except Exception as exc:
                    st.error(f"Ошибка: {exc}")
                    st.session_state.pop("result", None)

    if "result" in st.session_state:
        elapsed = st.session_state.get("elapsed")
        if elapsed:
            st.caption(f"⏱ {elapsed:.1f} с")
        result_type, result_data = st.session_state["result"]
        if result_type == "answer":
            render_answer_result(result_data)
        else:
            render_compare_result(result_data)
    elif not search_btn:
        st.markdown(
            "<div style='color:#aaa; text-align:center; padding:60px 0;'>"
            "Введите вопрос и нажмите <b>Найти</b>"
            "</div>",
            unsafe_allow_html=True,
        )
