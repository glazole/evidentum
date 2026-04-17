"""Main page — clinical guidelines search and Q&A."""
from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv("UI_API_BASE_URL", "http://localhost:8000/api")
REQUEST_TIMEOUT = int(os.getenv("UI_REQUEST_TIMEOUT", "180"))

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


def api_feedback(log_id: int, feedback: int) -> None:
    requests.post(
        f"{API_BASE_URL}/feedback/{log_id}",
        json={"feedback": feedback},
        timeout=10,
    ).raise_for_status()


def api_upload(file_bytes: bytes, filename: str) -> dict[str, Any]:
    r = requests.post(
        f"{API_BASE_URL}/upload",
        files={"file": (filename, file_bytes, "application/pdf")},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


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
    min_score: float = 0.45,
    nosology_filter: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": question,
        "top_k": top_k,
        "translate_mode": translate_mode,
        "model": model,
        "temperature": temperature,
        "min_score": min_score,
        "nosology_filter": nosology_filter,
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

def _api_trigger(document_id: int, stage: str) -> dict[str, Any]:
    """POST /api/documents/{id}/{stage} and return response dict."""
    r = requests.post(
        f"{API_BASE_URL}/documents/{document_id}/{stage}",
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def render_status() -> None:
    """Render indexing status with per-document progress bars and trigger buttons."""
    items = load_sources()
    if not items:
        st.warning("Не удалось загрузить список источников. API недоступен?")
        return

    any_missing_emb = any(i.get("missing_embeddings", 0) > 0 for i in items)
    any_not_enriched = any(
        i.get("enriched_chunks", 0) < i.get("chunk_count", 0)
        and i.get("missing_embeddings", 0) == 0
        for i in items
    )

    # Summary banner
    banner_col, refresh_col = st.columns([8, 1])
    with banner_col:
        if any_missing_emb:
            st.error(
                "🔴 Часть документов ещё не проиндексирована — поиск по ним недоступен. "
                "Запустите этап **Эмбеддинги** вручную или дождитесь фонового процесса.",
            )
        elif any_not_enriched:
            st.warning(
                "🟡 Обогащение не завершено. Поиск работает, но **качество ответов ниже**: "
                "система использует сырой текст вместо LLM-дистиллята.",
            )
        else:
            st.success(f"✅ Все {len(items)} документов полностью проиндексированы и обогащены.")
    if refresh_col.button("↻", help="Обновить статус", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # Per-document rows
    for item in items:
        doc_id: int = item.get("document_id", 0)
        total: int = item.get("chunk_count", 0) or 1  # avoid /0
        missing_emb: int = item.get("missing_embeddings", 0)
        enriched: int = item.get("enriched_chunks", 0)
        embedded: int = total - missing_emb
        name = display_name(item)
        source_id_val = item.get("source_id") or "—"

        emb_pct   = embedded / total
        enrich_pct = enriched / total

        # Overall icon
        if missing_emb > 0:
            icon = "🔴"
        elif enriched < total:
            icon = "🟡"
        else:
            icon = "🟢"

        with st.container():
            # Header row: icon + name + source_id
            hdr, btn_col = st.columns([6, 2])
            hdr.markdown(
                f"{icon} **{name}**  "
                f"<span style='color:#888; font-size:0.78rem'>&nbsp;{source_id_val}"
                f" · {total} фрагментов</span>",
                unsafe_allow_html=True,
            )

            # Action buttons (inline, right-aligned)
            with btn_col:
                b1, b2, b3 = st.columns(3)
                trigger_key = f"trig_{doc_id}"

                if b1.button(
                    "▶ Embed",
                    key=f"embed_{doc_id}",
                    help="Принудительно построить эмбеддинги для этого документа",
                    width="stretch",
                ):
                    try:
                        resp = _api_trigger(doc_id, "embed")
                        st.session_state[trigger_key] = resp.get("message", "Запущено")
                        st.cache_data.clear()
                    except Exception as e:
                        st.session_state[trigger_key] = f"Ошибка: {e}"

                if b2.button(
                    "▶ Enrich",
                    key=f"enrich_{doc_id}",
                    help="Принудительно запустить LLM-обогащение фрагментов",
                    width="stretch",
                ):
                    try:
                        resp = _api_trigger(doc_id, "enrich")
                        st.session_state[trigger_key] = resp.get("message", "Запущено")
                        st.cache_data.clear()
                    except Exception as e:
                        st.session_state[trigger_key] = f"Ошибка: {e}"

                if b3.button(
                    "▶ Title",
                    key=f"title_{doc_id}",
                    help="Перегенерировать читаемое название через LLM",
                    width="stretch",
                ):
                    try:
                        resp = _api_trigger(doc_id, "title")
                        st.session_state[trigger_key] = resp.get("message", "Запущено")
                        st.cache_data.clear()
                    except Exception as e:
                        st.session_state[trigger_key] = f"Ошибка: {e}"

            # Show last trigger message if any
            if st.session_state.get(trigger_key):
                st.caption(f"ℹ️ {st.session_state[trigger_key]}")

            # Progress bars
            pb1, pb2, pb3 = st.columns(3)
            with pb1:
                st.caption("📥 Фрагменты (инgest)")
                st.progress(1.0, text=f"{total}/{total}")
            with pb2:
                st.caption("🔢 Эмбеддинги")
                st.progress(
                    emb_pct,
                    text=f"{embedded}/{total}"
                    + (" ✅" if missing_emb == 0 else f" — {missing_emb} не готово"),
                )
            with pb3:
                st.caption("✨ LLM-обогащение")
                st.progress(
                    enrich_pct,
                    text=f"{enriched}/{total}"
                    + (" ✅" if enriched >= total else f" — {total - enriched} не готово"),
                )

        st.markdown("<hr style='margin:6px 0; border-color:#e0e0e0'>", unsafe_allow_html=True)


# ── Result renderers ──────────────────────────────────────────────────────────

def render_feedback_buttons(log_id: int | None, key_prefix: str) -> None:
    """Render 👍/👎 feedback buttons for a query."""
    if log_id is None:
        return
    fb_key = f"fb_sent_{log_id}"
    if st.session_state.get(fb_key):
        st.caption("✅ Спасибо за оценку!")
        return
    st.markdown("**Был ли ответ полезен?**")
    c1, c2, _ = st.columns([1, 1, 6])
    if c1.button("👍 Да", key=f"{key_prefix}_up"):
        try:
            api_feedback(log_id, 1)
            st.session_state[fb_key] = True
            st.rerun()
        except Exception:
            st.error("Не удалось отправить оценку.")
    if c2.button("👎 Нет", key=f"{key_prefix}_down"):
        try:
            api_feedback(log_id, -1)
            st.session_state[fb_key] = True
            st.rerun()
        except Exception:
            st.error("Не удалось отправить оценку.")


def render_answer_result(data: dict) -> None:
    answer = (data.get("answer") or "").strip()
    sources = data.get("sources") or []
    disclaimer = data.get("disclaimer") or ""
    effective_query = data.get("effective_query") or ""
    log_id = data.get("log_id")

    if not answer:
        st.warning("Ответ не получен. Попробуйте переформулировать вопрос.")
        return

    render_feedback_buttons(log_id, "answer")
    st.divider()
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
    log_id = data.get("log_id")

    if effective_query:
        st.caption(f"🔍 Поисковый запрос: `{effective_query}`")

    if not sources:
        st.warning(
            "Релевантные источники не найдены. "
            "Попробуйте переформулировать вопрос или снизить порог схожести."
        )
        return

    # Build positions lookup: label → {text, evidence_levels}
    positions_by_label: dict[str, str] = {}
    for p in positions_list:
        src_key = (p.get("source") or "").strip()
        if src_key:
            positions_by_label[src_key] = (p.get("text") or "").strip()

    # Collect evidence levels per source from fragments
    def _collect_evidence(s: dict) -> str:
        levels: list[str] = []
        for frag in (s.get("fragments") or []):
            ev = (frag.get("evidence_level") or "").strip()
            if ev and ev not in levels:
                levels.append(ev)
        return ", ".join(levels) if levels else "—"

    # ── 1. Feedback buttons ───────────────────────────────────────────────────
    render_feedback_buttons(log_id, "compare")
    st.divider()

    # ── 2. Comparison table ───────────────────────────────────────────────────
    st.subheader("Сравнение позиций")

    for s in sources:
        label = s.get("label") or display_name(s)
        pos_text = positions_by_label.get(label) or (s.get("combined_text") or "")[:300]
        ev_levels = _collect_evidence(s)
        score = s.get("score")
        score_str = f"\n\n`score {score:.2f}`" if score is not None else ""

        col_name, col_ev, col_pos = st.columns([2, 1, 3], gap="medium")
        with col_name:
            st.markdown(f"**{label}**{score_str}")
        with col_ev:
            st.markdown(f"**УД**\n\n{ev_levels}")
        with col_pos:
            st.markdown(pos_text or "_нет данных_")
        st.divider()

    # ── 3. Synthesis & disagreements ──────────────────────────────────────────
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
                for src_name in src_list:
                    st.markdown(f"🔹 **{src_name}**")
                if text:
                    st.markdown(text)
    elif consensus:
        st.info("Существенных расхождений между источниками не выявлено.")

    if recommendation:
        st.info(f"**💡 Итоговая рекомендация**\n\n{recommendation}")

    # ── 4. Per-source detail blocks (collapsed, at the bottom) ────────────────
    st.divider()
    with st.expander(f"📚 Позиции источников — подробно ({len(sources)} гайдлайн(а/ов))", expanded=False):
        for s in sources:
            score = s.get("score")
            label = s.get("label") or display_name(s)
            url = s.get("url")
            score_str = f"  `score {score:.2f}`" if score is not None else ""

            st.markdown(f"### {label}{score_str}")
            if url:
                st.caption(f"📄 [Перейти к источнику]({url})")

            position_text = positions_by_label.get(label, "")
            if position_text:
                st.markdown(position_text)
            else:
                combined = (s.get("combined_text") or "").strip()
                if combined:
                    st.markdown(combined[:900] + (" …" if len(combined) > 900 else ""))

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
            st.divider()

    st.caption("_Информация носит справочный характер и не заменяет врачебное решение._")


# ── Main layout ───────────────────────────────────────────────────────────────

st.title("🏥 Evidentum")
st.caption("Поиск по клиническим рекомендациям с генерацией структурированного ответа")

st.divider()

# ── Upload section ────────────────────────────────────────────────────────────
with st.expander("📤 Загрузить гайдлайн", expanded=False):
    st.markdown(
        "Загрузите PDF-файл клинической рекомендации. После загрузки система автоматически:\n"
        "1. Разобьёт документ на фрагменты и построит эмбеддинги (~30 сек) — документ появится в поиске\n"
        "2. Обогатит фрагменты через LLM: summary, нозология, специальность (несколько минут в фоне)\n\n"
        "Прогресс отображается в блоке **«Статус индексации»** внизу страницы."
    )
    uploaded_file = st.file_uploader(
        "Выберите PDF-файл",
        type=["pdf"],
        key="guideline_upload",
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        if st.button("⬆️ Загрузить на сервер", type="primary"):
            with st.spinner(f"Загрузка «{uploaded_file.name}»…"):
                try:
                    result = api_upload(uploaded_file.getvalue(), uploaded_file.name)
                    st.success(
                        f"✅ **{uploaded_file.name}** принят.\n\n"
                        f"{result.get('message', '')}"
                    )
                    st.cache_data.clear()
                except requests.HTTPError as exc:
                    code = exc.response.status_code
                    detail = exc.response.json().get("detail", exc.response.text[:200])
                    if code == 409:
                        st.warning(f"⚠️ {detail}")
                    else:
                        st.error(f"Ошибка загрузки ({code}): {detail}")
                except Exception as exc:
                    st.error(f"Ошибка: {exc}")

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

    if mode == "Сравнение гайдлайнов":
        col_ms, col_nf = st.columns(2)
        with col_ms:
            min_score = st.slider(
                "Мин. релевантность",
                0.10, 0.90, 0.45, step=0.01,
                help=(
                    "Источники с лучшим score ниже порога исключаются из сравнения. "
                    "Рекомендуемый диапазон: 0.50–0.65."
                ),
            )
        with col_nf:
            nosology_filter = st.checkbox(
                "Фильтр по нозологии",
                value=True,
                help=(
                    "Исключать источники, чьи LLM-обогащённые фрагменты "
                    "содержат нозологию/специальность, не пересекающуюся с темой вопроса. "
                    "Работает только для обогащённых документов."
                ),
            )
    else:
        min_score = 0.55
        nosology_filter = False

    search_btn = st.button("🔍 Найти", type="primary", width="stretch")

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
                            min_score=min_score,
                            nosology_filter=nosology_filter,
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

# ── Indexing status (bottom, always collapsed) ────────────────────────────────
st.divider()
with st.expander("📊 Статус индексации документов", expanded=False):
    render_status()
