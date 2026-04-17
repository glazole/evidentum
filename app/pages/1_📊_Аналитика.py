"""Analytics page — LLM-judge scores and user feedback metrics."""
from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv("UI_API_BASE_URL", "http://localhost:8000/api")

st.set_page_config(page_title="Аналитика — Evidentum", page_icon="📊", layout="wide")
st.title("📊 Аналитика качества ответов")
st.caption(
    "Метрики обновляются после каждого запроса. "
    "LLM-judge оценивает каждый ответ по 4 критериям в фоне (~10 сек). "
    "Пользователь может поставить 👍/👎 прямо в поле ответа."
)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=15, show_spinner=False)
def load_metrics(limit: int = 100) -> dict[str, Any]:
    try:
        r = requests.get(f"{API_BASE_URL}/metrics", params={"limit": limit}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


col_refresh, _ = st.columns([1, 9])
if col_refresh.button("↻ Обновить", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

data = load_metrics()

if "error" in data:
    st.error(f"Не удалось загрузить метрики: {data['error']}")
    st.stop()

total = data.get("total_queries", 0)
if total == 0:
    st.info("Пока нет данных. Сделайте несколько поисковых запросов — метрики появятся здесь.")
    st.stop()


# ── Summary cards ─────────────────────────────────────────────────────────────

st.subheader("Общая статистика")

by_mode = data.get("by_mode", {})
fb = data.get("feedback", {})
avg = data.get("avg_scores", {})

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Всего запросов", total)
c2.metric("Режим «Ответ»", by_mode.get("answer", 0))
c3.metric("Режим «Сравнение»", by_mode.get("compare", 0))
c4.metric("👍 Полезно", fb.get("thumbs_up", 0))
c5.metric("👎 Не то", fb.get("thumbs_down", 0))
c6.metric("Без оценки", fb.get("not_rated", 0))

st.divider()


# ── LLM-judge average scores ─────────────────────────────────────────────────

judged = avg.get("judged_count", 0)
st.subheader(f"LLM-judge оценки (по {judged} из {total} запросов)")

if judged == 0:
    st.info("LLM-judge ещё не оценил ни одного ответа. Подождите ~30 секунд после запросов.")
else:
    score_cols = st.columns(4)
    criteria = [
        ("faithfulness", "Верность", "Ответ основан только на контексте"),
        ("relevance", "Релевантность", "Ответ отвечает на вопрос"),
        ("completeness", "Полнота", "Учтены все аспекты из контекста"),
        ("consistency", "Согласованность", "Нет внутренних противоречий"),
    ]
    for col, (key, label, help_text) in zip(score_cols, criteria):
        val = avg.get(key)
        val_str = f"{val:.2f} / 5" if val is not None else "—"
        delta = f"{(val - 3):.2f}" if val is not None else None
        col.metric(label, val_str, delta=delta, help=help_text)

    # Bar chart of average scores
    chart_data = {
        c[1]: [avg.get(c[0]) or 0]
        for c in criteria
    }
    import pandas as pd
    df_chart = pd.DataFrame(chart_data)
    st.bar_chart(df_chart, height=200, use_container_width=True)

st.divider()


# ── User feedback chart ───────────────────────────────────────────────────────

st.subheader("Обратная связь пользователей")
rated = fb.get("thumbs_up", 0) + fb.get("thumbs_down", 0)
if rated == 0:
    st.info("Пользователи ещё не оставили оценок.")
else:
    satisfaction = fb["thumbs_up"] / rated * 100
    st.metric(
        "Удовлетворённость",
        f"{satisfaction:.0f}%",
        help=f"👍 {fb['thumbs_up']} / (👍 {fb['thumbs_up']} + 👎 {fb['thumbs_down']})",
    )
    import pandas as pd
    fb_df = pd.DataFrame({
        "Оценка": ["👍 Полезно", "👎 Не то", "Без оценки"],
        "Количество": [fb["thumbs_up"], fb["thumbs_down"], fb["not_rated"]],
    }).set_index("Оценка")
    st.bar_chart(fb_df, height=180, use_container_width=True)

st.divider()


# ── Recent queries table ──────────────────────────────────────────────────────

st.subheader("Последние запросы")

recent = data.get("recent", [])
if not recent:
    st.info("Нет данных о запросах.")
else:
    import pandas as pd

    rows = []
    for r in recent:
        fb_icon = {1: "👍", -1: "👎"}.get(r.get("user_feedback"), "—")
        scores = [r.get(f"score_{k}") for k in ("faithfulness", "relevance", "completeness", "consistency")]
        avg_score = sum(s for s in scores if s is not None)
        avg_score_count = sum(1 for s in scores if s is not None)
        avg_val = f"{avg_score / avg_score_count:.2f}" if avg_score_count else "—"
        rows.append({
            "ID": r["id"],
            "Вопрос": r["question"][:80],
            "Режим": r["mode"],
            "Модель": r.get("model") or "—",
            "Время (мс)": r.get("elapsed_ms"),
            "Верность": r.get("score_faithfulness"),
            "Релев.": r.get("score_relevance"),
            "Полнота": r.get("score_completeness"),
            "Согл.": r.get("score_consistency"),
            "Avg": avg_val,
            "Фидбек": fb_icon,
            "Когда": (r.get("created_at") or "")[:16],
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn(width="small"),
            "Вопрос": st.column_config.TextColumn(width="large"),
            "Верность": st.column_config.NumberColumn(format="%.1f", width="small"),
            "Релев.": st.column_config.NumberColumn(format="%.1f", width="small"),
            "Полнота": st.column_config.NumberColumn(format="%.1f", width="small"),
            "Согл.": st.column_config.NumberColumn(format="%.1f", width="small"),
            "Avg": st.column_config.TextColumn(width="small"),
            "Время (мс)": st.column_config.NumberColumn(width="small"),
        },
    )

    # Reasoning expander for selected rows with judge text
    reasoned = [r for r in recent if r.get("judge_reasoning")]
    if reasoned:
        st.markdown("---")
        st.markdown("**Обоснования LLM-judge:**")
        for r in reasoned[:5]:
            with st.expander(f"[{r['id']}] {r['question'][:60]}…"):
                st.caption(r.get("judge_reasoning") or "")
