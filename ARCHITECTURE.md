# Evidentum — Архитектура системы

> Интеллектуальный поиск по клиническим рекомендациям с генерацией структурированных ответов.  
> Стек: **FastAPI · PostgreSQL/pgvector · Yandex Foundation Models · Streamlit · Docker**

---

## 1. Обзор системы

```
┌──────────────────────────────────────────────────────────────────┐
│                          Docker Compose                          │
│                                                                  │
│  ┌───────────┐    REST     ┌───────────┐    SQL/pgvector         │
│  │  Streamlit│ ──────────► │  FastAPI  │ ──────────────► ┌─────┐ │
│  │   (UI)    │             │   (API)   │                 │  PG │ │
│  │  :7860    │ ◄────────── │   :8000   │ ◄────────────── │  DB │ │
│  └───────────┘   JSON      └─────┬─────┘                 └─────┘ │
│                                  │                               │
│                         Yandex Foundation Models (HTTPS)         │
│                         ┌────────┼─────────────────┐            │
│                         ▼        ▼                  ▼            │
│                    Embeddings  YandexGPT        Translate        │
│                    (256-dim)  (AliceAI/5.1)     (RU↔EN)          │
└──────────────────────────────────────────────────────────────────┘
```

Три Docker-контейнера:
| Контейнер | Образ | Роль |
|-----------|-------|------|
| `mvp_db` | `pgvector/pgvector:pg16` | PostgreSQL с расширением `pgvector` |
| `mvp_api` | собственный Python-образ | FastAPI + все backend-сервисы |
| `mvp_ui` | собственный Python-образ | Streamlit frontend |

Все три сервиса работают в одной Docker-сети `mvp-net` и общаются только внутри неё.  
Снаружи доступны порты `127.0.0.1:7860` (UI) и `127.0.0.1:8000` (API).

---

## 2. Схема базы данных

```sql
documents
  id              SERIAL PK
  source_id       VARCHAR(256)   -- slug для фильтрации: "esc_af_2024"
  source_name     VARCHAR(256)   -- читаемое название: "ESC 2024 (Фибрилляция предсердий)"
  title           TEXT           -- заголовок из PDF-метаданных
  region          VARCHAR(8)     -- "RU" | "EU" | "US"
  year            INTEGER
  url             TEXT
  file_path       TEXT
  -- LLM-обогащение на уровне документа
  specialty           VARCHAR(128)
  nosology_primary    VARCHAR(256)
  summary_ru          TEXT       -- TL;DR всего документа
  previous_document_id INTEGER FK→documents(id)
  version_delta_ru    TEXT       -- что изменилось vs предыдущая версия

chunks
  id              SERIAL PK
  document_id     INTEGER FK→documents(id)
  chunk_index     INTEGER
  section_title   TEXT
  chunk_text      TEXT
  char_count      INTEGER
  embedding       vector(256)    -- pgvector, индекс ivfflat/cosine
  -- LLM-обогащение на уровне фрагмента
  summary         TEXT           -- 2-3 предложения от YandexGPT
  nosology        VARCHAR(256)   -- "Фибрилляция предсердий I48"
  specialty       VARCHAR(128)   -- "кардиология"
  topic           VARCHAR(128)   -- therapy | diagnosis | prevention | ...
  evidence_level  VARCHAR(32)    -- "1A" | "2B" | null
```

Индексы: `ix_chunks_nosology`, `ix_chunks_specialty`, pgvector-индекс на `embedding`.

---

## 3. RAG-схема (подробно)

### 3.1 Индексация (offline, при старте)

```
PDF / HTML файл
      │
      ▼  app/services/ingest.py
  ┌──────────────┐
  │   Парсинг    │  PyPDF2 (PDF) или BeautifulSoup (HTML)
  │   текста     │  → извлечение структуры: заголовки разделов
  └──────┬───────┘
         │ иерархический чанкинг
         │ (target_chars ≈ 1000, overlap_chars ≈ 100)
         ▼
  ┌──────────────┐
  │   Chunks     │  Запись в таблицу chunks (без embedding)
  └──────┬───────┘
         │
         ▼  app/services/enricher.py  [background thread]
  ┌──────────────────────────────────────┐
  │  LLM-обогащение каждого фрагмента    │
  │  YandexGPT → JSON:                   │
  │  {                                   │
  │    "summary":        "2-3 предл.",   │
  │    "nosology":       "ФП I48",       │
  │    "specialty":      "кардиология",  │
  │    "topic":          "therapy",      │
  │    "evidence_level": "1A"            │
  │  }                                   │
  └──────┬───────────────────────────────┘
         │  chunks.summary / nosology / ...
         ▼  app/services/embedder.py
  ┌──────────────────────────────┐
  │  Формирование embed-текста   │
  │  text = summary + "\n\n" +   │
  │         chunk_text           │  (если summary есть)
  └──────┬───────────────────────┘
         │
         ▼  Yandex Embeddings API
  emb://folder/text-search-doc/latest
         │  → vector(256)
         ▼
  chunks.embedding  (pgvector)
```

**Автоматическая обработка** (`app/services/autoprocess.py`, запускается через `threading` на старте API):
1. Сканирует `data/raw/` → ингестирует новые PDF/HTML
2. Создаёт эмбеддинги для чанков без них
3. Генерирует читаемые названия документов (LLM) если `source_name` — технический slug
4. Запускает обогащение чанков в фоновом потоке + переэмбеддинг после обогащения

---

### 3.2 Retrieval (online, на каждый запрос)

```
Вопрос пользователя (RU)
        │
        ▼  translate_mode = "dual_query"  (рекомендуемый)
  ┌─────┴──────────────────────────────┐
  │           Два прохода              │
  │                                    │
  │  Проход 1: RU-запрос              │
  │  query_ru → Yandex Embeddings     │
  │  (text-search-query/latest)       │
  │  → vector(256)                    │
  │  → pgvector cosine_distance()     │
  │  → top-K*4 кандидатов             │
  │                                    │
  │  Проход 2: EN-запрос              │
  │  query_ru → Yandex Translate      │
  │           → query_en              │
  │  → Yandex Embeddings              │
  │  → vector(256)                    │
  │  → pgvector cosine_distance()     │
  │  → top-K*4 кандидатов             │
  └─────┬──────────────────────────────┘
        │
        ▼  merge_and_rerank_hits()
  ┌─────────────────────────────────────────┐
  │           Слияние и реранкинг           │
  │                                         │
  │  1. Дедупликация по chunk_id            │
  │     (берём лучший из двух проходов)     │
  │                                         │
  │  2. Переосчёт score:                    │
  │     base = 1.0 - cosine_distance        │
  │     + region_bonus   (RU +0.05)         │
  │     + section_bonus  ("лечение" +0.04,  │
  │                       "оглавление" -0.08)│
  │     + lexical_overlap (+0..+0.08)       │
  │                                         │
  │  3. Ограничение: max 2 чанка/документ   │
  │     (для режима "Ответ")                │
  └─────┬───────────────────────────────────┘
        │  top-K hits
        ▼
  RetrievalResult { hits: [RetrievalHit] }
```

**Фильтры поиска** (применяются в SQL WHERE):
- `source_id` — конкретный документ (режим "Ответ" с выбранным источником)
- `document_id` — конкретный документ по ID
- `specialty` — врачебная специальность (из LLM-обогащения)
- `nosology` — нозология (ILIKE `%...%`)
- `region` — регион документа

**per_source_k** (режим "Сравнение"): SQL-оконная функция `ROW_NUMBER() OVER (PARTITION BY source_id)` — берём не более N лучших чанков из каждого источника отдельно.

---

### 3.3 Генерация ответа: режим «Ответ»

```
Вопрос + top-K hits
        │
        ▼  answer_question(synthesize=False)
  ┌─────────────────────────────────────────────────┐
  │  YandexOpenAIAnswerer.generate()                │
  │                                                  │
  │  System prompt:                                 │
  │  "Ты медицинский ассистент. Отвечай только      │
  │   на основе фрагментов. После утверждений        │
  │   ставь ссылки [1],[2]..."                       │
  │                                                  │
  │  User prompt:                                   │
  │  [1] source_id=... title=... section=...        │
  │      score=...  fragment=<текст>                │
  │  [2] ...                                        │
  │  Вопрос: <вопрос>                               │
  │                                                  │
  │  → YandexGPT (gpt://folder/aliceai-llm/latest)  │
  │                                                  │
  └─────────────────────────────────────────────────┘
        │
        ▼  AnswerResult
  "Дозировка апиксабана составляет 5 мг 2 р/сут [1].
   При CrCl < 30 доза снижается до 2,5 мг 2 р/сут [2].
   _Не является медицинской рекомендацией._"
```

---

### 3.4 Генерация ответа: режим «Сравнение гайдлайнов»

```
Вопрос
        │
        ▼  retrieve(top_k=12)  [без per_source фильтрации]
  top-12 hits из всех источников
        │
        ▼  Группировка по source_id
  {
    "esc_af_2024":  [hit1, hit2, hit3],
    "kr_rf_af_2024": [hit4, hit5],
    "acc_aha_2023":  [hit6, hit7, hit8],
    ...
  }
        │
        ▼  Фильтрация по min_score=0.35
  Источники с max(score) < 0.35 — отбрасываются
  (нерелевантные гайдлайны не попадают в сравнение)
        │
        ▼  synthesize_structured()
  ┌────────────────────────────────────────────────────────┐
  │  System prompt:                                        │
  │  "Верни строгий JSON:                                  │
  │  {                                                     │
  │    positions: [                                        │
  │      {source: "Название гайдлайна",                    │
  │       text:   "Позиция этого гайдлайна по вопросу"}    │
  │    ],                                                  │
  │    consensus:     "Что рекомендуют все/большинство",   │
  │    disagreements: [{sources:[...], text: "Суть"}],     │
  │    recommendation: "Итоговая рекомендация врачу"       │
  │  }"                                                    │
  │                                                        │
  │  User prompt:                                          │
  │  Гайдлайны в контексте:                               │
  │    ESC 2024 (Фибрилляция предсердий)                   │
  │    КР МЗ РФ 2024 (Фибрилляция предсердий)              │
  │    ACC/AHA/ACCP/HRS 2023 ...                           │
  │                                                        │
  │  === ГАЙДЛАЙН: ESC 2024 (Фибрилляция предсердий) ===  │
  │  [1] Раздел: Anticoagulation                          │
  │      Текст: ПОАК рекомендованы предпочтительно...     │
  │  [2] ...                                              │
  │                                                        │
  │  === ГАЙДЛАЙН: КР МЗ РФ 2024 ... ===                  │
  │  [3] ...                                              │
  │                                                        │
  │  → YandexGPT                                          │
  └────────────────────────────────────────────────────────┘
        │
        ▼  JSON-ответ
  {
    "positions": [
      {"source": "ESC 2024...", "text": "ПОАК рекомендованы..."},
      {"source": "КР МЗ РФ...", "text": "Пожизненная АКТ при ГКМП..."}
    ],
    "consensus": "Все гайдлайны рекомендуют ПОАК...",
    "disagreements": [
      {"sources": ["КР МЗ РФ...", "ESC 2024..."],
       "text": "КР МЗ РФ особо выделяет ГКМП..."}
    ],
    "recommendation": "Назначить ПОАК, учитывая..."
  }
```

---

## 4. Архитектура кода (backend)

```
app/
├── main.py               # FastAPI app, startup → autoprocess thread
├── db.py                 # SQLAlchemy engine, session_scope()
├── models.py             # ORM: Document, Chunk
│
├── api/
│   └── routes.py         # APIRouter: /health /sources /answer /compare /retrieve
│
└── services/
    ├── ingest.py         # PDF/HTML → Document + Chunks
    ├── embedder.py       # Chunks → vector(256) via Yandex Embeddings
    ├── enricher.py       # Chunks → summary/nosology/topic via YandexGPT
    ├── retriever.py      # Query → RetrievalResult (dual_query + reranking)
    ├── answer.py         # RetrievalResult → AnswerResult (plain или structured)
    ├── summarizer.py     # Document → summary_ru + version_delta_ru
    ├── autoprocess.py    # Оркестратор: ingest → embed → titles → enrich
    └── llm.py            # YandexLLMClient (OpenAI-compatible, retry, JSON parsing)
```

### Ключевые классы и типы

| Класс | Файл | Описание |
|-------|------|----------|
| `YandexAPIEmbedder` | embedder.py | HTTP-клиент к Yandex Embeddings API |
| `YandexTranslateClient` | retriever.py | HTTP-клиент к Yandex Translate API с retry |
| `RetrievalHit` | retriever.py | dataclass: один найденный фрагмент со score |
| `RetrievalResult` | retriever.py | dataclass: полный результат поиска |
| `SourceItem` | answer.py | dataclass: фрагмент для LLM-контекста |
| `AnswerResult` | answer.py | dataclass: готовый ответ системы |
| `YandexOpenAIAnswerer` | answer.py | LLM-клиент через OpenAI-compatible API |
| `YandexLLMClient` | llm.py | Универсальный LLM-клиент с `complete_json()` |

---

## 5. API эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/api/health` | Проверка работоспособности |
| `GET` | `/api/sources` | Список документов со статусом индексации |
| `POST` | `/api/answer` | Поиск + генерация ответа (режим «Ответ») |
| `POST` | `/api/compare` | Поиск + сравнение гайдлайнов |
| `POST` | `/api/retrieve` | Только поиск (без генерации), для отладки |
| `GET` | `/api/catalog` | Уникальные specialty/nosology для фильтров |
| `GET/POST` | `/api/documents/{id}/summary` | Получить/сгенерировать TL;DR документа |

### Схема запроса /api/answer

```json
{
  "question": "Длительность тройной антикоагулянтной терапии при ФП+ИМ?",
  "top_k": 6,
  "translate_mode": "dual_query",
  "model": "alice",
  "temperature": 0.2,
  "synthesize": false,
  "source_id": "kr_rf_af_2024"
}
```

### Схема запроса /api/compare

```json
{
  "question": "Всегда ли назначается антикоагулянтная терапия при ФП?",
  "top_k": 12,
  "translate_mode": "dual_query",
  "model": "alice",
  "temperature": 0.2,
  "min_score": 0.35,
  "max_chunks_per_source": 3
}
```

---

## 6. Yandex Foundation Models — использование

| Сервис | Модель / endpoint | Применение |
|--------|-------------------|------------|
| Embeddings | `emb://folder/text-search-doc/latest` | Индексация фрагментов (dim=256) |
| Embeddings | `emb://folder/text-search-query/latest` | Эмбеддинг пользовательского запроса |
| Translate | `translate.api.cloud.yandex.net` | RU→EN для dual_query; опционально EN→RU для хитов |
| YandexGPT | `gpt://folder/aliceai-llm/latest` | Обогащение чанков, генерация ответов (быстрее) |
| YandexGPT | `gpt://folder/yandexgpt-5.1/latest` | Сравнение и синтез (точнее) |

**Аутентификация**: `Api-Key` в заголовке `Authorization` + `x-folder-id`.

---

## 7. Жизненный цикл запроса (end-to-end)

```
Пользователь: "Как лечат ожирение?"
│
│  Streamlit (ui.py)
├── api_answer(question, source_id="...", synthesize=False)
│   POST /api/answer
│
│  FastAPI (routes.py → answer_route)
├── answer_question(question, source_id="...", synthesize=False)
│
│  services/answer.py
├── retrieve(question, translate_mode="dual_query", source_id="...")
│   │
│   │  services/retriever.py
│   ├── Yandex Translate: "Как лечат ожирение?" → "How to treat obesity?"
│   ├── Yandex Embeddings: query_ru  → vector_ru[256]
│   ├── Yandex Embeddings: query_en  → vector_en[256]
│   ├── pgvector: SELECT ... ORDER BY embedding <=> vector_ru LIMIT 24
│   ├── pgvector: SELECT ... ORDER BY embedding <=> vector_en LIMIT 24
│   ├── merge + rerank (region/section/lexical bonuses)
│   └── → top-6 RetrievalHit
│
├── YandexOpenAIAnswerer.generate(question, sources)
│   └── POST gpt://folder/aliceai-llm/latest
│       → "Для лечения ожирения используются орлистат [1], лираглутид [2]..."
│
└── → JSON { answer: "...", sources: [...], effective_query: "..." }

Streamlit: render_answer_result(data)
```

---

## 8. Деплой (docker-compose.yml)

```yaml
services:
  db:   pgvector/pgvector:pg16  # порт 55432 локально, 5432 внутри сети
  api:  evidentum-api           # :8000, healthcheck /api/health
  ui:   evidentum-ui            # :7860, streamlit run app/streamlit_app.py

volumes:
  postgres_data:  # данные БД сохраняются между перезапусками

networks:
  mvp-net:  # изолированная сеть для всех сервисов
```

Код примонтирован как volume (`./:/app`), поэтому изменения в `.py`-файлах применяются **без пересборки образа**:
- Streamlit: перезагружается сам при изменении файла
- FastAPI/API: нужен `docker compose restart api`

---

## 9. Миграции БД (Alembic)

```
alembic/versions/
├── 20260414_0001_initial.py        # documents + chunks + embeddings
└── 20260417_0003_llm_enrichment.py # summary/nosology/topic/... на chunks и documents
```

Применение: `docker exec mvp_api alembic upgrade head`

---

## 10. Статус индексации и автопроцесс

При каждом старте API (`@app.on_event("startup")`):

```
startup (5s delay)
  └── autoprocess.run_autoprocess()
        ├── 1. ingest  — новые файлы из data/raw/ → chunks
        ├── 2. embed   — chunks без embedding → Yandex Embeddings
        ├── 3. titles  — boring source_name (slug/код) → LLM-название
        └── 4. enrich  — [background thread]
                chunks без summary → YandexGPT enrichment
                → re-embed с summary+chunk
```

UI (Streamlit) отображает прогресс в реальном времени (`@st.cache_data(ttl=20)`):
- 🔴 есть чанки без эмбеддингов (поиск недоступен)
- 🟡 поиск работает, обогащение не завершено
- 🟢 полностью готов

---

## 11. Где и как используется LLM-обогащение (важно)

Ключевое отличие нашей системы от IBD-пайплайна: обогащение у нас **асинхронное** (background thread), у IBD — blocking (сначала обогатили, потом запустили). Это означает, что система работает на всех стадиях готовности, но качество нарастает.

### Полная таблица использования `summary` и метаданных чанка

| Место в коде | Использует `summary`? | Влияет на качество ответа? |
|---|---|---|
| `embedder.py` — индексация | ✅ `f"{summary}\n\n{chunk_text}"` | ✅ поиск точнее |
| `synthesize_structured()` — LLM-синтез | ✅ `"Резюме: {summary}\n"` в промпте | ✅ позиции и консенсус лучше |
| `compare_route` — `fragments` для UI | ✅ передаётся в `fragments[].summary` | ❌ только отображение |
| `compare_route` — `combined_text` для UI | ❌ только `chunk_text` | ❌ только отображение |
| Фильтры `specialty` / `nosology` | ✅ SQL WHERE по этим полям | ✅ точность выборки |

### Что получает LLM при сравнении (пример одного фрагмента)

```
=== ГАЙДЛАЙН: ESC 2024 (Фибрилляция предсердий) ===
[1] Раздел: Anticoagulation [Уровень: 1A]
Резюме: ПОАК рекомендованы предпочтительно перед АВК при ФП у большинства пациентов.
Текст: <сырой PDF-текст до 1500 символов>
```

Если `summary` пустой (чанк не обогащён) — строка `Резюме:` просто опускается, LLM видит только сырой текст. Система работает, но качество позиций ниже.

### Принципиальное совпадение с IBD-архитектурой

IBD (arch.md, стр. 15): `embed_text = summary + "\n\n" + chunk_text`  
Наша система (`embedder.py:289`): `text = f"{summary}\n\n{body}" if summary else body`

IBD (arch.md, стр. 31): LLM получает контекст из разных гайдлайнов и генерирует структурированный ответ в трёх блоках  
Наша система (`synthesize_structured`): JSON-схема `{positions, consensus, disagreements, recommendation}`

Архитектурный подход идентичен. Разница только в **синхронности** подготовки данных.
