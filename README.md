# evidentum
AI-агрегатор медицинских гайдлайнов
```
project/
├── docker-compose.yml
├── .env
├── Dockerfile
├── requirements.txt
├── nginx/
│   └── conf.d/
│       └── app.conf
├── certbot/
│   └── init-letsencrypt.sh
├── app/
│   ├── main.py      # FastAPI
│   ├── streamlit_app.py # Streamlit
│   ├── db.py        # Create DB
│   ├── models.py    # Data models
│   └── services/
│       ├── ingest.py
│       ├── retriever.py
│       └── embedder.py
├── data/
│   └── raw/
│       └── your_file.pdf # document
```

**Краткое описание:**

1. **Ingest (загрузка данных)**
   Извлекается текст, разбивается на чанки с метаданными (источник, раздел, ссылка) и сохраняется в БД. 

2. **Embeddings (векторизация)**
   Для каждого чанка считается embedding через API (Yandex AI) и сохраняется в `pgvector`, формируя векторное представление корпуса.

3. **Подготовка к поиску**
   В базе уже есть:

   * текстовые фрагменты (chunks)
   * их embeddings
   * привязка к источникам (РФ / ESC / AHA)

4. **Retriever (поиск)**
   Пользовательский запрос преобразуется в embedding, затем выполняется similarity search по `pgvector` для получения top-k релевантных фрагментов.

---

**Итого:**

> документы → парсинг → чанки → embeddings → pgvector → similarity search (retriever)


---
## Первый запуск

1. Сборка образов
```bash
docker compose up -d --build
```
2. Проверка миграций БД
```bash
docker compose exec api alembic current
```

Миграции применяются автоматически при старте API через `alembic upgrade head`.

3. Загрузка (парсинг) тестового файла (любого)
```bash
docker compose exec app python -m app.services.ingest --file data/raw/your_file.pdf # рандомный тестовый файл
```
4. Получение эмбеддингов для документа
```bash
docker compose exec app python -m app.services.embedder --document-id 1
```
5. Получение списка документов
```bash
docker compose exec app python -m app.services.embedder --list-documents
```
6. Просмотр в консоли 1 чанка документа (1 строка)
```bash
docker compose exec app python -m app.services.embedder --document-id 1 --limit 1 --debug
```
7. Тестирование эмбеддингов
```bash
docker compose exec app python -m app.services.retriever   "антикоагулянтная терапия при фибрилляции предсердий"   --translate-mode dual_query   --top-k 8
```

## Важные переменные окружения

```bash
ADMIN_API_TOKEN=change-me              # обязателен для upload/delete/enrich/metrics
UI_ADMIN_API_TOKEN=change-me           # опционально, если UI должен слать отдельный токен
CORS_ALLOW_ORIGINS=http://localhost:7860
UPLOAD_MAX_BYTES=26214400              # лимит загрузки PDF/HTML, по умолчанию 25 МБ
DATA_RAW_DIR=/app/data/raw
```

Если `ADMIN_API_TOKEN` не задан, административные API-ручки возвращают `503`, чтобы случайно не оставить загрузку и удаление документов открытыми.
