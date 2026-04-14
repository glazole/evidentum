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
│   ├── ui.py        # Gradio
│   ├── db.py        # Create DB
│   ├── models.py    # Data models
│   └── services/
│       ├── ingest.py
│       └── embedder.py
├── data/
│   └── raw/
│       └── your_file.pdf # document
```
---
## Первый запуск

1. Сборка образов
```bash
docker compose up -d --build
```
2. Инициализация БД
```bash
docker compose exec app python -c "from app.db import init_db; init_db()"
```
3. Загрука (парсинг) тестового файла (любого)
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
