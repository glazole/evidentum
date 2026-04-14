from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any

import requests
from sqlalchemy import Select, func, select

from app.db import session_scope
from app.models import Chunk, Document, EMBEDDING_DIM


YANDEX_EMBEDDINGS_URL = os.getenv(
    "YANDEX_EMBEDDINGS_URL",
    "https://ai.api.cloud.yandex.net/foundationModels/v1/textEmbedding",
)

YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")
YANDEX_IAM_TOKEN = os.getenv("YANDEX_IAM_TOKEN", "")

YANDEX_EMBEDDING_DOC_MODEL_URI = os.getenv(
    "YANDEX_EMBEDDING_DOC_MODEL_URI",
    f"emb://{YANDEX_FOLDER_ID}/text-search-doc/latest" if YANDEX_FOLDER_ID else "",
)

YANDEX_EMBEDDING_QUERY_MODEL_URI = os.getenv(
    "YANDEX_EMBEDDING_QUERY_MODEL_URI",
    f"emb://{YANDEX_FOLDER_ID}/text-search-query/latest" if YANDEX_FOLDER_ID else "",
)

REQUEST_TIMEOUT = int(os.getenv("YANDEX_EMBEDDINGS_TIMEOUT", "60"))
MAX_RETRIES = int(os.getenv("YANDEX_EMBEDDINGS_MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.getenv("YANDEX_EMBEDDINGS_RETRY_BASE_DELAY", "1.5"))


@dataclass
class EmbeddingStats:
    selected_chunks: int = 0
    embedded_chunks: int = 0
    skipped_chunks: int = 0
    document_id: int | None = None
    source_id: str | None = None


class YandexAPIEmbedder:
    """
    Embedding provider for Yandex Foundation Models Embeddings API.

    Supports:
    - document embeddings: emb://<folder_id>/text-search-doc/latest
    - query embeddings:    emb://<folder_id>/text-search-query/latest
    """

    def __init__(
        self,
        *,
        api_url: str = YANDEX_EMBEDDINGS_URL,
        doc_model_uri: str = YANDEX_EMBEDDING_DOC_MODEL_URI,
        query_model_uri: str = YANDEX_EMBEDDING_QUERY_MODEL_URI,
        folder_id: str = YANDEX_FOLDER_ID,
        api_key: str | None = YANDEX_API_KEY,
        iam_token: str | None = YANDEX_IAM_TOKEN,
        timeout: int = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        retry_base_delay: float = RETRY_BASE_DELAY,
        debug: bool = False,
    ) -> None:
        self.api_url = api_url
        self.doc_model_uri = doc_model_uri
        self.query_model_uri = query_model_uri
        self.folder_id = folder_id or ""
        self.api_key = api_key or ""
        self.iam_token = iam_token or ""
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.debug = debug

        if not self.doc_model_uri:
            raise ValueError(
                "YANDEX_EMBEDDING_DOC_MODEL_URI is empty. "
                "Set YANDEX_FOLDER_ID or pass explicit YANDEX_EMBEDDING_DOC_MODEL_URI."
            )

        if not self.query_model_uri:
            raise ValueError(
                "YANDEX_EMBEDDING_QUERY_MODEL_URI is empty. "
                "Set YANDEX_FOLDER_ID or pass explicit YANDEX_EMBEDDING_QUERY_MODEL_URI."
            )

        if not self.api_key and not self.iam_token:
            raise ValueError(
                "Neither YANDEX_API_KEY nor YANDEX_IAM_TOKEN is set. "
                "Provide one of them for Yandex authentication."
            )

        self.session = requests.Session()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Api-Key {self.api_key}"
        else:
            headers["Authorization"] = f"Bearer {self.iam_token}"

        if self.folder_id:
            headers["x-folder-id"] = self.folder_id

        return headers

    def _model_uri_for_type(self, text_type: str) -> str:
        if text_type == "doc":
            return self.doc_model_uri
        if text_type == "query":
            return self.query_model_uri
        raise ValueError(f"Unsupported text_type={text_type!r}. Use 'doc' or 'query'.")

    def _extract_embedding_from_response(self, data: dict[str, Any]) -> list[float]:
        if "embedding" not in data:
            raise ValueError(f"API did not return embedding. Response: {data}")
        return [float(x) for x in data["embedding"]]

    def _request_embedding(self, text: str, *, text_type: str) -> list[float]:
        payload = {
            "modelUri": self._model_uri_for_type(text_type),
            "text": text,
        }

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(
                    self.api_url,
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )

                if self.debug:
                    print("STATUS:", response.status_code)
                    print("BODY:", response.text)

                response.raise_for_status()

                data = response.json()
                vector = self._extract_embedding_from_response(data)

                if EMBEDDING_DIM and len(vector) != EMBEDDING_DIM:
                    raise ValueError(
                        f"Embedding dimension mismatch: got {len(vector)}, "
                        f"expected {EMBEDDING_DIM}. "
                        "Update EMBEDDING_DIM in app.models."
                    )

                return vector

            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                sleep_s = self.retry_base_delay * (2 ** (attempt - 1))
                time.sleep(sleep_s)

        raise RuntimeError(
            f"Failed to get {text_type} embedding from Yandex API: {last_error}"
        ) from last_error

    def embed_doc(self, text: str) -> list[float]:
        return self._request_embedding(text, text_type="doc")

    def embed_query(self, text: str) -> list[float]:
        return self._request_embedding(text, text_type="query")

    def embed_texts(self, texts: list[str], *, text_type: str = "doc") -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            results.append(self._request_embedding(text, text_type=text_type))
        return results


def build_chunks_query(
    *,
    document_id: int | None = None,
    only_missing: bool = True,
    limit: int | None = None,
) -> Select:
    stmt = select(Chunk).order_by(Chunk.document_id, Chunk.chunk_index)

    if document_id is not None:
        stmt = stmt.where(Chunk.document_id == document_id)

    if only_missing:
        stmt = stmt.where(Chunk.embedding.is_(None))

    if limit is not None:
        stmt = stmt.limit(limit)

    return stmt


def count_chunks(
    *,
    document_id: int | None = None,
    only_missing: bool = True,
) -> int:
    with session_scope() as session:
        stmt = select(func.count(Chunk.id))

        if document_id is not None:
            stmt = stmt.where(Chunk.document_id == document_id)

        if only_missing:
            stmt = stmt.where(Chunk.embedding.is_(None))

        return int(session.scalar(stmt) or 0)


def embed_chunks(
    *,
    document_id: int | None = None,
    batch_size: int = 8,
    limit: int | None = None,
    only_missing: bool = True,
    api_url: str = YANDEX_EMBEDDINGS_URL,
    doc_model_uri: str = YANDEX_EMBEDDING_DOC_MODEL_URI,
    query_model_uri: str = YANDEX_EMBEDDING_QUERY_MODEL_URI,
    folder_id: str = YANDEX_FOLDER_ID,
    api_key: str | None = YANDEX_API_KEY,
    iam_token: str | None = YANDEX_IAM_TOKEN,
    debug: bool = False,
) -> EmbeddingStats:
    provider = YandexAPIEmbedder(
        api_url=api_url,
        doc_model_uri=doc_model_uri,
        query_model_uri=query_model_uri,
        folder_id=folder_id,
        api_key=api_key,
        iam_token=iam_token,
        debug=debug,
    )

    stats = EmbeddingStats()

    with session_scope() as session:
        source_id: str | None = None

        if document_id is not None:
            document = session.scalar(select(Document).where(Document.id == document_id))
            if document is None:
                raise ValueError(f"Document with id={document_id} not found.")
            stats.document_id = document.id
            stats.source_id = document.source_id
            source_id = document.source_id

        chunks = list(
            session.scalars(
                build_chunks_query(
                    document_id=document_id,
                    only_missing=only_missing,
                    limit=limit,
                )
            )
        )

        stats.selected_chunks = len(chunks)
        stats.source_id = stats.source_id or source_id

        if not chunks:
            return stats

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]

            valid_pairs: list[tuple[Chunk, str]] = []
            for chunk in batch:
                text = (chunk.chunk_text or "").strip()
                if not text:
                    stats.skipped_chunks += 1
                    continue
                valid_pairs.append((chunk, text))

            if not valid_pairs:
                continue

            batch_chunks = [pair[0] for pair in valid_pairs]
            batch_texts = [pair[1] for pair in valid_pairs]

            vectors = provider.embed_texts(batch_texts, text_type="doc")

            if len(vectors) != len(batch_chunks):
                raise RuntimeError(
                    f"Embedding provider returned {len(vectors)} vectors "
                    f"for {len(batch_chunks)} texts."
                )

            for chunk, vector in zip(batch_chunks, vectors, strict=False):
                chunk.embedding = vector
                stats.embedded_chunks += 1

            session.flush()

    return stats


def get_query_embedding(
    text: str,
    *,
    api_url: str = YANDEX_EMBEDDINGS_URL,
    doc_model_uri: str = YANDEX_EMBEDDING_DOC_MODEL_URI,
    query_model_uri: str = YANDEX_EMBEDDING_QUERY_MODEL_URI,
    folder_id: str = YANDEX_FOLDER_ID,
    api_key: str | None = YANDEX_API_KEY,
    iam_token: str | None = YANDEX_IAM_TOKEN,
    debug: bool = False,
) -> list[float]:
    provider = YandexAPIEmbedder(
        api_url=api_url,
        doc_model_uri=doc_model_uri,
        query_model_uri=query_model_uri,
        folder_id=folder_id,
        api_key=api_key,
        iam_token=iam_token,
        debug=debug,
    )
    return provider.embed_query(text)


def reembed_document(
    document_id: int,
    *,
    batch_size: int = 8,
    limit: int | None = None,
    api_url: str = YANDEX_EMBEDDINGS_URL,
    doc_model_uri: str = YANDEX_EMBEDDING_DOC_MODEL_URI,
    query_model_uri: str = YANDEX_EMBEDDING_QUERY_MODEL_URI,
    folder_id: str = YANDEX_FOLDER_ID,
    api_key: str | None = YANDEX_API_KEY,
    iam_token: str | None = YANDEX_IAM_TOKEN,
    debug: bool = False,
) -> EmbeddingStats:
    return embed_chunks(
        document_id=document_id,
        batch_size=batch_size,
        limit=limit,
        only_missing=False,
        api_url=api_url,
        doc_model_uri=doc_model_uri,
        query_model_uri=query_model_uri,
        folder_id=folder_id,
        api_key=api_key,
        iam_token=iam_token,
        debug=debug,
    )


def list_documents() -> list[dict]:
    with session_scope() as session:
        documents = list(session.scalars(select(Document).order_by(Document.id.asc())))

        result: list[dict] = []
        for doc in documents:
            total_chunks = session.scalar(
                select(func.count(Chunk.id)).where(Chunk.document_id == doc.id)
            ) or 0
            missing_embeddings = session.scalar(
                select(func.count(Chunk.id)).where(
                    Chunk.document_id == doc.id,
                    Chunk.embedding.is_(None),
                )
            ) or 0

            result.append(
                {
                    "document_id": doc.id,
                    "source_id": doc.source_id,
                    "title": doc.title,
                    "year": doc.year,
                    "region": doc.region,
                    "total_chunks": int(total_chunks),
                    "missing_embeddings": int(missing_embeddings),
                }
            )

        return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute embeddings for chunks via Yandex API and store them in Postgres/pgvector"
    )

    parser.add_argument("--document-id", type=int, help="Embed chunks only for one document.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of texts processed in one loop batch.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of selected chunks.")
    parser.add_argument("--reembed", action="store_true", help="Recompute embeddings even for chunks that already have vectors.")
    parser.add_argument("--list-documents", action="store_true", help="Print documents with chunk counters and exit.")
    parser.add_argument("--debug", action="store_true", help="Print raw Yandex API response.")

    parser.add_argument("--api-url", default=YANDEX_EMBEDDINGS_URL, help="Yandex embeddings API URL.")
    parser.add_argument("--doc-model-uri", default=YANDEX_EMBEDDING_DOC_MODEL_URI, help="Yandex doc embedding model URI.")
    parser.add_argument("--query-model-uri", default=YANDEX_EMBEDDING_QUERY_MODEL_URI, help="Yandex query embedding model URI.")
    parser.add_argument("--folder-id", default=YANDEX_FOLDER_ID, help="Yandex folder id.")
    parser.add_argument("--api-key", default=YANDEX_API_KEY, help="Yandex API key.")
    parser.add_argument("--iam-token", default=YANDEX_IAM_TOKEN, help="Yandex IAM token.")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_documents:
        for item in list_documents():
            print(item)
        return

    if args.document_id is not None and args.reembed:
        stats = reembed_document(
            document_id=args.document_id,
            batch_size=args.batch_size,
            limit=args.limit,
            api_url=args.api_url,
            doc_model_uri=args.doc_model_uri,
            query_model_uri=args.query_model_uri,
            folder_id=args.folder_id,
            api_key=args.api_key,
            iam_token=args.iam_token,
            debug=args.debug,
        )
    else:
        stats = embed_chunks(
            document_id=args.document_id,
            batch_size=args.batch_size,
            limit=args.limit,
            only_missing=not args.reembed,
            api_url=args.api_url,
            doc_model_uri=args.doc_model_uri,
            query_model_uri=args.query_model_uri,
            folder_id=args.folder_id,
            api_key=args.api_key,
            iam_token=args.iam_token,
            debug=args.debug,
        )

    print(
        {
            "document_id": stats.document_id,
            "source_id": stats.source_id,
            "selected_chunks": stats.selected_chunks,
            "embedded_chunks": stats.embedded_chunks,
            "skipped_chunks": stats.skipped_chunks,
        }
    )


if __name__ == "__main__":
    main()