from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal

import requests
from sqlalchemy import Select, select

from app.db import session_scope
from app.models import Chunk, Document
from app.services.embedder import (
    YANDEX_API_KEY,
    YANDEX_FOLDER_ID,
    YANDEX_IAM_TOKEN,
    YANDEX_EMBEDDING_DOC_MODEL_URI,
    YANDEX_EMBEDDING_QUERY_MODEL_URI,
    YANDEX_EMBEDDINGS_URL,
    get_query_embedding,
)

# -----------------------------
# Translation config
# -----------------------------

YANDEX_TRANSLATE_URL = os.getenv(
    "YANDEX_TRANSLATE_URL",
    "https://translate.api.cloud.yandex.net/translate/v2/translate",
)

TRANSLATE_MODE = os.getenv("RETRIEVER_TRANSLATE_MODE", "off")
# off | query | query_and_hits | dual_query

TRANSLATE_TARGET_LANG = os.getenv("RETRIEVER_TRANSLATE_TARGET_LANG", "ru")
TRANSLATE_QUERY_TARGET_LANG = os.getenv("RETRIEVER_TRANSLATE_QUERY_TARGET_LANG", "en")
TRANSLATE_SOURCE_LANG = os.getenv("RETRIEVER_TRANSLATE_SOURCE_LANG", "")
TRANSLATE_TIMEOUT = int(os.getenv("YANDEX_TRANSLATE_TIMEOUT", "60"))
TRANSLATE_MAX_RETRIES = int(os.getenv("YANDEX_TRANSLATE_MAX_RETRIES", "5"))
TRANSLATE_RETRY_BASE_DELAY = float(os.getenv("YANDEX_TRANSLATE_RETRY_BASE_DELAY", "1.5"))
TRANSLATE_MAX_TEXT_LENGTH = int(os.getenv("YANDEX_TRANSLATE_MAX_TEXT_LENGTH", "10000"))

TRANSLATE_CREDENTIAL_PROFILE = os.getenv("YANDEX_CREDENTIAL_PROFILE", "default")
# default | personal

PERSONAL_YANDEX_FOLDER_ID = os.getenv("PERSONAL_YANDEX_FOLDER_ID", "")
PERSONAL_YANDEX_API_KEY = os.getenv("PERSONAL_YANDEX_API_KEY", "")
PERSONAL_YANDEX_IAM_TOKEN = os.getenv("PERSONAL_YANDEX_IAM_TOKEN", "")


TranslateMode = Literal["off", "query", "query_and_hits", "dual_query"]
CredentialProfile = Literal["default", "personal"]


@dataclass
class TranslationResult:
    original_text: str
    translated_text: str
    detected_language_code: str | None = None
    target_language_code: str | None = None
    used_profile: str | None = None


@dataclass
class RetrievalHit:
    chunk_id: int
    document_id: int
    chunk_index: int
    source_id: str
    source_name: str
    title: str
    region: str | None
    year: int | None
    url: str | None
    file_path: str | None
    section_title: str | None
    chunk_text: str
    char_count: int
    score: float
    distance: float
    translated_chunk_text: str | None = None
    translation_detected_language: str | None = None


@dataclass
class RetrievalResult:
    query: str
    effective_query: str
    translate_mode: str
    translation_used: bool
    query_translation: TranslationResult | None
    top_k: int
    hits: list[RetrievalHit]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hits"] = [asdict(hit) for hit in self.hits]
        return data


class YandexTranslateClient:
    def __init__(
        self,
        *,
        api_url: str = YANDEX_TRANSLATE_URL,
        timeout: int = TRANSLATE_TIMEOUT,
        max_retries: int = TRANSLATE_MAX_RETRIES,
        retry_base_delay: float = TRANSLATE_RETRY_BASE_DELAY,
        credential_profile: CredentialProfile = TRANSLATE_CREDENTIAL_PROFILE,
        default_folder_id: str = YANDEX_FOLDER_ID,
        default_api_key: str | None = YANDEX_API_KEY,
        default_iam_token: str | None = YANDEX_IAM_TOKEN,
        personal_folder_id: str = PERSONAL_YANDEX_FOLDER_ID,
        personal_api_key: str | None = PERSONAL_YANDEX_API_KEY,
        personal_iam_token: str | None = PERSONAL_YANDEX_IAM_TOKEN,
        debug: bool = False,
    ) -> None:
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.credential_profile = credential_profile
        self.default_folder_id = default_folder_id or ""
        self.default_api_key = default_api_key or ""
        self.default_iam_token = default_iam_token or ""
        self.personal_folder_id = personal_folder_id or ""
        self.personal_api_key = personal_api_key or ""
        self.personal_iam_token = personal_iam_token or ""
        self.debug = debug
        self.session = requests.Session()

        folder_id, api_key, iam_token = self._resolve_credentials()
        if not folder_id:
            raise ValueError(
                f"Folder id is empty for credential_profile={credential_profile!r}."
            )
        if not api_key and not iam_token:
            raise ValueError(
                f"Neither API key nor IAM token is set for credential_profile={credential_profile!r}."
            )

    def _resolve_credentials(self) -> tuple[str, str, str]:
        if self.credential_profile == "personal":
            return (
                self.personal_folder_id,
                self.personal_api_key,
                self.personal_iam_token,
            )
        return (
            self.default_folder_id,
            self.default_api_key,
            self.default_iam_token,
        )

    def _headers(self) -> dict[str, str]:
        folder_id, api_key, iam_token = self._resolve_credentials()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Api-Key {api_key}"
        else:
            headers["Authorization"] = f"Bearer {iam_token}"
        headers["x-folder-id"] = folder_id
        return headers

    def _folder_id(self) -> str:
        folder_id, _, _ = self._resolve_credentials()
        return folder_id

    def translate(
        self,
        text: str,
        *,
        target_language_code: str,
        source_language_code: str | None = None,
        format_: str = "PLAIN_TEXT",
    ) -> TranslationResult:
        text = (text or "").strip()
        if not text:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                detected_language_code=None,
                target_language_code=target_language_code,
                used_profile=self.credential_profile,
            )

        payload: dict[str, Any] = {
            "folderId": self._folder_id(),
            "texts": [text[:TRANSLATE_MAX_TEXT_LENGTH]],
            "targetLanguageCode": target_language_code,
            "format": format_,
        }
        if source_language_code:
            payload["sourceLanguageCode"] = source_language_code

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
                    print("TRANSLATE STATUS:", response.status_code)
                    print("TRANSLATE BODY:", response.text)
                response.raise_for_status()
                data = response.json()
                translations = data.get("translations") or []
                if not translations:
                    raise ValueError(f"Yandex Translate API returned no translations: {data}")
                first = translations[0]
                translated_text = str(first.get("text") or "").strip()
                detected_language_code = first.get("detectedLanguageCode")
                if not translated_text:
                    raise ValueError(f"Yandex Translate API returned empty text: {data}")
                return TranslationResult(
                    original_text=text,
                    translated_text=translated_text,
                    detected_language_code=detected_language_code,
                    target_language_code=target_language_code,
                    used_profile=self.credential_profile,
                )
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_base_delay * (2 ** (attempt - 1)))

        raise RuntimeError(f"Failed to translate text via Yandex Translate: {last_error}") from last_error


def contains_cyrillic(text: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", text or ""))


def contains_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or ""))


def should_translate_query(query: str, mode: TranslateMode) -> bool:
    if mode in {"off", "query_and_hits"}:
        return False
    # query / dual_query:
    # translate RU or mixed query to English when translation is enabled.
    return contains_cyrillic(query)


def clamp_top_k(value: int) -> int:
    return max(1, min(value, 50))


def build_retrieval_stmt(
    query_embedding: list[float],
    *,
    top_k: int,
    document_id: int | None = None,
    source_id: str | None = None,
    region: str | None = None,
    only_with_embeddings: bool = True,
) -> Select:
    distance_expr = Chunk.embedding.cosine_distance(query_embedding)

    stmt = (
        select(Chunk, Document, distance_expr.label("distance"))
        .join(Document, Document.id == Chunk.document_id)
        .order_by(distance_expr.asc(), Chunk.document_id.asc(), Chunk.chunk_index.asc())
        .limit(clamp_top_k(top_k))
    )

    if only_with_embeddings:
        stmt = stmt.where(Chunk.embedding.is_not(None))
    if document_id is not None:
        stmt = stmt.where(Chunk.document_id == document_id)
    if source_id:
        stmt = stmt.where(Document.source_id == source_id)
    if region:
        stmt = stmt.where(Document.region == region)

    return stmt

def fetch_hits_for_embedding(
    *,
    query_embedding: list[float],
    top_k: int,
    document_id: int | None = None,
    source_id: str | None = None,
    region: str | None = None,
) -> list[RetrievalHit]:
    with session_scope() as session:
        rows = session.execute(
            build_retrieval_stmt(
                query_embedding,
                top_k=top_k,
                document_id=document_id,
                source_id=source_id,
                region=region,
            )
        ).all()

    hits: list[RetrievalHit] = []
    for chunk, document, distance in rows:
        dist = float(distance)
        hits.append(
            RetrievalHit(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                source_id=document.source_id,
                source_name=document.source_name,
                title=document.title,
                region=document.region,
                year=document.year,
                url=document.url,
                file_path=document.file_path,
                section_title=chunk.section_title,
                chunk_text=chunk.chunk_text,
                char_count=chunk.char_count,
                distance=dist,
                score=score_from_distance(dist),
            )
        )
    return hits

def merge_and_rerank_hits(
    *hits_lists: Iterable[RetrievalHit],
    top_k: int,
) -> list[RetrievalHit]:
    best_by_chunk_id: dict[int, RetrievalHit] = {}

    for hits in hits_lists:
        for hit in hits:
            existing = best_by_chunk_id.get(hit.chunk_id)
            if existing is None or hit.distance < existing.distance:
                best_by_chunk_id[hit.chunk_id] = hit

    merged = list(best_by_chunk_id.values())
    merged.sort(key=lambda x: (x.distance, x.document_id, x.chunk_index))
    return merged[:clamp_top_k(top_k)]

def score_from_distance(distance: float) -> float:
    # cosine distance in pgvector: lower is better; similarity-like score for UI.
    score = 1.0 - float(distance)
    if score < -1.0:
        return -1.0
    if score > 1.0:
        return 1.0
    return score


def translate_hits_if_needed(
    hits: Iterable[RetrievalHit],
    *,
    translate_mode: TranslateMode,
    target_language_code: str,
    translator: YandexTranslateClient | None,
    source_language_code: str | None = None,
) -> list[RetrievalHit]:
    if translate_mode != "query_and_hits" or translator is None:
        return list(hits)

    result: list[RetrievalHit] = []
    for hit in hits:
        text = (hit.chunk_text or "").strip()
        if not text:
            result.append(hit)
            continue

        # avoid paying for unnecessary translation if hit is already in target language family
        if target_language_code == "ru" and contains_cyrillic(text):
            result.append(hit)
            continue
        if target_language_code == "en" and contains_latin(text) and not contains_cyrillic(text):
            result.append(hit)
            continue

        translated = translator.translate(
            text,
            target_language_code=target_language_code,
            source_language_code=source_language_code,
        )
        hit.translated_chunk_text = translated.translated_text
        hit.translation_detected_language = translated.detected_language_code
        result.append(hit)
    return result


def retrieve(
    query: str,
    *,
    top_k: int = 5,
    document_id: int | None = None,
    source_id: str | None = None,
    region: str | None = None,
    translate_mode: TranslateMode = TRANSLATE_MODE,  # type: ignore[assignment]
    translate_query_target_lang: str = TRANSLATE_QUERY_TARGET_LANG,
    translate_hits_target_lang: str = TRANSLATE_TARGET_LANG,
    translate_source_lang: str | None = TRANSLATE_SOURCE_LANG or None,
    translate_api_url: str = YANDEX_TRANSLATE_URL,
    translate_credential_profile: CredentialProfile = TRANSLATE_CREDENTIAL_PROFILE,  # type: ignore[assignment]
    embeddings_api_url: str = YANDEX_EMBEDDINGS_URL,
    doc_model_uri: str = YANDEX_EMBEDDING_DOC_MODEL_URI,
    query_model_uri: str = YANDEX_EMBEDDING_QUERY_MODEL_URI,
    folder_id: str = YANDEX_FOLDER_ID,
    api_key: str | None = YANDEX_API_KEY,
    iam_token: str | None = YANDEX_IAM_TOKEN,
    debug: bool = False,
) -> RetrievalResult:
    raw_query = (query or "").strip()
    if not raw_query:
        raise ValueError("Query is empty.")

    query_translation: TranslationResult | None = None
    effective_query = raw_query

    translator: YandexTranslateClient | None = None
    if translate_mode != "off":
        translator = YandexTranslateClient(
            api_url=translate_api_url,
            credential_profile=translate_credential_profile,
            debug=debug,
        )

    use_dual_query = translate_mode == "dual_query" and contains_cyrillic(raw_query)

    ru_hits: list[RetrievalHit] = []
    translated_hits: list[RetrievalHit] = []

    if use_dual_query:
        raw_query_embedding = get_query_embedding(
            raw_query,
            api_url=embeddings_api_url,
            doc_model_uri=doc_model_uri,
            query_model_uri=query_model_uri,
            folder_id=folder_id,
            api_key=api_key,
            iam_token=iam_token,
            debug=debug,
        )
        ru_hits = fetch_hits_for_embedding(
            query_embedding=raw_query_embedding,
            top_k=max(top_k, 8),
            document_id=document_id,
            source_id=source_id,
            region=region,
        )

        if translator is not None:
            query_translation = translator.translate(
                raw_query,
                target_language_code=translate_query_target_lang,
                source_language_code=translate_source_lang,
            )
            effective_query = query_translation.translated_text

            translated_query_embedding = get_query_embedding(
                effective_query,
                api_url=embeddings_api_url,
                doc_model_uri=doc_model_uri,
                query_model_uri=query_model_uri,
                folder_id=folder_id,
                api_key=api_key,
                iam_token=iam_token,
                debug=debug,
            )
            translated_hits = fetch_hits_for_embedding(
                query_embedding=translated_query_embedding,
                top_k=max(top_k, 8),
                document_id=document_id,
                source_id=source_id,
                region=region,
            )

        hits = merge_and_rerank_hits(ru_hits, translated_hits, top_k=top_k)

    else:
        if should_translate_query(raw_query, translate_mode) and translator is not None:
            query_translation = translator.translate(
                raw_query,
                target_language_code=translate_query_target_lang,
                source_language_code=translate_source_lang,
            )
            effective_query = query_translation.translated_text

        query_embedding = get_query_embedding(
            effective_query,
            api_url=embeddings_api_url,
            doc_model_uri=doc_model_uri,
            query_model_uri=query_model_uri,
            folder_id=folder_id,
            api_key=api_key,
            iam_token=iam_token,
            debug=debug,
        )

        hits = fetch_hits_for_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id,
            source_id=source_id,
            region=region,
        )

    hits = translate_hits_if_needed(
        hits,
        translate_mode=translate_mode,
        target_language_code=translate_hits_target_lang,
        translator=translator,
        source_language_code=translate_source_lang,
    )

    return RetrievalResult(
        query=raw_query,
        effective_query=effective_query,
        translate_mode=translate_mode,
        translation_used=query_translation is not None,
        query_translation=query_translation,
        top_k=clamp_top_k(top_k),
        hits=hits,
    )


def format_hits_for_console(result: RetrievalResult) -> str:
    lines: list[str] = []
    lines.append(f"query={result.query!r}")
    lines.append(f"effective_query={result.effective_query!r}")
    lines.append(f"translate_mode={result.translate_mode}")
    lines.append(f"translation_used={result.translation_used}")

    if result.query_translation is not None:
        lines.append(
            "query_translation="
            f"{result.query_translation.original_text!r} -> {result.query_translation.translated_text!r}"
        )

    for idx, hit in enumerate(result.hits, start=1):
        text_preview = (hit.translated_chunk_text or hit.chunk_text or "").replace("\n", " ").strip()
        if len(text_preview) > 400:
            text_preview = text_preview[:400].rstrip() + "..."
        lines.extend(
            [
                "",
                f"[{idx}] score={hit.score:.6f} distance={hit.distance:.6f}",
                f"source_id={hit.source_id} document_id={hit.document_id} chunk_id={hit.chunk_id}",
                f"title={hit.title}",
                f"section={hit.section_title}",
                f"url={hit.url}",
                f"text={text_preview}",
            ]
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve top-k relevant chunks from Postgres/pgvector for a user query"
    )
    parser.add_argument("query", nargs="?", help="User question text.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return.")
    parser.add_argument("--document-id", type=int, help="Search only in one document.")
    parser.add_argument("--source-id", help="Search only in one source_id.")
    parser.add_argument("--region", help="Search only in one region.")
    parser.add_argument(
        "--translate-mode",
        choices=["off", "query", "query_and_hits", "dual_query"],
        default=TRANSLATE_MODE,
        help="Translation strategy: off | query | query_and_hits | dual_query",
    )
    parser.add_argument(
        "--translate-query-target-lang",
        default=TRANSLATE_QUERY_TARGET_LANG,
        help="Target language for query translation, usually en.",
    )
    parser.add_argument(
        "--translate-hits-target-lang",
        default=TRANSLATE_TARGET_LANG,
        help="Target language for hit translation, usually ru.",
    )
    parser.add_argument(
        "--translate-source-lang",
        default=TRANSLATE_SOURCE_LANG,
        help="Optional explicit source language code for translation.",
    )
    parser.add_argument(
        "--translate-profile",
        choices=["default", "personal"],
        default=TRANSLATE_CREDENTIAL_PROFILE,
        help="Which translate credentials to use.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full result as JSON-like dict.",
    )
    parser.add_argument("--debug", action="store_true", help="Print raw provider responses.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.query:
        parser.error("query is required")

    result = retrieve(
        args.query,
        top_k=args.top_k,
        document_id=args.document_id,
        source_id=args.source_id,
        region=args.region,
        translate_mode=args.translate_mode,
        translate_query_target_lang=args.translate_query_target_lang,
        translate_hits_target_lang=args.translate_hits_target_lang,
        translate_source_lang=args.translate_source_lang or None,
        translate_credential_profile=args.translate_profile,
        debug=args.debug,
    )

    if args.json:
        print(result.to_dict())
    else:
        print(format_hits_for_console(result))


if __name__ == "__main__":
    main()
