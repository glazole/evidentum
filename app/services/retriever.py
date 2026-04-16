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
    # LLM-enrichment fields
    summary: str | None = None
    nosology: str | None = None
    specialty: str | None = None
    topic: str | None = None
    evidence_level: str | None = None
    # retrieval metadata
    retrieval_source: list[str] | None = None   # raw_query | translated_query
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

RAW_CANDIDATE_MIN = int(os.getenv("RETRIEVER_RAW_CANDIDATE_MIN", "24"))
RAW_CANDIDATE_MULTIPLIER = int(os.getenv("RETRIEVER_RAW_CANDIDATE_MULTIPLIER", "4"))
MAX_HITS_PER_DOCUMENT = int(os.getenv("RETRIEVER_MAX_HITS_PER_DOCUMENT", "2"))

RU_PRIORITY_BONUS = float(os.getenv("RETRIEVER_RU_PRIORITY_BONUS", "0.05"))
SECTION_PREFERRED_BONUS = float(os.getenv("RETRIEVER_SECTION_PREFERRED_BONUS", "0.04"))
SECTION_NOISE_PENALTY = float(os.getenv("RETRIEVER_SECTION_NOISE_PENALTY", "0.08"))
LEXICAL_OVERLAP_MAX_BONUS = float(os.getenv("RETRIEVER_LEXICAL_OVERLAP_MAX_BONUS", "0.08"))

def raw_candidate_top_k(top_k: int) -> int:
    return max(clamp_top_k(top_k) * RAW_CANDIDATE_MULTIPLIER, RAW_CANDIDATE_MIN)

def build_retrieval_stmt(
    query_embedding: list[float],
    *,
    top_k: int,
    document_id: int | None = None,
    source_id: str | None = None,
    region: str | None = None,
    specialty: str | None = None,
    nosology: str | None = None,
    only_with_embeddings: bool = True,
    per_source_k: int | None = None,
) -> Select:
    from sqlalchemy import func as _func

    distance_expr = Chunk.embedding.cosine_distance(query_embedding)

    if per_source_k is not None:
        rn_expr = _func.row_number().over(
            partition_by=Document.source_id,
            order_by=distance_expr.asc(),
        ).label("rn")
        inner = (
            select(
                Chunk.id.label("chunk_id"),
                Chunk.document_id,
                Chunk.chunk_index,
                Chunk.section_title,
                Chunk.chunk_text,
                Chunk.char_count,
                Chunk.summary,
                Chunk.nosology,
                Chunk.specialty,
                Chunk.topic,
                Chunk.evidence_level,
                Document.id.label("doc_id"),
                Document.source_id,
                Document.source_name,
                Document.title,
                Document.region,
                Document.year,
                Document.url,
                Document.file_path,
                distance_expr.label("distance"),
                rn_expr,
            )
            .join(Document, Document.id == Chunk.document_id)
        )
        if only_with_embeddings:
            inner = inner.where(Chunk.embedding.is_not(None))
        if document_id is not None:
            inner = inner.where(Chunk.document_id == document_id)
        if source_id:
            inner = inner.where(Document.source_id == source_id)
        if region:
            inner = inner.where(Document.region == region)
        if specialty:
            inner = inner.where(Chunk.specialty == specialty)
        if nosology:
            inner = inner.where(Chunk.nosology.ilike(f"%{nosology}%"))
        subq = inner.subquery()
        return (
            select(subq)
            .where(subq.c.rn <= per_source_k)
            .order_by(subq.c.distance.asc())
            .limit(clamp_top_k(top_k))
        )

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
    if specialty:
        stmt = stmt.where(Chunk.specialty == specialty)
    if nosology:
        stmt = stmt.where(Chunk.nosology.ilike(f"%{nosology}%"))

    return stmt


def _hit_from_row(chunk: Any, document: Any, distance: float, retrieval_source: str) -> RetrievalHit:
    dist = float(distance)
    return RetrievalHit(
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
        summary=getattr(chunk, "summary", None),
        nosology=getattr(chunk, "nosology", None),
        specialty=getattr(chunk, "specialty", None),
        topic=getattr(chunk, "topic", None),
        evidence_level=getattr(chunk, "evidence_level", None),
        distance=dist,
        score=score_from_distance(dist),
        retrieval_source=[retrieval_source],
    )


def fetch_hits_for_embedding(
    *,
    query_embedding: list[float],
    top_k: int,
    retrieval_source: str,
    document_id: int | None = None,
    source_id: str | None = None,
    region: str | None = None,
    specialty: str | None = None,
    nosology: str | None = None,
    per_source_k: int | None = None,
) -> list[RetrievalHit]:
    stmt = build_retrieval_stmt(
        query_embedding,
        top_k=top_k,
        document_id=document_id,
        source_id=source_id,
        region=region,
        specialty=specialty,
        nosology=nosology,
        per_source_k=per_source_k,
    )

    with session_scope() as session:
        if per_source_k is not None:
            rows_mapped = session.execute(stmt).mappings().all()
            hits: list[RetrievalHit] = []
            for row in rows_mapped:
                dist = float(row["distance"])
                hits.append(
                    RetrievalHit(
                        chunk_id=row["chunk_id"],
                        document_id=row["document_id"],
                        chunk_index=row["chunk_index"],
                        source_id=row["source_id"],
                        source_name=row["source_name"],
                        title=row["title"],
                        region=row["region"],
                        year=row["year"],
                        url=row["url"],
                        file_path=row["file_path"],
                        section_title=row["section_title"],
                        chunk_text=row["chunk_text"],
                        char_count=row["char_count"],
                        summary=row.get("summary"),
                        nosology=row.get("nosology"),
                        specialty=row.get("specialty"),
                        topic=row.get("topic"),
                        evidence_level=row.get("evidence_level"),
                        distance=dist,
                        score=score_from_distance(dist),
                        retrieval_source=[retrieval_source],
                    )
                )
            return hits

        rows = session.execute(stmt).all()

    hits = []
    for chunk, document, distance in rows:
        hits.append(_hit_from_row(chunk, document, float(distance), retrieval_source))
    return hits

def merge_and_rerank_hits(
    *hits_lists: Iterable[RetrievalHit],
    top_k: int,
    query: str,
) -> list[RetrievalHit]:
    best_by_chunk_id: dict[int, RetrievalHit] = {}

    for hits in hits_lists:
        for hit in hits:
            existing = best_by_chunk_id.get(hit.chunk_id)

            if existing is None:
                best_by_chunk_id[hit.chunk_id] = hit
                continue

            existing_sources = set(existing.retrieval_source or [])
            new_sources = set(hit.retrieval_source or [])
            merged_sources = list(existing_sources | new_sources)

            # оставляем лучший по distance hit
            if hit.distance < existing.distance:
                hit.retrieval_source = merged_sources
                best_by_chunk_id[hit.chunk_id] = hit
            else:
                existing.retrieval_source = merged_sources

    query_terms = extract_query_terms(query)

    merged_hits = list(best_by_chunk_id.values())

    for hit in merged_hits:
        hit.score = rerank_score(hit, query_terms)

    merged_hits.sort(
        key=lambda x: (
            -x.score,
            x.distance,
            x.document_id,
            x.chunk_index,
        )
    )

    merged_hits = limit_hits_per_document(
        merged_hits,
        max_hits_per_document=MAX_HITS_PER_DOCUMENT,
    )

    return merged_hits[:clamp_top_k(top_k)]

def score_from_distance(distance: float) -> float:
    # cosine distance in pgvector: lower is better; similarity-like score for UI.
    score = 1.0 - float(distance)
    if score < -1.0:
        return -1.0
    if score > 1.0:
        return 1.0
    return score

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9\-\+\.]{1,}")

PREFERRED_SECTION_PATTERNS = (
    "лечение",
    "терап",
    "диагност",
    "профилакти",
    "ведение",
    "management",
    "treatment",
    "therapy",
    "diagnos",
    "recommend",
)

NOISE_SECTION_PATTERNS = (
    "оглавление",
    "содержание",
    "список литературы",
    "references",
    "appendix",
    "table of contents",
    "abbreviations",
    "glossary",
)


def extract_query_terms(query: str) -> list[str]:
    """
    Берём только относительно полезные токены из текущего вопроса.
    Это универсально и не привязано к конкретной теме.
    """
    raw_terms = TOKEN_RE.findall((query or "").lower())

    stopwords = {
        "и", "в", "во", "на", "с", "со", "к", "ко", "по", "о", "об", "от", "до",
        "the", "a", "an", "of", "for", "to", "in", "on", "with", "and", "or",
        "как", "что", "при", "для", "или", "у", "из", "над", "под",
        "is", "are", "was", "were", "be", "this", "that",
    }

    terms: list[str] = []
    for term in raw_terms:
        if len(term) < 3:
            continue
        if term in stopwords:
            continue
        if term.isdigit():
            continue
        terms.append(term)

    # сохраняем порядок, убираем дубли
    return list(dict.fromkeys(terms))


def normalize_for_match(text: str | None) -> str:
    return (text or "").lower()


def compute_lexical_overlap_bonus(hit: RetrievalHit, query_terms: list[str]) -> float:
    if not query_terms:
        return 0.0

    haystack = " ".join(
        filter(
            None,
            [
                normalize_for_match(hit.title),
                normalize_for_match(hit.section_title),
                normalize_for_match(hit.chunk_text),
            ],
        )
    )

    matched = 0
    for term in query_terms:
        if term in haystack:
            matched += 1

    if matched == 0:
        return 0.0

    ratio = matched / max(len(query_terms), 1)
    return min(ratio * LEXICAL_OVERLAP_MAX_BONUS, LEXICAL_OVERLAP_MAX_BONUS)


def compute_section_bonus(hit: RetrievalHit) -> float:
    section = normalize_for_match(hit.section_title)

    if not section:
        return 0.0

    if any(pattern in section for pattern in NOISE_SECTION_PATTERNS):
        return -SECTION_NOISE_PENALTY

    if any(pattern in section for pattern in PREFERRED_SECTION_PATTERNS):
        return SECTION_PREFERRED_BONUS

    return 0.0


def compute_region_bonus(hit: RetrievalHit) -> float:
    region = normalize_for_match(hit.region)
    source_id = normalize_for_match(hit.source_id)

    if region == "ru":
        return RU_PRIORITY_BONUS

    # fallback на source_id, если region не заполнен
    if source_id.startswith("ru_") or source_id.startswith("кр") or "minzdrav" in source_id:
        return RU_PRIORITY_BONUS

    return 0.0


def rerank_score(hit: RetrievalHit, query_terms: list[str]) -> float:
    base_score = score_from_distance(hit.distance)
    final_score = (
        base_score
        + compute_region_bonus(hit)
        + compute_section_bonus(hit)
        + compute_lexical_overlap_bonus(hit, query_terms)
    )
    return final_score


def limit_hits_per_document(
    hits: list[RetrievalHit],
    *,
    max_hits_per_document: int = MAX_HITS_PER_DOCUMENT,
) -> list[RetrievalHit]:
    limited: list[RetrievalHit] = []
    per_doc_counter: dict[int, int] = {}

    for hit in hits:
        used = per_doc_counter.get(hit.document_id, 0)
        if used >= max_hits_per_document:
            continue

        limited.append(hit)
        per_doc_counter[hit.document_id] = used + 1

    return limited


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
    specialty: str | None = None,
    nosology: str | None = None,
    per_source_k: int | None = None,
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
            top_k=raw_candidate_top_k(top_k),
            retrieval_source="ru_pass",
            document_id=document_id,
            source_id=source_id,
            region=region,
            specialty=specialty,
            nosology=nosology,
            per_source_k=per_source_k,
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
                top_k=raw_candidate_top_k(top_k),
                retrieval_source="en_pass",
                document_id=document_id,
                source_id=source_id,
                region=region,
                specialty=specialty,
                nosology=nosology,
                per_source_k=per_source_k,
            )

        hits = merge_and_rerank_hits(
            ru_hits,
            translated_hits,
            top_k=top_k,
            query=raw_query,
        )

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
            top_k=raw_candidate_top_k(top_k),
            retrieval_source="translated_query" if query_translation is not None else "raw_query",
            document_id=document_id,
            source_id=source_id,
            region=region,
            specialty=specialty,
            nosology=nosology,
            per_source_k=per_source_k,
        )

        hits = merge_and_rerank_hits(
            hits,
            top_k=top_k,
            query=raw_query,
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

        sources = ",".join(hit.retrieval_source or [])

        lines.extend(
            [
                "",
                f"[{idx}] score={hit.score:.6f} distance={hit.distance:.6f} retrieval_sources={sources}",
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
