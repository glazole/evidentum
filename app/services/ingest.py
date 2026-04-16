from __future__ import annotations

import argparse
import hashlib
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sqlalchemy import delete, select

from app.db import init_db, session_scope
from app.models import Chunk, Document


USER_AGENT = "Mozilla/5.0 (compatible; MedGuidesIngestor/1.0)"
DEFAULT_TIMEOUT = 60


@dataclass
class ParsedDocument:
    title: str
    text: str
    mime_type: Optional[str]
    checksum: str
    source_name: str
    source_id: str
    region: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    file_path: Optional[str] = None
    metadata_json: Optional[dict] = None

KR_SECTION_HEADING_RE = re.compile(
    r"^(?P<num>[1-9](?:\.\d+)*)(?:[.)])?\s+(?P<title>\S.*)$"
)

KR_TOP_LEVEL_HEADING_RE = re.compile(
    r"^(?P<num>[1-9])(?:[.)])?\s+(?P<title>\S.*)$"
)

KR_CANONICAL_SECTION_TITLES = [
    "1. Краткая информация по заболеванию или состоянию (группе заболеваний или состояний)",
    "2. Диагностика заболевания или состояния (группы заболеваний или состояний), медицинские показания и противопоказания к применению методов диагностики",
    "3. Лечение, включая медикаментозную и немедикаментозную терапии, диетотерапию, обезболивание, медицинские показания и противопоказания к применению методов лечения",
    "4. Медицинская реабилитация и санаторно-курортное лечение, медицинские показания и противопоказания к применению методов медицинской реабилитации, в том числе основанных на использовании природных лечебных факторов",
    "5. Профилактика и диспансерное наблюдение, медицинские показания и противопоказания к применению методов профилактики",
    "6. Организация оказания медицинской помощи",
    "7. Дополнительная информация (в том числе факторы, влияющие на исход заболевания или состояния)",
]

def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def normalize_kr_title_for_match(text: str) -> str:
    text = normalize_whitespace(text)
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = text.replace("̆", "")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text

KR_CANONICAL_SECTION_MAP = {
    normalize_kr_title_for_match(title): title
    for title in KR_CANONICAL_SECTION_TITLES
}

KNOWN_CANONICAL_URLS = {
    "ru_af": "https://cr.minzdrav.gov.ru/preview-cr/382_2",
    "esc_af_2024": "https://academic.oup.com/eurheartj/article/45/36/3314/7738779",
    "aha_af_2023": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001193",
}


def infer_document_identity(
    *,
    file_stem: str,
    text: str,
    title: str,
    url: Optional[str] = None,
    explicit_source_id: Optional[str] = None,
    explicit_source_name: Optional[str] = None,
    explicit_region: Optional[str] = None,
) -> dict:
    """
    Автоопределение region/source_id/source_name/url.
    Приоритет:
    1) явные аргументы пользователя
    2) точные правила под известные документы MVP
    3) более общие эвристики по домену / содержимому
    4) fallback
    """
    norm_stem = normalize_whitespace(file_stem).lower()
    norm_title = normalize_whitespace(title).lower()
    norm_url = (url or "").lower()
    text_head = normalize_whitespace(text[:8000]).lower()

    haystack = " | ".join(
        part for part in [norm_stem, norm_title, norm_url, text_head] if part
    )

    detected_source_id: Optional[str] = None
    detected_source_name: Optional[str] = None
    detected_region: Optional[str] = None
    detected_url: Optional[str] = url

    # -----------------------------
    # 1. Точные правила под текущий MVP
    # -----------------------------

    # Российские КР по ФП
    if (
        "preview-cr/382_2" in norm_url
        or "кр382" in haystack
        or "kp382" in haystack
        or ("фибрилляц" in haystack and "клиническ" in haystack and "минздрав" in haystack)
    ):
        detected_source_id = "кр382_2_new"
        detected_source_name = "Клинические рекомендации Минздрав РФ — Фибрилляция предсердий"
        detected_region = "RU"
        detected_url = KNOWN_CANONICAL_URLS["ru_af"]

    # ESC 2024
    elif (
        "7738779" in norm_url
        or "ehae176" in haystack
        or "2024 esc guidelines for the management" in haystack
        or ("esc" in haystack and "atrial fibrillation" in haystack)
        or ("eur heart j" in haystack and "atrial fibrillation" in haystack)
    ):
        detected_source_id = "ehae176_new"
        detected_source_name = "2024 ESC Guidelines for the management of atrial fibrillation"
        detected_region = "EU"
        detected_url = KNOWN_CANONICAL_URLS["esc_af_2024"]

    # AHA/ACC/HRS 2023
    elif (
        "cir.0000000000001193" in norm_url
        or "joglar" in haystack
        or "acc/aha/accp/hrs" in haystack
        or "2023 acc/aha/accp/hrs guideline" in haystack
        or ("circulation" in haystack and "atrial fibrillation" in haystack)
    ):
        detected_source_id = "joglar-et-al-2023-2023-acc-aha-accp-hrs-guideline_new"
        detected_source_name = "2023 ACC/AHA/ACCP/HRS Guideline for the Diagnosis and Management of Atrial Fibrillation"
        detected_region = "US"
        detected_url = KNOWN_CANONICAL_URLS["aha_af_2023"]

    # -----------------------------
    # 2. Более общие эвристики
    # -----------------------------
    else:
        if "cr.minzdrav.gov.ru" in norm_url or "минздрав" in haystack:
            detected_region = "RU"
        elif any(domain in norm_url for domain in ["escardio.org", "academic.oup.com"]):
            detected_region = "EU"
        elif any(domain in norm_url for domain in ["ahajournals.org", "professional.heart.org", "acc.org"]):
            detected_region = "US"

    final_source_id = explicit_source_id or detected_source_id or slugify(file_stem)
    final_source_name = explicit_source_name or detected_source_name or file_stem
    final_region = explicit_region or detected_region
    final_url = url or detected_url

    return {
        "source_id": final_source_id,
        "source_name": final_source_name,
        "region": final_region,
        "url": final_url,
    }


def is_likely_section_header(line: str, prev_line: Optional[str] = None) -> bool:
    """Более строгая эвристика определения заголовка раздела."""
    line = normalize_whitespace(line)
    if not line:
        return False

    if len(line) > 180:
        return False

    low = line.lower()

    # Явный шум / куски таблиц / ссылок / уровни доказательности
    noise_patterns = [
        r"\b95% ci\b",
        r"\bhr\b",
        r"\bp\s*[=<]",
        r"\bnct\d+\b",
        r"\bet al\b",
        r"\[\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*\]",
        r"^\d+[\d\s.,;%–\-]*$",
        r"^(еок|уур|удд)\b",
        r"^\d+(?:[–\-]\d+)?\s*,\s*\d+$",
        r"^\d+%.*$",
    ]
    if any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in noise_patterns):
        return False

    # Канонические верхнеуровневые русские разделы
    if KR_TOP_LEVEL_HEADING_RE.match(line):
        return True

    normalized = normalize_kr_title_for_match(line)
    if normalized in KR_CANONICAL_SECTION_MAP:
        return True

    # Английские / общие явные заголовки
    if re.match(r"^(chapter|section|part)\b", low):
        return True
    if re.match(r"^(глава|раздел|часть)\b", low):
        return True

    # Короткие UPPERCASE заголовки
    if line.isupper() and 4 <= len(line) <= 80 and 1 <= len(line.split()) <= 10:
        return True

    # Осторожная эвристика по пустой строке сверху
    # Только если строка не слишком длинная, без явных цифро-табличных паттернов
    if prev_line is not None and not prev_line.strip():
        if len(line) <= 90 and line[0].isupper():
            if not re.search(r"\d{2,}", line) and len(line.split()) <= 12:
                return True

    return False

def split_into_sections_by_headers(text: str) -> list[dict]:
    """Разбивает текст на секции, используя is_likely_section_header."""
    lines = text.splitlines()
    if not lines:
        return []

    sections = []
    current_title = None
    current_content = []
    prev_line = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            # Пустые строки не добавляем в контент, но запоминаем для эвристики
            prev_line = line
            continue

        is_header = is_likely_section_header(line, prev_line)

        if is_header:
            # Сохраняем предыдущую секцию
            if current_title:
                content = "\n".join(current_content).strip()
                if content:
                    sections.append({
                        "section_title": current_title,
                        "section_text": content
                    })
            # Начинаем новую
            current_title = line[:512]
            current_content = []
        else:
            # Если мы ещё не внутри секции (начало документа), пропускаем текст до первого заголовка
            if current_title is not None:
                current_content.append(raw_line)  # сохраняем исходный перенос строк

        prev_line = line

    # Последняя секция
    if current_title and current_content:
        content = "\n".join(current_content).strip()
        if content:
            sections.append({
                "section_title": current_title,
                "section_text": content
            })

    return sections

def is_toc_or_reference_section(section_title: str) -> bool:
    """Проверяет, является ли секция оглавлением, списком литературы, приложением и т.п."""
    title = section_title.lower()
    # Список стоп-слов (русские и английские)
    stop_patterns = [
        r'^оглавление$', r'^содержание$', r'^список литературы$', r'^references?$',
        r'^приложение', r'^appendix', r'^table of contents', r'^список сокращений',
        r'^термины и определения',  # часто идёт перед разделами
    ]
    for pattern in stop_patterns:
        if re.match(pattern, title):
            return True
    return False

def filter_sections(sections: list[dict], keep_top_level_numbers: Optional[list[str]] = None) -> list[dict]:
    """
    Фильтрует секции:
    - удаляет секции, помеченные как оглавление/литература
    - если указан keep_top_level_numbers, оставляет только секции, чей заголовок начинается с этих чисел
    """
    filtered = []
    for sec in sections:
        if is_toc_or_reference_section(sec["section_title"]):
            continue
        if keep_top_level_numbers:
            # Проверяем, начинается ли заголовок с одного из номеров (например, "1.", "2.")
            title = sec["section_title"]
            if any(title.startswith(num) for num in keep_top_level_numbers):
                filtered.append(sec)
        else:
            filtered.append(sec)
    return filtered


def normalize_heading_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:512]

def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w\-]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "document"


def guess_year(text: str) -> Optional[int]:
    years = re.findall(r"\b(20\d{2}|19\d{2})\b", text[:4000])
    if not years:
        return None
    year_ints = [int(y) for y in years]
    plausible = [y for y in year_ints if 1900 <= y <= 2100]
    return max(plausible) if plausible else None


def detect_title(text: str, fallback: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:15]:
        if len(line) >= 10:
            return line[:512]
    return fallback[:512]


def is_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if len(line) > 120:
        return False
    if re.match(r"^\d+(\.\d+)*[.)]?\s+\S+", line):
        return True
    if line.isupper() and len(line.split()) <= 12:
        return True
    if re.match(r"^(chapter|section|part)\b", line.lower()):
        return True
    if re.match(r"^(глава|раздел|часть)\b", line.lower()):
        return True
    return False


def extract_pdf_text(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text)
    return normalize_whitespace("\n\n".join(pages))


def extract_txt_text(file_path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return normalize_whitespace(file_path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Could not decode file: {file_path}")


def extract_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    return normalize_whitespace(text)


def download_file(url: str, target_dir: Path) -> tuple[Path, Optional[str], bytes]:
    target_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(
        url,
        timeout=DEFAULT_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").split(";")[0].strip() or None
    parsed = urlparse(url)

    filename = Path(parsed.path).name or "downloaded_file"
    if "." not in filename and content_type:
        ext = mimetypes.guess_extension(content_type) or ""
        filename += ext

    file_path = target_dir / filename
    file_path.write_bytes(response.content)

    return file_path, content_type, response.content


def parse_local_file(
    file_path: Path,
    *,
    source_id: Optional[str] = None,
    source_name: Optional[str] = None,
    region: Optional[str] = None,
    url: Optional[str] = None,
) -> ParsedDocument:
    mime_type, _ = mimetypes.guess_type(str(file_path))
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        text = extract_pdf_text(file_path)
    elif suffix in {".txt", ".md"}:
        text = extract_txt_text(file_path)
    elif suffix in {".html", ".htm"}:
        text = extract_html_text(file_path.read_text(encoding="utf-8", errors="ignore"))
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    if not text.strip():
        raise ValueError(f"Extracted empty text from file: {file_path}")

    title = detect_title(text, file_path.stem)
    checksum = sha256_text(text)
    doc_source_id = source_id or slugify(file_path.stem)
    doc_source_name = source_name or file_path.stem

    return ParsedDocument(
        title=title,
        text=text,
        mime_type=mime_type,
        checksum=checksum,
        source_name=doc_source_name,
        source_id=doc_source_id,
        region=region,
        year=guess_year(text),
        url=url,
        file_path=str(file_path),
        metadata_json={"origin": "local_file"},
    )


def parse_url(
    url: str,
    *,
    download_dir: Path,
    source_id: Optional[str] = None,
    source_name: Optional[str] = None,
    region: Optional[str] = None,
) -> ParsedDocument:
    file_path, content_type, raw_bytes = download_file(url, download_dir)

    if content_type == "application/pdf" or file_path.suffix.lower() == ".pdf":
        text = extract_pdf_text(file_path)
    elif content_type in {"text/html", "application/xhtml+xml"} or file_path.suffix.lower() in {".html", ".htm"}:
        html = raw_bytes.decode("utf-8", errors="ignore")
        text = extract_html_text(html)
    else:
        # fallback: try as text
        try:
            text = normalize_whitespace(raw_bytes.decode("utf-8"))
        except UnicodeDecodeError:
            text = extract_txt_text(file_path)

    if not text.strip():
        raise ValueError(f"Extracted empty text from URL: {url}")

    default_name = Path(urlparse(url).path).stem or "web_document"
    title = detect_title(text, default_name)
    checksum = sha256_text(text)
    doc_source_id = source_id or slugify(default_name)
    doc_source_name = source_name or default_name

    return ParsedDocument(
        title=title,
        text=text,
        mime_type=content_type,
        checksum=checksum,
        source_name=doc_source_name,
        source_id=doc_source_id,
        region=region,
        year=guess_year(text),
        url=url,
        file_path=str(file_path),
        metadata_json={"origin": "url"},
    )


def split_into_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs


def split_section_text(
    text: str,
    *,
    target_chars: int = 1200,
    overlap_chars: int = 200,
    min_chars: int = 80,
    min_words: int = 5,
) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + target_chars, text_len)

        if end < text_len:
            cut = text.rfind(" ", start, end)
            if cut > start + target_chars // 2:
                end = cut

        part = text[start:end].strip()
        if part and len(part) >= min_chars and len(part.split()) >= min_words:
            chunks.append(part)

        if end >= text_len:
            break

        start = max(end - overlap_chars, start + 1)

    return chunks

def split_text_by_size(
    text: str,
    *,
    target_chars: int = 1200,
    overlap_chars: int = 200,
    min_chars: int = 150,
    min_words: int = 5,
) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + target_chars, text_len)

        if end < text_len:
            cut = text.rfind(" ", start, end)
            if cut > start + target_chars // 2:
                end = cut

        part = text[start:end].strip()
        if part and len(part) >= min_chars and len(part.split()) >= min_words:
            chunks.append(part)

        if end >= text_len:
            break

        start = max(end - overlap_chars, start + 1)

    return chunks

def detect_document_sections(parsed: ParsedDocument) -> list[dict]:
    text = normalize_whitespace(parsed.text)
    if not text:
        return []

    sections = split_into_sections_by_headers(text)

    # fallback: если секции не найдены, режем весь текст как один "безымянный" блок
    if not sections:
        return [{"section_title": None, "section_text": text}]

    # для российских КР убираем мусорные секции и оставляем полезные верхние уровни
    if (parsed.region or "").upper() == "RU":
        sections = filter_sections(
            sections,
            keep_top_level_numbers=["1.", "2.", "3.", "4.", "5.", "6.", "7."],
        )
    else:
        sections = filter_sections(sections)

    if not sections:
        return [{"section_title": None, "section_text": text}]

    return sections


def build_chunks_from_sections(
    parsed: ParsedDocument,
    *,
    target_chars: int = 1200,
    overlap_chars: int = 200,
    min_chars: int = 150,
    min_words: int = 5,
) -> list[dict]:
    sections = detect_document_sections(parsed)

    chunks: list[dict] = []
    for sec in sections:
        section_title = normalize_heading_text(sec.get("section_title") or "") or None
        section_text = sec.get("section_text") or ""

        for chunk_body in split_section_text(
            section_text,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
            min_chars=min_chars,
            min_words=min_words,
        ):
            embedding_text = chunk_body
            if section_title:
                embedding_text = f"{section_title}\n\n{chunk_body}"

            chunks.append(
                {
                    "section_title": section_title,
                    "chunk_text": chunk_body,
                    "embedding_text": embedding_text,
                    "char_count": len(chunk_body),
                }
            )

    # fallback на старый режим, если вдруг всё отфильтровалось
    if not chunks:
        for chunk in chunk_text(
            parsed.text,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
            min_chars=min_chars,
            min_words=min_words,
        ):
            chunks.append(
                {
                    "section_title": chunk.get("section_title"),
                    "chunk_text": chunk["chunk_text"],
                    "embedding_text": chunk["chunk_text"],
                    "char_count": chunk["char_count"],
                }
            )

    return chunks

def chunk_text(
    text: str,
    *,
    target_chars: int = 1200,
    overlap_chars: int = 200,
    min_chars: int = 150,
    min_words: int = 5,
) -> list[dict]:
    text = normalize_whitespace(text)
    if not text:
        return []

    text_chunks = split_text_by_size(
        text,
        target_chars=target_chars,
        overlap_chars=overlap_chars,
        min_chars=min_chars,
        min_words=min_words,
    )

    chunks: list[dict] = []
    for chunk_body in text_chunks:
        chunks.append(
            {
                "section_title": None,
                "chunk_text": chunk_body,
                "char_count": len(chunk_body),
            }
        )

    return chunks

def upsert_document(
    parsed: ParsedDocument,
    *,
    force_reingest: bool = False,
    target_chars: int = 1200,
    overlap_chars: int = 200,
) -> tuple[Document, int, bool]:
    with session_scope() as session:
        existing = session.scalar(
            select(Document).where(Document.source_id == parsed.source_id)
        )

        if existing and existing.checksum == parsed.checksum and not force_reingest:
            return existing, len(existing.chunks), False

        if existing:
            existing.source_name = parsed.source_name
            existing.region = parsed.region
            existing.title = parsed.title
            existing.year = parsed.year
            existing.url = parsed.url
            existing.file_path = parsed.file_path
            existing.mime_type = parsed.mime_type
            existing.raw_text = parsed.text
            existing.metadata_json = parsed.metadata_json or {}
            existing.checksum = parsed.checksum
            document = existing

            session.execute(delete(Chunk).where(Chunk.document_id == existing.id))
            session.flush()
        else:
            document = Document(
                source_id=parsed.source_id,
                source_name=parsed.source_name,
                region=parsed.region,
                title=parsed.title,
                year=parsed.year,
                url=parsed.url,
                file_path=parsed.file_path,
                mime_type=parsed.mime_type,
                raw_text=parsed.text,
                metadata_json=parsed.metadata_json or {},
                checksum=parsed.checksum,
            )
            session.add(document)
            session.flush()

        chunks = build_chunks_from_sections(
            parsed,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
        )

        for idx, chunk in enumerate(chunks):
            session.add(
                Chunk(
                    document_id=document.id,
                    chunk_index=idx,
                    section_title=chunk["section_title"],
                    chunk_text=chunk["embedding_text"],   # section_title участвует в эмбеддингах
                    char_count=chunk["char_count"],
                    metadata_json={
                        "source_id": parsed.source_id,
                        "source_name": parsed.source_name,
                        "title": parsed.title,
                        "url": parsed.url,
                        "region": parsed.region,
                        "year": parsed.year,
                        "file_path": parsed.file_path,
                        "section_title": chunk["section_title"],
                        "display_text": chunk["chunk_text"],  # для UI/ответа можно брать чистый текст
                    },
                )
            )

        session.flush()
        return document, len(chunks), True

def ingest_file(
    file_path: str,
    *,
    source_id: Optional[str] = None,
    source_name: Optional[str] = None,
    region: Optional[str] = None,
    url: Optional[str] = None,
    force_reingest: bool = False,
    target_chars: int = 1200,
    overlap_chars: int = 200,
) -> dict:
    parsed = parse_local_file(
        Path(file_path),
        source_id=source_id,
        source_name=source_name,
        region=region,
        url=url,
    )
    document, chunk_count, updated = upsert_document(
        parsed,
        force_reingest=force_reingest,
        target_chars=target_chars,
        overlap_chars=overlap_chars,
    )
    return {
        "document_id": document.id,
        "source_id": document.source_id,
        "title": document.title,
        "chunk_count": chunk_count,
        "updated": updated,
    }


def ingest_url(
    url: str,
    *,
    download_dir: str = "data/raw",
    source_id: Optional[str] = None,
    source_name: Optional[str] = None,
    region: Optional[str] = None,
    force_reingest: bool = False,
    target_chars: int = 1200,
    overlap_chars: int = 200,
) -> dict:
    parsed = parse_url(
        url,
        download_dir=Path(download_dir),
        source_id=source_id,
        source_name=source_name,
        region=region,
    )
    document, chunk_count, updated = upsert_document(
        parsed,
        force_reingest=force_reingest,
        target_chars=target_chars,
        overlap_chars=overlap_chars,
    )
    return {
        "document_id": document.id,
        "source_id": document.source_id,
        "title": document.title,
        "chunk_count": chunk_count,
        "updated": updated,
    }


def ingest_many_local_files(
    paths: Iterable[str],
    *,
    force_reingest: bool = False,
    target_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[dict]:
    results: list[dict] = []
    for path in paths:
        result = ingest_file(
            path,
            force_reingest=force_reingest,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
        )
        results.append(result)
    return results

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest documents into Postgres/pgvector")

    parser.add_argument("--file", action="append", help="Local file path. Can be passed multiple times.")
    parser.add_argument("--url", action="append", help="Document URL. Can be passed multiple times.")
    parser.add_argument("--dir", help="Path to directory with files.")
    parser.add_argument("--source-id", help="Override source_id for single file/url ingest.")
    parser.add_argument("--source-name", help="Override source_name for single file/url ingest.")
    parser.add_argument("--region", help="Region tag, e.g. RU / EU / US.")
    parser.add_argument("--download-dir", default="data/raw", help="Directory for downloaded files.")
    parser.add_argument("--target-chars", type=int, default=1200, help="Approx target chunk size.")
    parser.add_argument("--overlap-chars", type=int, default=200, help="Chunk overlap.")
    parser.add_argument("--force-reingest", action="store_true", help="Rebuild document and chunks even if checksum matches.")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.file and not args.url and not args.dir:
        parser.error("At least one --file, --url, or --dir must be provided.")

    init_db()

    if args.file:
        for file_path in args.file:
            result = ingest_file(
                file_path=file_path,
                source_id=args.source_id,
                source_name=args.source_name,
                region=args.region,
                force_reingest=args.force_reingest,
                target_chars=args.target_chars,
                overlap_chars=args.overlap_chars,
            )
            print(result)

    if args.dir:
        dir_path = Path(args.dir)

        if not dir_path.exists():
            parser.error(f"Directory does not exist: {args.dir}")

        if not dir_path.is_dir():
            parser.error(f"Path is not a directory: {args.dir}")

        for file_path in sorted(dir_path.iterdir()):
            if not file_path.is_file():
                continue

            result = ingest_file(
                file_path=str(file_path),
                source_id=args.source_id,
                source_name=args.source_name,
                region=args.region,
                force_reingest=args.force_reingest,
                target_chars=args.target_chars,
                overlap_chars=args.overlap_chars,
            )
            print(result)

    if args.url:
        for url in args.url:
            result = ingest_url(
                url=url,
                download_dir=args.download_dir,
                source_id=args.source_id,
                source_name=args.source_name,
                region=args.region,
                force_reingest=args.force_reingest,
                target_chars=args.target_chars,
                overlap_chars=args.overlap_chars,
            )
            print(result)


if __name__ == "__main__":
    main()