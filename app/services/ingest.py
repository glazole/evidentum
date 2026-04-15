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


def normalize_heading_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:512]


def is_ru_clinical_recommendation(parsed: ParsedDocument) -> bool:
    source = f"{parsed.source_id} {parsed.source_name} {parsed.title}".lower()
    body = parsed.text[:5000].lower()
    return (
        "кр" in source
        or "clinical_rekom" in source
        or "клиническ" in body
        or "оглавление" in body
    )


def is_kr_heading(paragraph: str) -> bool:
    paragraph = normalize_heading_text(paragraph)
    if not paragraph or len(paragraph) > 500:
        return False
    return KR_SECTION_HEADING_RE.match(paragraph) is not None


def extract_kr_sections_1_to_7(text: str) -> str:
    """
    Keeps only top-level sections 1..7 from Russian clinical recommendations.
    Stops before section 8 / literature / appendices if present.
    """
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return text

    kept: list[str] = []
    in_target_block = False

    for paragraph in paragraphs:
        p = normalize_heading_text(paragraph)

        top_match = KR_TOP_LEVEL_HEADING_RE.match(p)
        if top_match:
            top_num = int(top_match.group("num"))

            if top_num == 1:
                in_target_block = True

            if in_target_block and 1 <= top_num <= 7:
                kept.append(paragraph)
                continue

            if in_target_block and top_num >= 8:
                break

        if in_target_block:
            upper_p = p.upper()
            if upper_p.startswith("СПИСОК ЛИТЕРАТУРЫ") or upper_p.startswith("ПРИЛОЖЕНИЕ"):
                break
            kept.append(paragraph)

    if not kept:
        return text

    return normalize_whitespace("\n\n".join(kept))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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


def chunk_text(
    text: str,
    *,
    target_chars: int = 1200,
    overlap_chars: int = 200,
    min_chars: int = 150,
    min_words: int = 5,
    prefer_kr_headings: bool = False,
) -> list[dict]:
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[dict] = []
    current_section: Optional[str] = None
    current_parts: list[str] = []
    current_len = 0

    def is_valid_chunk(chunk_body: str) -> bool:
        chunk_body = chunk_body.strip()
        if not chunk_body:
            return False
        if len(chunk_body) < min_chars:
            return False
        if len(chunk_body.split()) < min_words:
            return False
        return True

    def append_chunk(chunk_body: str) -> None:
        chunk_body = chunk_body.strip()
        if not is_valid_chunk(chunk_body):
            return

        chunks.append(
            {
                "section_title": current_section,
                "chunk_text": chunk_body,
                "char_count": len(chunk_body),
            }
        )

    def flush() -> None:
        nonlocal current_parts, current_len
        chunk_body = "\n\n".join(current_parts).strip()
        append_chunk(chunk_body)
        current_parts = []
        current_len = 0

    for paragraph in paragraphs:
        normalized_paragraph = normalize_heading_text(paragraph)

        heading_detected = (
            is_kr_heading(normalized_paragraph)
            if prefer_kr_headings
            else is_heading(normalized_paragraph)
        )

        if heading_detected:
            current_section = normalized_paragraph[:512]
            if current_parts:
                flush()
            continue

        paragraph_len = len(paragraph)

        if paragraph_len > target_chars:
            start = 0
            while start < paragraph_len:
                end = min(start + target_chars, paragraph_len)
                part = paragraph[start:end].strip()
                if part:
                    if current_parts:
                        flush()
                    append_chunk(part)
                if end >= paragraph_len:
                    break
                start = max(end - overlap_chars, start + 1)
            continue

        projected = current_len + paragraph_len + (2 if current_parts else 0)

        if projected <= target_chars:
            current_parts.append(paragraph)
            current_len = projected
        else:
            flush()
            if overlap_chars > 0 and chunks:
                tail = chunks[-1]["chunk_text"][-overlap_chars:].strip()
                if tail:
                    current_parts = [tail, paragraph]
                    current_len = len(tail) + len(paragraph) + 2
                else:
                    current_parts = [paragraph]
                    current_len = paragraph_len
            else:
                current_parts = [paragraph]
                current_len = paragraph_len

    if current_parts:
        flush()

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

        text_for_chunks = parsed.text
        prefer_kr_headings = False

        if is_ru_clinical_recommendation(parsed):
            text_for_chunks = extract_kr_sections_1_to_7(parsed.text)
            prefer_kr_headings = True

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

        chunks = chunk_text(
            text_for_chunks,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
            prefer_kr_headings=prefer_kr_headings,
        )

        for idx, chunk in enumerate(chunks):
            session.add(
                Chunk(
                    document_id=document.id,
                    chunk_index=idx,
                    section_title=chunk["section_title"],
                    chunk_text=chunk["chunk_text"],
                    char_count=chunk["char_count"],
                    metadata_json={
                        "source_id": parsed.source_id,
                        "source_name": parsed.source_name,
                        "title": parsed.title,
                        "url": parsed.url,
                        "region": parsed.region,
                        "year": parsed.year,
                        "file_path": parsed.file_path,
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