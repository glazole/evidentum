from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "256"))


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    source_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    source_name: Mapped[str] = mapped_column(String(256))
    region: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    title: Mapped[str] = mapped_column(String(512))
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    file_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    mime_type: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)

    checksum: Mapped[str] = mapped_column(String(64), index=True)
    loaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # LLM-enrichment + versioning
    specialty: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    nosology_primary: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    summary_ru: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    previous_document_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    version_delta_ru: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Chunk.chunk_index",
    )

    def __repr__(self) -> str:
        return f"Document(id={self.id}, source_id={self.source_id!r}, title={self.title!r})"


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_chunk_index"),
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_section_title", "section_title"),
        Index("ix_chunks_nosology", "nosology"),
        Index("ix_chunks_specialty", "specialty"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    section_title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)

    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)

    # LLM-enrichment fields
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    nosology: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    specialty: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    topic: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    evidence_level: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(EMBEDDING_DIM),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return f"Chunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})"
