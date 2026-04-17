"""initial schema (pgvector, documents, chunks)

Revision ID: 20260414_0001
Revises: 
Create Date: 2026-04-14

"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

import pgvector.sqlalchemy


# revision identifiers, used by Alembic.
revision = "20260414_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("create extension if not exists vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_id", sa.String(length=128), nullable=False),
        sa.Column("source_name", sa.String(length=256), nullable=False),
        sa.Column("region", sa.String(length=64), nullable=True),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column("mime_type", sa.String(length=128), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("checksum", sa.String(length=64), nullable=False),
        sa.Column("loaded_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_id"),
    )
    op.create_index(op.f("ix_documents_checksum"), "documents", ["checksum"], unique=False)
    op.create_index(op.f("ix_documents_source_id"), "documents", ["source_id"], unique=False)

    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("section_title", sa.String(length=512), nullable=True),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=256), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_chunk_index"),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"], unique=False)
    op.create_index("ix_chunks_section_title", "chunks", ["section_title"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_chunks_section_title", table_name="chunks")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_table("chunks")

    op.drop_index(op.f("ix_documents_source_id"), table_name="documents")
    op.drop_index(op.f("ix_documents_checksum"), table_name="documents")
    op.drop_table("documents")
