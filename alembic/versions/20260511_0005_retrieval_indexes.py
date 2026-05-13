"""Add retrieval indexes

Revision ID: 20260511_0005
Revises: 20260418_0004
Create Date: 2026-05-11

"""
from __future__ import annotations

from alembic import op


revision = "20260511_0005"
down_revision = "20260418_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw_cosine
        ON chunks
        USING hnsw (embedding vector_cosine_ops)
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_documents_region ON documents (region)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_documents_year ON documents (year)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_documents_year")
    op.execute("DROP INDEX IF EXISTS ix_documents_region")
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw_cosine")
