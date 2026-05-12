"""Add full-text index for hybrid retrieval

Revision ID: 20260512_0006
Revises: 20260511_0005
Create Date: 2026-05-12

"""
from __future__ import annotations

from alembic import op


revision = "20260512_0006"
down_revision = "20260511_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_chunks_fts_simple
        ON chunks
        USING gin (
            to_tsvector(
                'simple',
                concat_ws(
                    ' ',
                    coalesce(section_title, ''),
                    coalesce(summary, ''),
                    coalesce(nosology, ''),
                    coalesce(specialty, ''),
                    coalesce(topic, ''),
                    coalesce(evidence_level, ''),
                    coalesce(chunk_text, '')
                )
            )
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_fts_simple")
