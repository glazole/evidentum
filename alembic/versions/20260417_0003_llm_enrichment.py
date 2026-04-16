"""llm enrichment fields

Revision ID: 20260417_0003
Revises: 20260414_0002
Create Date: 2026-04-17

"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260417_0003"
down_revision = "20260414_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # chunks: LLM-enrichment fields
    op.add_column("chunks", sa.Column("summary", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("nosology", sa.String(length=256), nullable=True))
    op.add_column("chunks", sa.Column("specialty", sa.String(length=128), nullable=True))
    op.add_column("chunks", sa.Column("topic", sa.String(length=128), nullable=True))
    op.add_column("chunks", sa.Column("evidence_level", sa.String(length=32), nullable=True))
    op.create_index("ix_chunks_nosology", "chunks", ["nosology"], unique=False)
    op.create_index("ix_chunks_specialty", "chunks", ["specialty"], unique=False)

    # documents: LLM-enrichment + versioning fields
    op.add_column("documents", sa.Column("specialty", sa.String(length=128), nullable=True))
    op.add_column("documents", sa.Column("nosology_primary", sa.String(length=256), nullable=True))
    op.add_column("documents", sa.Column("summary_ru", sa.Text(), nullable=True))
    op.add_column("documents", sa.Column("previous_document_id", sa.Integer(), nullable=True))
    op.add_column("documents", sa.Column("version_delta_ru", sa.Text(), nullable=True))
    op.create_foreign_key(
        "fk_documents_previous_document_id",
        "documents",
        "documents",
        ["previous_document_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_documents_previous_document_id", "documents", type_="foreignkey")
    op.drop_column("documents", "version_delta_ru")
    op.drop_column("documents", "previous_document_id")
    op.drop_column("documents", "summary_ru")
    op.drop_column("documents", "nosology_primary")
    op.drop_column("documents", "specialty")

    op.drop_index("ix_chunks_specialty", table_name="chunks")
    op.drop_index("ix_chunks_nosology", table_name="chunks")
    op.drop_column("chunks", "evidence_level")
    op.drop_column("chunks", "topic")
    op.drop_column("chunks", "specialty")
    op.drop_column("chunks", "nosology")
    op.drop_column("chunks", "summary")
