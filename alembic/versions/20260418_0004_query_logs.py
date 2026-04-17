"""Add query_logs table for LLM-judge evaluation and user feedback

Revision ID: 20260418_0004
Revises: 20260417_0003
Create Date: 2026-04-18

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260418_0004"
down_revision = "20260417_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "query_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(length=20), nullable=False, server_default="answer"),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("context_text", sa.Text(), nullable=True),
        sa.Column("model", sa.String(length=64), nullable=True),
        sa.Column("elapsed_ms", sa.Integer(), nullable=True),
        # LLM-judge scores
        sa.Column("score_faithfulness", sa.Float(), nullable=True),
        sa.Column("score_relevance", sa.Float(), nullable=True),
        sa.Column("score_completeness", sa.Float(), nullable=True),
        sa.Column("score_consistency", sa.Float(), nullable=True),
        sa.Column("judge_reasoning", sa.Text(), nullable=True),
        # User feedback: 1=👍, -1=👎, NULL=not rated
        sa.Column("user_feedback", sa.SmallInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_query_logs_created_at", "query_logs", ["created_at"])
    op.create_index("ix_query_logs_mode", "query_logs", ["mode"])


def downgrade() -> None:
    op.drop_index("ix_query_logs_mode", table_name="query_logs")
    op.drop_index("ix_query_logs_created_at", table_name="query_logs")
    op.drop_table("query_logs")
