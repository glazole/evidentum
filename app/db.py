from __future__ import annotations

import os
import hashlib
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session, declarative_base, sessionmaker


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/medguides",
)

Base = declarative_base()

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _advisory_lock_key(name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


@contextmanager
def advisory_lock(name: str) -> Generator[bool, None, None]:
    """Session-level PostgreSQL advisory lock for coarse idempotency guards."""
    key = _advisory_lock_key(name)
    conn: Connection = engine.connect()
    locked = False
    try:
        locked = bool(conn.execute(text("select pg_try_advisory_lock(:key)"), {"key": key}).scalar())
        yield locked
    finally:
        if locked:
            conn.execute(text("select pg_advisory_unlock(:key)"), {"key": key})
        conn.close()


def init_db() -> None:
    """
    Initialize the pgvector extension.

    Tables and indexes are managed by Alembic migrations.
    """
    with engine.begin() as conn:
        conn.execute(text("create extension if not exists vector"))
