"""
Database connection and session factory for DataMind AI.

Supports SQLite (development) and PostgreSQL (production).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import DATABASE_URL, ensure_project_directories

# Ensure runtime directories (including artifacts) exist
ensure_project_directories()

# For SQLite, check_same_thread=False allows Streamlit's multi-threaded model
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context manager for obtaining a database session."""
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
