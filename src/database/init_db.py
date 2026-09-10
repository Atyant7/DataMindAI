"""
Database initialization routines for DataMind AI.
"""

from __future__ import annotations

from sqlalchemy import inspect, text

from src.core.logger import get_logger
from src.database.models import Base
from src.database.session import engine

logger = get_logger(__name__)


def init_db() -> None:
    """Create all database tables and perform lightweight schema updates."""
    try:
        logger.info("Initializing DataMind AI database tables...")
        Base.metadata.create_all(bind=engine)

        # Ensure column additions for existing SQLite/Postgres tables
        with engine.begin() as conn:
            inspector = inspect(conn)
            if "projects" in inspector.get_table_names():
                columns = [c["name"] for c in inspector.get_columns("projects")]
                if "is_archived" not in columns:
                    logger.info("Adding 'is_archived' column to projects table...")
                    conn.execute(text("ALTER TABLE projects ADD COLUMN is_archived BOOLEAN DEFAULT 0 NOT NULL"))

        logger.info("DataMind AI database tables initialized successfully.")
    except Exception as exc:
        logger.exception("Failed to initialize database: %s", exc)
        raise


if __name__ == "__main__":
    init_db()
