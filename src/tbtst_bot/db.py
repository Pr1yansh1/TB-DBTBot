from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def init_db() -> None:
    """Create our small metadata table if it doesn't exist."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS thread_meta (
                  thread_id TEXT PRIMARY KEY,
                  user_name TEXT,
                  mode TEXT,
                  created_at TIMESTAMP DEFAULT NOW()
                );
                """
            )
        )


def upsert_thread_meta(thread_id: str, *, user_name: str | None = None, mode: str | None = None) -> None:
    """Insert/update metadata for a thread."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO thread_meta (thread_id, user_name, mode)
                VALUES (:thread_id, :user_name, :mode)
                ON CONFLICT (thread_id)
                DO UPDATE SET
                  user_name = COALESCE(EXCLUDED.user_name, thread_meta.user_name),
                  mode = COALESCE(EXCLUDED.mode, thread_meta.mode);
                """
            ),
            {"thread_id": thread_id, "user_name": user_name, "mode": mode},
        )

