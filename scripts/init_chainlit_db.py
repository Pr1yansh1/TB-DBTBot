"""
One-time schema initializer for the Chainlit SQLAlchemyDataLayer tables.

Run once before deploying or after pointing at a new database:

    uv run python scripts/init_chainlit_db.py

The script reads the same connection logic as chainlit_app.py so it targets
whichever DB is configured via TBTST_CHAINLIT_DB, DATABASE_URL, or the
default SQLite fallback.

Tables created (if they don't exist):
  users, threads, steps, elements, feedbacks
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Allow importing from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from chainlit_app import _build_async_conninfo

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# ---------------------------------------------------------------------------
# Schema — derived from SQLAlchemyDataLayer SQL queries.
# All columns are TEXT/INTEGER; Chainlit serialises JSON fields to strings.
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    "id"          TEXT PRIMARY KEY,
    "identifier"  TEXT NOT NULL UNIQUE,
    "createdAt"   TEXT,
    "metadata"    TEXT
);

CREATE TABLE IF NOT EXISTS threads (
    "id"             TEXT PRIMARY KEY,
    "createdAt"      TEXT,
    "name"           TEXT,
    "userId"         TEXT,
    "userIdentifier" TEXT,
    "tags"           TEXT,
    "metadata"       TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    "id"            TEXT PRIMARY KEY,
    "name"          TEXT,
    "type"          TEXT,
    "threadId"      TEXT,
    "parentId"      TEXT,
    "streaming"     INTEGER,
    "waitForAnswer" INTEGER,
    "isError"       INTEGER,
    "metadata"      TEXT,
    "tags"          TEXT,
    "input"         TEXT,
    "output"        TEXT,
    "createdAt"     TEXT,
    "start"         TEXT,
    "end"           TEXT,
    "generation"    TEXT,
    "showInput"     TEXT,
    "defaultOpen"   INTEGER,
    "language"      TEXT,
    "command"       TEXT
);

CREATE TABLE IF NOT EXISTS elements (
    "id"           TEXT PRIMARY KEY,
    "threadId"     TEXT,
    "type"         TEXT,
    "chainlitKey"  TEXT,
    "url"          TEXT,
    "objectKey"    TEXT,
    "name"         TEXT,
    "props"        TEXT,
    "display"      TEXT,
    "size"         TEXT,
    "language"     TEXT,
    "page"         INTEGER,
    "autoPlay"     INTEGER,
    "playerConfig" TEXT,
    "forId"        TEXT,
    "mime"         TEXT
);

CREATE TABLE IF NOT EXISTS feedbacks (
    "id"       TEXT PRIMARY KEY,
    "forId"    TEXT,
    "threadId" TEXT,
    "value"    INTEGER,
    "comment"  TEXT
);
"""


async def main() -> None:
    conninfo = _build_async_conninfo()
    print(f"Initializing Chainlit schema on: {conninfo!r}")

    engine = create_async_engine(conninfo)
    async with engine.begin() as conn:
        # SQLite doesn't support multi-statement execute; split on ";"
        for stmt in SCHEMA_SQL.split(";"):
            stmt = stmt.strip()
            if stmt:
                await conn.execute(text(stmt))
    await engine.dispose()
    print("Done — Chainlit tables ready.")


if __name__ == "__main__":
    asyncio.run(main())
