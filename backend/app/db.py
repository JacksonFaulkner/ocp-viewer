import sqlite3
from contextlib import contextmanager

from app.config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    duration TEXT,
    view_count INTEGER,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | downloading | done | failed
    s3_key TEXT,
    size_bytes INTEGER,
    error TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


@contextmanager
def conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute(_SCHEMA)
    try:
        yield c
        c.commit()
    finally:
        c.close()


def upsert_pending(c: sqlite3.Connection, videos) -> None:
    c.executemany(
        "INSERT OR IGNORE INTO videos (id, url, title, duration, view_count) VALUES (?,?,?,?,?)",
        [(v.id, v.url, v.title, v.duration, v.view_count) for v in videos],
    )


def set_status(c, vid: str, status: str, **fields) -> None:
    cols = "".join(f", {k} = ?" for k in fields)
    c.execute(
        f"UPDATE videos SET status = ?, updated_at = CURRENT_TIMESTAMP{cols} WHERE id = ?",
        (status, *fields.values(), vid),
    )
    c.commit()
