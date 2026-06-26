from __future__ import annotations
import sqlite3
import hashlib
import json
from pathlib import Path
from typing import Optional
from config.paths import CACHE_DIR

CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = CACHE_DIR / "cache.sqlite"

class CacheManager:
    """Simple sqlite-backed cache for image embeddings and metadata."""

    def __init__(self, db_path: str | None = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self._conn = sqlite3.connect(str(self.db_path))
        self._create_tables()

    def _create_tables(self):
        cur = self._conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            key TEXT PRIMARY KEY,
            type TEXT,
            vector BLOB,
            meta TEXT
        )""")
        self._conn.commit()

    def _hash(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def save_embedding(self, key: str, vector: bytes, meta: dict):
        cur = self._conn.cursor()
        cur.execute("REPLACE INTO embeddings (key,type,vector,meta) VALUES (?,?,?,?)",
                    (key, meta.get("type", "face"), vector, json.dumps(meta)))
        self._conn.commit()

    def load_embedding(self, key: str) -> Optional[dict]:
        cur = self._conn.cursor()
        cur.execute("SELECT vector, meta FROM embeddings WHERE key=", (key,))
        row = cur.fetchone()
        if not row:
            return None
        return {"vector": row[0], "meta": json.loads(row[1])}
