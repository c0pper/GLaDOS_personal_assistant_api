from typing import Optional
from pathlib import Path
import sqlite3
import threading
import json

from src.config import Config


class LLMCache:
    """
    SQLite cache for LLM outputs.

    Entries are keyed by (namespace, query) so each agent owns its own entries,
    instead of all agents merging fields into a single row per user query.
    """

    def __init__(self, db_path: str = "db/llm_cache.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._create_table()

    def _create_table(self):
        with self.lock, self.conn:
            # The old single-key table cached responses without their tool result, so it held stale answers.
            self.conn.execute("DROP TABLE IF EXISTS cache")
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                namespace TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                PRIMARY KEY (namespace, query)
            )
            """)

    def normalize(self, text: str) -> str:
        return text.strip().lower()

    def get(self, namespace: str, query: str) -> Optional[dict]:
        key = self.normalize(query)

        with self.lock:
            row = self.conn.execute(
                "SELECT response FROM llm_cache WHERE namespace=? AND query=?", (namespace, key)
            ).fetchone()

        if row:
            return json.loads(row[0])

        return None

    def set(self, namespace: str, query: str, response: dict):
        key = self.normalize(query)

        with self.lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO llm_cache(namespace, query, response) VALUES (?, ?, ?)",
                (namespace, key, json.dumps(response)),
            )

    def delete(self, namespace: str, query: str):
        key = self.normalize(query)

        with self.lock, self.conn:
            self.conn.execute("DELETE FROM llm_cache WHERE namespace=? AND query=?", (namespace, key))

    def wipe(self):
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM llm_cache")


cache = LLMCache(Config.LLM_CACHE_DB_PATH)
