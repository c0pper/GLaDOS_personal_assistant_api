import sqlite3
import json
from pathlib import Path


class LLMCache:
    def __init__(self, db_path: str = "db/llm_cache.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            query TEXT PRIMARY KEY,
            response TEXT
        )
        """)
        self.conn.commit()

    def normalize(self, text: str) -> str:
        return text.strip().lower()

    def get(self, query: str):
        key = self.normalize(query)

        cur = self.conn.execute("SELECT response FROM cache WHERE query=?", (key,))

        row = cur.fetchone()

        if row:
            return json.loads(row[0])

        return None

    def set(self, query: str, response: dict):
        key = self.normalize(query)

        cur = self.conn.execute("SELECT response FROM cache WHERE query=?", (key,))

        row = cur.fetchone()

        if row:
            existing = json.loads(row[0])
            existing.update(response)
            final = existing
        else:
            final = response

        self.conn.execute(
            "INSERT OR REPLACE INTO cache(query, response) VALUES (?, ?)",
            (key, json.dumps(final)),
        )

        self.conn.commit()

    def delete(self, query: str):
        key = self.normalize(query)
        with self.conn:
            self.conn.execute("DELETE FROM cache WHERE query=?", (key,))

    def wipe(self):
        with self.conn:
            self.conn.execute("DELETE FROM cache")


Path("db").mkdir(parents=True, exist_ok=True)
cache = LLMCache()


if __name__ == "__main__":
    cache.delete("what are my pending tasks?")