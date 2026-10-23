from pathlib import Path
from typing import List, Optional, Tuple
import sqlite3
import threading
import time

from src.config import Config


class ConversationStore:
    """
    SQLite store of past exchanges (user message + GLaDOS reply), grouped by conversation id
    (e.g. one per Telegram chat), so follow-up messages can be understood in context.
    """

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._create_table()

    def _create_table(self):
        with self.lock, self.conn:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                reply TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """)
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS conversation_turns_by_id ON conversation_turns(conversation_id, id)"
            )

    def recent_turns(self, conversation_id: str) -> List[Tuple[str, str]]:
        """The last exchanges (oldest first) that are recent enough to still be relevant."""
        oldest = time.time() - Config.CONVERSATION_MAX_AGE_SECONDS

        with self.lock:
            rows = self.conn.execute(
                "SELECT user_message, reply FROM conversation_turns "
                "WHERE conversation_id=? AND created_at>=? ORDER BY id DESC LIMIT ?",
                (conversation_id, oldest, Config.CONVERSATION_MAX_TURNS),
            ).fetchall()

        return list(reversed(rows))

    def history_text(self, conversation_id: str) -> Optional[str]:
        """Recent exchanges formatted for a prompt, or None if there are none."""
        turns = self.recent_turns(conversation_id)
        if not turns:
            return None
        return "\n".join(f"User: {user_message}\nGLaDOS: {reply}" for user_message, reply in turns)

    def add_turn(self, conversation_id: str, user_message: str, reply: str):
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO conversation_turns(conversation_id, user_message, reply, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, user_message, reply, time.time()),
            )

    def clear(self, conversation_id: str):
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM conversation_turns WHERE conversation_id=?", (conversation_id,))

    def wipe(self):
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM conversation_turns")


conversations = ConversationStore(Config.CONVERSATION_DB_PATH)
