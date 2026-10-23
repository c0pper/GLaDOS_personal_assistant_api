import time

from src.config import Config
from src.services.conversation_store import conversations


def test_history_is_oldest_first_and_per_conversation():
    conversations.add_turn("a", "Turn on the lights", "Done.")
    conversations.add_turn("b", "Hello", "Oh. It's you.")
    conversations.add_turn("a", "Turn them off", "Off.")

    assert conversations.history_text("a") == (
        "User: Turn on the lights\nGLaDOS: Done.\nUser: Turn them off\nGLaDOS: Off."
    )
    assert conversations.history_text("missing") is None


def test_only_the_last_turns_are_kept(monkeypatch):
    monkeypatch.setattr(Config, "CONVERSATION_MAX_TURNS", 2)
    for i in range(4):
        conversations.add_turn("a", f"message {i}", f"reply {i}")

    assert conversations.recent_turns("a") == [("message 2", "reply 2"), ("message 3", "reply 3")]


def test_old_turns_are_ignored(monkeypatch):
    conversations.add_turn("a", "yesterday", "old")
    monkeypatch.setattr(time, "time", lambda: Config.CONVERSATION_MAX_AGE_SECONDS * 2 + 1e10)

    assert conversations.history_text("a") is None


def test_clear_forgets_one_conversation():
    conversations.add_turn("a", "one", "1")
    conversations.add_turn("b", "two", "2")

    conversations.clear("a")

    assert conversations.history_text("a") is None
    assert conversations.history_text("b") == "User: two\nGLaDOS: 2"
