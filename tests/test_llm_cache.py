import sqlite3

import pytest

from src.services.llm_cache.llm_cache import LLMCache


@pytest.fixture
def llm_cache(tmp_path) -> LLMCache:
    return LLMCache(str(tmp_path / "cache.db"))


def test_get_missing_returns_none(llm_cache):
    assert llm_cache.get("ns", "unknown") is None


def test_set_then_get_round_trips(llm_cache):
    llm_cache.set("ns", "hello", {"answer": 42})

    assert llm_cache.get("ns", "hello") == {"answer": 42}


def test_keys_are_normalized(llm_cache):
    llm_cache.set("ns", "  Turn On The Lights ", {"x": 1})

    assert llm_cache.get("ns", "turn on the lights") == {"x": 1}


def test_namespaces_are_isolated(llm_cache):
    llm_cache.set("orchestrator", "query", {"tool_name": "No Tool"})
    llm_cache.set("responder", "query", {"final_response": "Hi."})

    assert llm_cache.get("orchestrator", "query") == {"tool_name": "No Tool"}
    assert llm_cache.get("responder", "query") == {"final_response": "Hi."}


def test_set_replaces_instead_of_merging(llm_cache):
    llm_cache.set("ns", "q", {"a": 1, "b": 2})
    llm_cache.set("ns", "q", {"a": 3})

    assert llm_cache.get("ns", "q") == {"a": 3}


def test_delete_removes_only_that_entry(llm_cache):
    llm_cache.set("ns", "q1", {"a": 1})
    llm_cache.set("ns", "q2", {"a": 2})

    llm_cache.delete("ns", "q1")

    assert llm_cache.get("ns", "q1") is None
    assert llm_cache.get("ns", "q2") == {"a": 2}


def test_wipe_removes_everything(llm_cache):
    llm_cache.set("ns1", "q", {"a": 1})
    llm_cache.set("ns2", "q", {"a": 2})

    llm_cache.wipe()

    assert llm_cache.get("ns1", "q") is None
    assert llm_cache.get("ns2", "q") is None


def test_persists_across_instances(tmp_path):
    path = str(tmp_path / "cache.db")
    LLMCache(path).set("ns", "q", {"a": 1})

    assert LLMCache(path).get("ns", "q") == {"a": 1}


def test_legacy_table_is_dropped(tmp_path):
    path = tmp_path / "cache.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE cache (query TEXT PRIMARY KEY, response TEXT)")
    conn.commit()
    conn.close()

    LLMCache(str(path))

    conn = sqlite3.connect(path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert tables == {"llm_cache"}


def test_creates_missing_parent_directory(tmp_path):
    path = tmp_path / "nested" / "dir" / "cache.db"

    LLMCache(str(path)).set("ns", "q", {"a": 1})

    assert path.exists()
