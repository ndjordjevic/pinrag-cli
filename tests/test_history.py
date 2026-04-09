"""Tests for ConversationStore."""

from __future__ import annotations

from pathlib import Path

from pinrag_cli.history import ConversationStore


def test_conversation_store_roundtrip(tmp_path: Path) -> None:
    store = ConversationStore(base_dir=tmp_path)
    sid = store.new_session()
    store.add_turn(
        sid,
        "hello?",
        {"answer": "hi", "sources": []},
        collection="pinrag",
    )
    turns = store.get_session(sid)
    assert len(turns) == 1
    assert turns[0]["query"] == "hello?"
    assert turns[0]["answer"] == "hi"
    assert turns[0]["collection"] == "pinrag"
