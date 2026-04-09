"""Tests for session conversational memory."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pinrag_cli.memory import (
    ConversationMemory,
    MemoryTurn,
    _positive_int_env,
    _summarize_answer,
    _truthy_env,
    load_conversation_memory_from_env,
)


def test_summarize_answer_short_unchanged() -> None:
    assert _summarize_answer("hello world", 200) == "hello world"


def test_summarize_answer_truncates_at_word_boundary() -> None:
    long = "one " * 80  # > 200 chars
    out = _summarize_answer(long, 30)
    assert out.endswith("…")
    assert len(out) <= 31
    assert "one" in out


def test_summarize_answer_empty() -> None:
    assert _summarize_answer("", 50) == ""
    assert _summarize_answer("   ", 50) == ""


def test_conversation_memory_build_context_empty() -> None:
    m = ConversationMemory()
    assert m.build_context_prefix() == ""


def test_conversation_memory_add_and_prefix() -> None:
    m = ConversationMemory(max_turns=3, max_answer_chars=100)
    m.add_turn("What is X?", "X is a letter.")
    p = m.build_context_prefix()
    assert "What is X?" in p
    assert "X is a letter." in p
    assert "Current question:" in p


def test_conversation_memory_skips_empty_query() -> None:
    m = ConversationMemory()
    m.add_turn("   ", "nope")
    m.add_turn("", "nope")
    assert m.build_context_prefix() == ""


def test_conversation_memory_disabled_no_add_no_prefix() -> None:
    m = ConversationMemory(enabled=False)
    m.add_turn("q", "a")
    assert m.build_context_prefix() == ""


def test_conversation_memory_rolling_eviction() -> None:
    m = ConversationMemory(max_turns=2, max_answer_chars=50)
    m.add_turn("q1", "a1")
    m.add_turn("q2", "a2")
    m.add_turn("q3", "a3")
    p = m.build_context_prefix()
    assert "q1" not in p
    assert "q2" in p
    assert "q3" in p


def test_conversation_memory_clear() -> None:
    m = ConversationMemory()
    m.add_turn("q", "a")
    m.clear()
    assert m.build_context_prefix() == ""


def test_truthy_env_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("T_DEF", raising=False)
    assert _truthy_env("T_DEF", default=True) is True
    assert _truthy_env("T_DEF", default=False) is False


def test_truthy_env_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("T_MEM", "0")
    assert _truthy_env("T_MEM", default=True) is False
    monkeypatch.setenv("T_MEM", "false")
    assert _truthy_env("T_MEM", default=True) is False
    monkeypatch.setenv("T_MEM", "1")
    assert _truthy_env("T_MEM", default=False) is True


def test_positive_int_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("T_PI", raising=False)
    assert _positive_int_env("T_PI", 5) == 5
    monkeypatch.setenv("T_PI", "notint")
    assert _positive_int_env("T_PI", 5) == 5
    monkeypatch.setenv("T_PI", "3")
    assert _positive_int_env("T_PI", 5) == 3
    monkeypatch.setenv("T_PI", "-10")
    assert _positive_int_env("T_PI", 5) == 1


def test_load_conversation_memory_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PINRAG_CLI_MEMORY", raising=False)
    monkeypatch.delenv("PINRAG_CLI_MEMORY_TURNS", raising=False)
    m = load_conversation_memory_from_env()
    assert m.enabled is True
    assert m._max_turns == 5  # noqa: SLF001

    monkeypatch.setenv("PINRAG_CLI_MEMORY", "0")
    monkeypatch.setenv("PINRAG_CLI_MEMORY_TURNS", "2")
    m2 = load_conversation_memory_from_env()
    assert m2.enabled is False
    assert m2._max_turns == 2  # noqa: SLF001


def test_memory_turn_frozen() -> None:
    t = MemoryTurn(query="q", answer_summary="a")
    with pytest.raises(FrozenInstanceError):
        t.query = "x"  # type: ignore[misc]
