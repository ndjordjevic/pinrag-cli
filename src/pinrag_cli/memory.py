"""Session conversational memory: rolling Q/A window folded into the next query."""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass

_DEFAULT_MAX_TURNS = 5
_DEFAULT_MAX_ANSWER_CHARS = 200


def _truthy_env(name: str, *, default: bool = True) -> bool:
    """Interpret env as enable flag: 0/false/no/off disables (case-insensitive)."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    v = raw.strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return default


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        n = int(str(raw).strip(), 10)
    except ValueError:
        return default
    return max(1, n)


def _summarize_answer(answer: str, max_chars: int) -> str:
    """Truncate answer at a word boundary; add ellipsis when shortened."""
    text = answer.strip()
    if not text:
        return ""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    chunk = text[:max_chars]
    if " " in chunk:
        chunk = chunk.rsplit(" ", 1)[0]
    if not chunk:
        chunk = text[:max_chars]
    return chunk + "…"


@dataclass(frozen=True)
class MemoryTurn:
    """One prior user question and condensed assistant reply."""

    query: str
    answer_summary: str


class ConversationMemory:
    """Rolling Q/A window for in-session follow-ups (bounded length)."""

    def __init__(
        self,
        *,
        max_turns: int = _DEFAULT_MAX_TURNS,
        max_answer_chars: int = _DEFAULT_MAX_ANSWER_CHARS,
        enabled: bool = True,
    ) -> None:
        self._max_turns = max(1, max_turns)
        self._max_answer = max(0, max_answer_chars)
        self._window: deque[MemoryTurn] = deque(maxlen=self._max_turns)
        self.enabled = enabled

    def add_turn(self, query: str, answer: str) -> None:
        if not self.enabled:
            return
        q = query.strip()
        if not q:
            return
        summary = _summarize_answer(answer, self._max_answer)
        self._window.append(MemoryTurn(query=q, answer_summary=summary))

    def build_context_prefix(self) -> str:
        if not self.enabled or not self._window:
            return ""
        lines: list[str] = []
        for t in self._window:
            lines.append(f"Q: {t.query}")
            if t.answer_summary:
                lines.append(f"A: {t.answer_summary}")
            else:
                lines.append("A: ")
        return (
            "Previous conversation for context (answer follow-ups naturally):\n"
            + "\n".join(lines)
            + "\n\nCurrent question: "
        )

    def clear(self) -> None:
        self._window.clear()


def load_conversation_memory_from_env() -> ConversationMemory:
    """Build memory from env: ``PINRAG_CLI_MEMORY``, ``PINRAG_CLI_MEMORY_TURNS``."""
    enabled = _truthy_env("PINRAG_CLI_MEMORY", default=True)
    turns = _positive_int_env("PINRAG_CLI_MEMORY_TURNS", _DEFAULT_MAX_TURNS)
    return ConversationMemory(max_turns=turns, enabled=enabled)
