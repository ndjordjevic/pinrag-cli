"""Persistent conversation history (JSON files)."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class TurnRecord:
    """One user query and model outcome (serializable)."""

    query: str
    answer: str
    sources: list[dict[str, Any]]
    timestamp: str
    collection: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


class ConversationStore:
    """Append-only session files under ``~/.pinrag-cli/history/``."""

    def __init__(self, base_dir: Path | None = None) -> None:
        default_hist = Path.home() / ".pinrag-cli" / "history"
        self.base_dir = base_dir if base_dir is not None else default_hist
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def new_session(self) -> str:
        sid = str(uuid.uuid4())
        path = self.base_dir / f"{sid}.json"
        path.write_text("[]\n", encoding="utf-8")
        return sid

    def _path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.json"

    def add_turn(
        self,
        session_id: str,
        query: str,
        result: dict[str, Any],
        *,
        collection: str | None = None,
    ) -> None:
        path = self._path(session_id)
        turns = self.get_session(session_id)
        rec = TurnRecord(
            query=query,
            answer=str(result.get("answer", "")),
            sources=list(result.get("sources") or []),
            timestamp=_utc_now_iso(),
            collection=collection,
        )
        turns.append(rec.to_json_dict())
        path.write_text(json.dumps(turns, indent=2) + "\n", encoding="utf-8")

    def get_session(self, session_id: str) -> list[dict[str, Any]]:
        path = self._path(session_id)
        if not path.exists():
            return []
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw or "[]")
        return data if isinstance(data, list) else []

    def list_sessions(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for p in sorted(self.base_dir.glob("*.json"), key=lambda x: x.stat().st_mtime):
            try:
                turns = self.get_session(p.stem)
            except (json.JSONDecodeError, OSError):
                continue
            last_q = ""
            if turns:
                last_q = str(turns[-1].get("query", ""))[:80]
            out.append(
                {
                    "id": p.stem,
                    "turns": len(turns),
                    "last_query": last_q,
                }
            )
        return out
