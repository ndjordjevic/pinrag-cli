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

    @property
    def _names_path(self) -> Path:
        return self.base_dir / "names.json"

    def get_session_names(self) -> dict[str, str]:
        """Return the full ``{session_id: name}`` mapping from the sidecar index."""
        p = self._names_path
        if not p.exists():
            return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def set_session_name(self, session_id: str, name: str) -> None:
        """Assign *name* to *session_id* in the sidecar names index."""
        names = self.get_session_names()
        names[session_id] = name
        self._names_path.write_text(
            json.dumps(names, indent=2) + "\n", encoding="utf-8"
        )

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

    def delete_session(self, session_id: str) -> bool:
        """Delete *session_id*'s JSON file and remove its name entry.

        Returns ``True`` if the file existed and was deleted, ``False`` otherwise.
        """
        path = self._path(session_id)
        deleted = False
        if path.exists():
            path.unlink()
            deleted = True
        names = self.get_session_names()
        if session_id in names:
            del names[session_id]
            self._names_path.write_text(
                json.dumps(names, indent=2) + "\n", encoding="utf-8"
            )
        return deleted

    def delete_all_sessions(self, *, keep_id: str | None = None) -> int:
        """Delete all session files except *keep_id*. Returns count deleted."""
        count = 0
        for p in list(self.base_dir.glob("*.json")):
            if p.name == "names.json":
                continue
            if keep_id and p.stem == keep_id:
                continue
            p.unlink(missing_ok=True)
            count += 1
        names = self.get_session_names()
        pruned = {k: v for k, v in names.items() if k == keep_id}
        self._names_path.write_text(
            json.dumps(pruned, indent=2) + "\n", encoding="utf-8"
        )
        return count

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return sessions sorted newest-first.

        Each entry: ``id``, ``turns``, ``last_query``, ``last_ts``,
        ``collection``, ``name``.
        """
        names = self.get_session_names()
        files = sorted(
            self.base_dir.glob("*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        out: list[dict[str, Any]] = []
        for p in files:
            if p.name == "names.json":
                continue
            try:
                turns = self.get_session(p.stem)
            except (json.JSONDecodeError, OSError):
                continue
            last_q = ""
            last_ts: str | None = None
            collection: str | None = None
            if turns:
                last = turns[-1]
                last_q = str(last.get("query", ""))[:80]
                last_ts = last.get("timestamp") or None
                collection = last.get("collection") or None
            out.append(
                {
                    "id": p.stem,
                    "turns": len(turns),
                    "last_query": last_q,
                    "last_ts": last_ts,
                    "collection": collection,
                    "name": names.get(p.stem),
                }
            )
        return out
