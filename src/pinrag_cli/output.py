"""Rich console output for the PinRAG CLI."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import threading
import time

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

console = Console(stderr=True)


class StreamingDisplay:
    """Rich.Live panel with animated spinner and elapsed timer."""

    def __init__(self, *, transient: bool = True) -> None:
        self._live: Live | None = None
        self._transient = transient
        self._phase = "working…"
        self._phase_start = time.monotonic()
        self._timer_stop = threading.Event()
        self._timer_thread: threading.Thread | None = None

    def __enter__(self) -> StreamingDisplay:
        self._phase_start = time.monotonic()
        spinner = Spinner("dots", text=Text(self._phase, style="cyan"))
        self._live = Live(
            Panel(spinner),
            console=console,
            refresh_per_second=10,
            transient=self._transient,
        )
        self._live.__enter__()
        self._timer_stop.clear()
        self._timer_thread = threading.Thread(target=self._tick, daemon=True)
        self._timer_thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._timer_stop.set()
        if self._timer_thread is not None:
            self._timer_thread.join(timeout=1)
            self._timer_thread = None
        if self._live is not None:
            self._live.__exit__(*args)
            self._live = None

    def _tick(self) -> None:
        """Background thread: refresh spinner text with elapsed time every 0.5s."""
        while not self._timer_stop.wait(0.5):
            self._refresh()

    def _refresh(self) -> None:
        if self._live is None:
            return
        elapsed = time.monotonic() - self._phase_start
        label = f"{self._phase}  ({elapsed:.1f}s)"
        spinner = Spinner("dots", text=Text(label, style="cyan"))
        self._live.update(Panel(spinner))

    def update(self, message: str) -> None:
        self._phase = message
        self._phase_start = time.monotonic()
        self._refresh()

    def update_progress(
        self, progress: float, total: float | None, message: str | None
    ) -> None:
        if message:
            self.update(message)


def render_banner(status: dict[str, Any]) -> None:
    """Print startup banner with PinRAG version and store info."""
    persist = status.get("persist_dir", "?")
    coll = status.get("collection", "?")
    prov = status.get("llm_provider", "?")
    model = status.get("llm_model", "?")
    lines = [
        f"PinRAG CLI — library {status.get('pinrag_version', '?')}",
        f"Store: {persist}  collection: {coll}",
        f"LLM: {prov}  model: {model}",
        "Type a question to query, or /help for commands. Ctrl+D or /exit to quit.",
    ]
    console.print(Panel("\n".join(lines), title="pinrag-cli", border_style="cyan"))


def _format_source_locations(sources: list[dict[str, Any]]) -> str:
    """Comma-separated page (or page:start) tokens, deduped and sorted."""
    locs: set[tuple[int, int | None]] = set()
    for s in sources:
        page = int(s.get("page", 0))
        start: int | None
        if "start" in s:
            start = int(s["start"])
        else:
            start = None
        locs.add((page, start))

    def sort_key(t: tuple[int, int | None]) -> tuple[int, int]:
        p, st = t
        return (p, st if st is not None else -1)

    parts: list[str] = []
    for page, start in sorted(locs, key=sort_key):
        if start is None:
            parts.append(str(page))
        else:
            parts.append(f"{page}:{start}")
    return ", ".join(parts)


def _source_table_label(sources_for_doc: list[dict[str, Any]]) -> str:
    """First column for Sources table: YouTube title + id when title is present."""
    if not sources_for_doc:
        return ""
    doc_id = str(sources_for_doc[0].get("document_id", ""))
    title = sources_for_doc[0].get("title")
    if title:
        return f"{title} ({doc_id})"
    return doc_id


def render_query_result(result: dict[str, Any]) -> None:
    """Render RAG answer and source table."""
    answer = result.get("answer", "")
    if answer:
        console.print(Markdown(str(answer)))
    sources = result.get("sources") or []
    if not sources:
        return
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in sources:
        doc_id = str(s.get("document_id", ""))
        by_doc[doc_id].append(s)

    table = Table(title="Sources", show_header=True)
    table.add_column("source", overflow="fold")
    table.add_column("pages")
    for doc_id in sorted(by_doc.keys()):
        chunk_list = by_doc[doc_id]
        pages_cell = _format_source_locations(chunk_list)
        table.add_row(_source_table_label(chunk_list), pages_cell)
    console.print(table)


_KNOWN_DOCUMENT_DETAIL_KEYS = frozenset(
    {
        "document_type",
        "chunks",
        "pages",
        "messages",
        "segments",
        "title",
        "ref",
        "tag",
        "upload_timestamp",
        "bytes",
    }
)


def _format_bytes_cell(n: Any) -> str:
    """Human-readable size for list_documents ``bytes`` field."""
    if n is None:
        return ""
    try:
        num = int(n)
    except (TypeError, ValueError):
        return str(n)
    if num < 0:
        return str(num)
    b = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(b)} B"
            return f"{b:.1f} {unit}"
        b /= 1024.0
    return f"{b:.1f} TB"


def _format_uploaded_cell(raw: Any) -> str:
    """Compact display for ISO ``upload_timestamp`` values (date + minute, UTC)."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    norm = s.replace("Z", "+00:00") if s.endswith("Z") else s
    try:
        dt = datetime.fromisoformat(norm)
    except ValueError:
        return s[:19] + ("…" if len(s) > 19 else "")
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M")


def _document_extent_and_extra(info: dict[str, Any]) -> str:
    """Pages / messages / segments plus any future document_details keys."""
    parts: list[str] = []
    if info.get("pages") is not None:
        parts.append(f"pages={info['pages']}")
    if info.get("messages") is not None:
        parts.append(f"messages={info['messages']}")
    if info.get("segments") is not None:
        parts.append(f"segments={info['segments']}")
    for k in sorted(info.keys()):
        if k in _KNOWN_DOCUMENT_DETAIL_KEYS:
            continue
        v = info[k]
        if v is None:
            continue
        s = str(v).strip()
        if s:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def render_documents_table(result: dict[str, Any]) -> None:
    """Render indexed documents from list_documents (all document_details fields)."""
    docs = result.get("documents") or []
    details = result.get("document_details") or {}
    total_chunks = result.get("total_chunks", 0)
    if not docs:
        console.print("No documents (0 chunks in view).")
        return
    table = Table(
        title=f"Documents ({total_chunks} chunks)",
        show_header=True,
        expand=True,
    )
    # expand=True + ratio: use terminal width; no_wrap avoids mid-cell line breaks.
    table.add_column(
        "title",
        overflow="ellipsis",
        no_wrap=True,
        min_width=52,
        ratio=4,
    )
    table.add_column(
        "ref",
        overflow="ellipsis",
        no_wrap=True,
        min_width=28,
        ratio=2,
    )
    table.add_column("type", overflow="ellipsis", no_wrap=True, max_width=10)
    table.add_column("chunks", overflow="ellipsis", no_wrap=True, max_width=8)
    table.add_column("tag", overflow="ellipsis", no_wrap=True, max_width=14)
    table.add_column(
        "uploaded",
        overflow="ellipsis",
        no_wrap=True,
        min_width=16,
        max_width=16,
    )
    table.add_column("bytes", overflow="ellipsis", no_wrap=True, max_width=12)
    table.add_column(
        "extent",
        overflow="ellipsis",
        no_wrap=True,
        min_width=22,
        ratio=1,
    )
    for doc_id in sorted(docs):
        info = details.get(doc_id) or {}
        dtype = str(info.get("document_type", ""))
        chunks = str(info.get("chunks", ""))
        title = str(info.get("title", "") or "")
        ref = str(info.get("ref") or doc_id)
        tag = str(info.get("tag", "") or "")
        uploaded = _format_uploaded_cell(info.get("upload_timestamp"))
        bytes_cell = _format_bytes_cell(info.get("bytes"))
        extent = _document_extent_and_extra(info)
        table.add_row(
            title,
            ref,
            dtype,
            chunks,
            tag,
            uploaded,
            bytes_cell,
            extent,
        )
    console.print(table)


def render_add_result(result: dict[str, Any]) -> None:
    """Summarize add_files batch result."""
    indexed = result.get("indexed") or []
    failed = result.get("failed") or []
    persist = result.get("persist_directory", "")
    coll = result.get("collection_name", "")
    console.print(
        f"Indexed: {len(indexed)}  Failed: {len(failed)}  "
        f"persist=[cyan]{persist}[/] collection=[cyan]{coll}[/]",
        highlight=False,
    )
    if indexed:
        t = Table(title="Indexed", show_header=True)
        t.add_column("path/label", overflow="fold")
        t.add_column("format")
        t.add_column("details", overflow="fold")
        for item in indexed:
            fmt = str(item.get("format", ""))
            path = str(item.get("path", item.get("repo", "")))
            detail_parts = []
            if item.get("total_chunks") is not None:
                detail_parts.append(f"chunks={item['total_chunks']}")
            if item.get("title"):
                detail_parts.append(str(item["title"]))
            t.add_row(path, fmt, ", ".join(detail_parts))
        console.print(t)
    if failed:
        t = Table(title="Failed", show_header=True)
        t.add_column("path", overflow="fold")
        t.add_column("error", overflow="fold")
        for item in failed:
            t.add_row(str(item.get("path", "")), str(item.get("error", "")))
        console.print(t)


def render_remove_result(result: dict[str, Any]) -> None:
    """Show remove_document outcome."""
    doc_id = str(result.get("document_id", "") or "")
    try:
        deleted = int(result.get("deleted_chunks", 0) or 0)
    except (TypeError, ValueError):
        deleted = 0
    if deleted == 0:
        body = (
            f"No chunks matched [bold]{doc_id}[/] (ref, title, or PDF stem). "
            "Check spelling or run /list and use the exact [bold]ref[/] or [bold]title[/]. "
            "Quoted args: /remove \"full name.pdf\"."
        )
        console.print(
            Panel(
                body,
                title="Remove — nothing deleted",
                border_style="yellow",
            )
        )
        return
    body = f"Removed document [bold]{doc_id}[/] — deleted {deleted} chunk(s)."
    console.print(Panel(body, title="Removed", border_style="green"))


def render_set_tag_result(result: dict[str, Any]) -> None:
    """Show set_document_tag outcome."""
    doc_id = str(result.get("document_id", "") or "")
    tag = str(result.get("tag", "") or "")
    try:
        updated = int(result.get("updated_chunks", 0) or 0)
    except (TypeError, ValueError):
        updated = 0
    try:
        parents = int(result.get("parents_updated", 0) or 0)
    except (TypeError, ValueError):
        parents = 0
    if updated == 0:
        body = (
            f"No chunks updated for [bold]{doc_id}[/]. "
            "Use /list for **ref** or **title**; quoted: `/tag \"Book title\" --tag AMIGA`."
        )
        console.print(
            Panel(
                body,
                title="Tag — no chunks updated",
                border_style="yellow",
            )
        )
        return
    extra = f" Parent docstore entries updated: {parents}." if parents else ""
    body = (
        f"Document [bold]{doc_id}[/] — tag set to [bold]{tag}[/] "
        f"({updated} chunk(s)).{extra}"
    )
    console.print(Panel(body, title="Tag updated", border_style="green"))


def render_status(status: dict[str, Any]) -> None:
    """Show backend configuration summary."""
    lines = [
        f"pinrag_version: {status.get('pinrag_version')}",
        f"persist_dir: {status.get('persist_dir')}",
        f"collection: {status.get('collection')}",
        f"llm_provider: {status.get('llm_provider')}",
        f"llm_model: {status.get('llm_model')}",
    ]
    console.print(Panel("\n".join(lines), title="Status", border_style="blue"))


def render_help() -> None:
    """Print slash command reference."""
    text = """
**Commands**

- `/add <path> [--tag TAG]` — index a file, directory, or URL; spaces in paths work unquoted or in ``'quotes'`` / ``"double quotes"``
- `/list [--tag TAG]` — list indexed documents (all PinRAG document_details fields)
- `/remove <ref|title>` — remove by **ref**, exact list **title**, or PDF stem (must match one doc)
- `/tag <ref|title> --tag TAG` — set or replace tag on all chunks for one document
- `/ask <ref|title> -- <question>` — RAG query scoped to one document (same ref/title/stem rules as `/remove`)
- `/switch` — list Chroma collections; `/switch NAME` — use that collection
- `/history` — show conversation turns for this session (JSON-backed)
- `/status` — show version, persist dir, collection, LLM
- `/help` — this help
- `/exit` — quit (same as Ctrl+D)

**Else:** plain text line = RAG query over the index.
"""
    console.print(Markdown(text.strip()))


def render_collection_names(
    names: list[str],
    *,
    title: str = "Collections",
    empty_persist_dir: str | None = None,
) -> None:
    """Print collection names as a simple table."""
    if not names:
        console.print("No collections found.")
        if empty_persist_dir:
            console.print(
                f"[dim]Effective store path: {empty_persist_dir}. "
                "Chroma lists only collections that exist—run /add to create one, "
                "or start pinrag server with PINRAG_PERSIST_DIR if your data lives elsewhere.[/]"
            )
        return
    table = Table(title=title, show_header=True)
    table.add_column("name", overflow="fold")
    for n in names:
        table.add_row(n)
    console.print(table)


def render_history_turns(turns: list[dict[str, Any]], *, limit: int = 20) -> None:
    """Print recent conversation turns (condensed)."""
    if not turns:
        console.print("No turns in this session yet.")
        return
    shown = turns[-limit:]
    table = Table(
        title=f"Session (last {len(shown)} of {len(turns)})", show_header=True
    )
    table.add_column("time", overflow="fold", max_width=22)
    table.add_column("query", overflow="fold", max_width=40)
    table.add_column("answer_preview", overflow="fold", max_width=45)
    for t in shown:
        ts = str(t.get("timestamp", ""))[:19]
        q = str(t.get("query", ""))[:200]
        a = str(t.get("answer", ""))[:120]
        if len(str(t.get("answer", ""))) > 120:
            a += "…"
        table.add_row(ts, q, a)
    console.print(table)


def render_error(message: str) -> None:
    """Render an error without crashing the REPL."""
    console.print(Panel(f"[red]{message}[/]", title="Error", border_style="red"))
