"""Rich console output for the PinRAG CLI."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console(stderr=True)


def render_banner(status: dict[str, Any]) -> None:
    """Print startup banner with PinRAG version and store info."""
    lines = [
        f"PinRAG CLI — library {status.get('pinrag_version', '?')}",
        f"Store: {status.get('persist_dir', '?')}  collection: {status.get('collection', '?')}",
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
    table.add_column("document_id", overflow="fold")
    table.add_column("pages")
    for doc_id in sorted(by_doc.keys()):
        chunk_list = by_doc[doc_id]
        pages_cell = _format_source_locations(chunk_list)
        table.add_row(doc_id, pages_cell)
    console.print(table)


def render_documents_table(result: dict[str, Any]) -> None:
    """Render indexed documents from list_documents."""
    docs = result.get("documents") or []
    details = result.get("document_details") or {}
    total_chunks = result.get("total_chunks", 0)
    if not docs:
        console.print(f"No documents (0 chunks in view).")
        return
    table = Table(title=f"Documents ({total_chunks} chunks)", show_header=True)
    table.add_column("document_id", overflow="fold")
    table.add_column("type")
    table.add_column("chunks")
    table.add_column("meta", overflow="fold")
    for doc_id in sorted(docs):
        info = details.get(doc_id) or {}
        dtype = str(info.get("document_type", ""))
        chunks = str(info.get("chunks", ""))
        extras: list[str] = []
        if info.get("pages") is not None:
            extras.append(f"pages={info['pages']}")
        if info.get("messages") is not None:
            extras.append(f"messages={info['messages']}")
        if info.get("segments") is not None:
            extras.append(f"segments={info['segments']}")
        if info.get("file_count") is not None:
            extras.append(f"files={info['file_count']}")
        if info.get("tag"):
            extras.append(f"tag={info['tag']}")
        meta = ", ".join(extras)
        table.add_row(str(doc_id), dtype, chunks, meta)
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
    doc_id = result.get("document_id", "")
    deleted = result.get("deleted_chunks", 0)
    body = f"Removed document [bold]{doc_id}[/] — deleted {deleted} chunk(s)."
    console.print(Panel(body, title="Removed", border_style="green"))


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

- `/add <path> [--tag TAG]` — index a file, directory, or URL (YouTube, GitHub)
- `/list [--tag TAG]` — list indexed documents
- `/remove <document_id>` — remove a document from the index
- `/status` — show version, persist dir, collection, LLM
- `/help` — this help
- `/exit` — quit (same as Ctrl+D)

**Else:** plain text line = RAG query over the index.
"""
    console.print(Markdown(text.strip()))


def render_error(message: str) -> None:
    """Render an error without crashing the REPL."""
    console.print(Panel(f"[red]{message}[/]", title="Error", border_style="red"))
