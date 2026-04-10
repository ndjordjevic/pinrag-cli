"""Slash-command dispatcher for the PinRAG CLI REPL."""

from __future__ import annotations

import asyncio
import shlex
from typing import TYPE_CHECKING

from pinrag_cli import output
from pinrag_cli.config import (
    USER_CONFIG_PATH,
    effective_config_rows,
    parse_set_args,
    set_user_config_key,
)

if TYPE_CHECKING:
    from pinrag_cli.repl import REPLApp


def _split_tag_args(args_str: str) -> tuple[str | None, str]:
    """Extract optional ``--tag TAG``; return (tag, remainder).

    Uses shell-style tokenization so paths may be quoted (``'...'`` / ``"..."``)
    or unquoted; spaces inside quotes are preserved. Multi-word tags work when
    quoted (e.g. ``--tag "my tag"``).
    """
    args_str = args_str.strip()
    if not args_str:
        return None, ""
    try:
        parts = shlex.split(args_str, posix=True)
    except ValueError:
        parts = args_str.split()
    tag: str | None = None
    rest: list[str] = []
    i = 0
    while i < len(parts):
        if parts[i] == "--tag" and i + 1 < len(parts):
            tag = parts[i + 1]
            i += 2
            continue
        rest.append(parts[i])
        i += 1
    return tag, " ".join(rest).strip()


def _split_ask_args(args_str: str) -> tuple[str | None, str | None]:
    """Parse ``/ask SELECTOR -- QUESTION`` via shell-style tokens.

    Returns ``(selector, question)`` or ``(None, None)`` if ``--`` is missing
    or either side is empty.
    """
    raw = args_str.strip()
    if not raw:
        return None, None
    parts: list[str]
    try:
        parts = shlex.split(raw, posix=True)
    except ValueError:
        sep_idx = raw.find(" -- ")
        if sep_idx == -1:
            return None, None
        sel = raw[:sep_idx].strip()
        q = raw[sep_idx + 4 :].strip()
        return (sel or None, q or None)
    try:
        dash_at = parts.index("--")
    except ValueError:
        return None, None
    sel_tokens = parts[:dash_at]
    q_tokens = parts[dash_at + 1 :]
    if not sel_tokens or not q_tokens:
        return None, None
    return " ".join(sel_tokens).strip(), " ".join(q_tokens).strip()


class CommandDispatcher:
    """Maps ``/name`` to handler methods."""

    def __init__(self, repl: REPLApp) -> None:
        self.repl = repl
        self.should_exit = False

    async def dispatch(self, text: str) -> None:
        raw = text.strip()
        if not raw.startswith("/"):
            output.render_error("Internal: dispatch called without slash command.")
            return
        parts = raw.split(maxsplit=1)
        name = parts[0][1:].lower()
        args_str = parts[1] if len(parts) > 1 else ""
        handler = getattr(self, f"cmd_{name}", None)
        if handler is None:
            output.render_error(
                f"Unknown command: /{name}. Type /help for available commands."
            )
            return
        try:
            await handler(args_str)
        except Exception as e:
            output.render_error(str(e))

    async def cmd_add(self, args_str: str) -> None:
        tag, path = _split_tag_args(args_str)
        if not path:
            output.render_error("Usage: /add <path> [--tag TAG]")
            return
        tags = [tag] if tag else None
        with output.StreamingDisplay() as stream:

            async def prog(p: float, tot: float | None, msg: str | None) -> None:
                stream.update_progress(p, tot, msg)

            def verb(message: str, _level: str) -> None:
                stream.update(message)

            if self.repl.mcp is not None:
                result = await self.repl.mcp.add(
                    [path], tags=tags, progress_callback=prog
                )
            else:
                result = await asyncio.to_thread(
                    self.repl.direct.add,
                    [path],
                    tags=tags,
                    verbose_emitter=verb,
                )
        output.render_add_result(result)

    async def cmd_list(self, args_str: str) -> None:
        tag, _ = _split_tag_args(args_str)
        with output.StreamingDisplay() as stream:

            async def prog(p: float, tot: float | None, msg: str | None) -> None:
                stream.update_progress(p, tot, msg)

            def verb(message: str, _level: str) -> None:
                stream.update(message)

            if self.repl.mcp is not None:
                result = await self.repl.mcp.list_documents(
                    tag=tag if tag else None,
                    progress_callback=prog,
                )
            else:
                result = await asyncio.to_thread(
                    self.repl.direct.list_documents,
                    tag=tag if tag else None,
                    verbose_emitter=verb,
                )
        output.render_documents_table(result)

    async def cmd_remove(self, args_str: str) -> None:
        args_str = args_str.strip()
        if not args_str:
            output.render_error("Usage: /remove <ref>")
            return
        try:
            parts = shlex.split(args_str, posix=True)
        except ValueError:
            parts = args_str.split()
        doc_id = " ".join(parts).strip() if parts else ""
        if not doc_id:
            output.render_error("Usage: /remove <ref>")
            return
        with output.StreamingDisplay() as stream:

            async def prog(p: float, tot: float | None, msg: str | None) -> None:
                stream.update_progress(p, tot, msg)

            def verb(message: str, _level: str) -> None:
                stream.update(message)

            if self.repl.mcp is not None:
                result = await self.repl.mcp.remove(
                    doc_id, progress_callback=prog
                )
            else:
                result = await asyncio.to_thread(
                    self.repl.direct.remove,
                    doc_id,
                    verbose_emitter=verb,
                )
        output.render_remove_result(result)

    async def cmd_tag(self, args_str: str) -> None:
        tag, doc_selector = _split_tag_args(args_str)
        if not tag or not doc_selector:
            output.render_error("Usage: /tag <ref|title> --tag TAG")
            return
        with output.StreamingDisplay() as stream:

            async def prog(p: float, tot: float | None, msg: str | None) -> None:
                stream.update_progress(p, tot, msg)

            def verb(message: str, _level: str) -> None:
                stream.update(message)

            if self.repl.mcp is not None:
                result = await self.repl.mcp.set_document_tag(
                    doc_selector,
                    tag,
                    progress_callback=prog,
                )
            else:
                result = await asyncio.to_thread(
                    self.repl.direct.set_document_tag,
                    doc_selector,
                    tag,
                    verbose_emitter=verb,
                )
        output.render_set_tag_result(result)

    async def cmd_ask(self, args_str: str) -> None:
        doc_sel, question = _split_ask_args(args_str)
        if not doc_sel or not question:
            output.render_error(
                "Usage: /ask <ref|title> -- <question> "
                "(use -- to separate document and question; "
                "quote paths/titles as needed)"
            )
            return
        hist_line = f"/ask {doc_sel} -- {question}"
        prefix = self.repl.memory.build_context_prefix()
        augmented = prefix + question if prefix else question
        try:
            with output.StreamingDisplay() as stream:

                async def prog(p: float, tot: float | None, msg: str | None) -> None:
                    stream.update_progress(p, tot, msg)

                def verb(message: str, _level: str) -> None:
                    stream.update(message)

                rs = self.repl._response_style_literal()
                if self.repl.mcp is not None:
                    result = await self.repl.mcp.query(
                        augmented,
                        document_id=doc_sel,
                        progress_callback=prog,
                        response_style=rs,
                    )
                else:
                    assert self.repl.direct is not None
                    result = await asyncio.to_thread(
                        self.repl.direct.query,
                        augmented,
                        document_id=doc_sel,
                        verbose_emitter=verb,
                        response_style=rs,
                    )
            output.render_query_result(result)
            # Memory: natural-language only; JSON history keeps full /ask line.
            self.repl.memory.add_turn(question, str(result.get("answer", "")))
            self.repl.history.add_turn(
                self.repl.session_id,
                hist_line,
                result,
                collection=self.repl._collection_for_history(),
            )
        except Exception as e:
            output.render_error(str(e))

    async def cmd_focus(self, args_str: str) -> None:
        doc_sel = args_str.strip()
        if not doc_sel:
            self.repl.focused_doc = None
            output.console.print("[green]Focus cleared.[/] Queries run across all documents.")
            return
        self.repl.focused_doc = doc_sel
        output.console.print(
            f"[green]Focused on:[/] [bold]{doc_sel}[/]\n"
            "[dim]All queries will be scoped to this document. Run /focus to clear.[/]"
        )

    async def cmd_status(self, args_str: str) -> None:
        _ = args_str
        if self.repl.mcp is not None:
            st = await self.repl.mcp.status()
        else:
            st = self.repl.direct.status()
        output.render_status(st)

    async def cmd_help(self, args_str: str) -> None:
        _ = args_str
        output.render_help()

    async def cmd_exit(self, args_str: str) -> None:
        _ = args_str
        self.should_exit = True

    async def cmd_quit(self, args_str: str) -> None:
        """Alias for /exit."""
        await self.cmd_exit(args_str)

    async def cmd_switch(self, args_str: str) -> None:
        name = args_str.strip()
        if not name:
            if self.repl.mcp is not None:
                names = await self.repl.mcp.list_collections()
                pd = (self.repl.mcp.persist_dir or "").strip() or None
            else:
                names = await asyncio.to_thread(self.repl.direct.list_collections)
                pd = (
                    (self.repl.direct.persist_dir or "").strip() or None
                    if self.repl.direct
                    else None
                )
            output.render_collection_names(
                names, empty_persist_dir=pd if not names else None
            )
            return
        if self.repl.mcp is not None:
            self.repl.mcp.collection = name
        else:
            assert self.repl.direct is not None
            self.repl.direct.collection = name
        self.repl.cli_config.collection = name
        self.repl.config_sources["collection"] = "repl"
        output.console.print(f"[green]Active collection:[/] [bold]{name}[/]")

    async def cmd_history(self, args_str: str) -> None:
        _ = args_str
        turns = self.repl.history.get_session(self.repl.session_id)
        output.render_history_turns(turns)

    async def cmd_clear(self, args_str: str) -> None:
        _ = args_str
        self.repl.memory.clear()
        output.console.print("[green]Conversation memory cleared.[/]")

    async def cmd_config(self, args_str: str) -> None:
        s = args_str.strip()
        if not s:
            st = await self.repl._status()
            rc = st.get("collection") if isinstance(st.get("collection"), str) else None
            output.render_config_table(
                effective_config_rows(
                    self.repl.cli_config,
                    self.repl.config_sources,
                    runtime_collection=rc,
                )
            )
            return
        if not s.lower().startswith("set "):
            output.render_error("Usage: /config | /config set <key> <value>")
            return
        try:
            key, value = parse_set_args(s)
        except ValueError as e:
            output.render_error(str(e))
            return
        try:
            set_user_config_key(key, value)
        except ValueError as e:
            output.render_error(str(e))
            return
        self.repl.reload_config_merged()
        output.console.print(
            f"[green]Updated[/] [bold]{USER_CONFIG_PATH}[/]. Settings reloaded."
        )
        if key in ("server_url",):
            output.console.print(
                "[yellow]Reconnect:[/] exit and restart pinrag-cli for MCP URL changes."
            )
