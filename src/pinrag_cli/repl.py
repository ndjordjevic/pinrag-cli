"""Interactive REPL using prompt_toolkit."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from pinrag_cli import output
from pinrag_cli.backend import BackendClient
from pinrag_cli.commands import CommandDispatcher
from pinrag_cli.history import ConversationStore

if TYPE_CHECKING:
    from pinrag_cli.mcp_backend import MCPBackendClient


class REPLApp:
    """Read lines; slash commands or RAG queries."""

    def __init__(
        self,
        *,
        direct: BackendClient | None = None,
        mcp: MCPBackendClient | None = None,
    ) -> None:
        if (direct is None) == (mcp is None):
            raise ValueError("Provide exactly one of direct= or mcp=")
        self.direct = direct
        self.mcp = mcp
        self.commands = CommandDispatcher(self)
        self.history = ConversationStore()
        self.session_id = self.history.new_session()
        history_path = Path.home() / ".pinrag_cli_history"
        self.session = PromptSession(history=FileHistory(str(history_path)))
        self._prompt_message = FormattedText([("class:pinrag-prompt", "pinrag> ")])
        self._prompt_style = Style.from_dict(
            {"pinrag-prompt": "bold ansibrightcyan"}
        )

    def _collection_for_history(self) -> str | None:
        if self.mcp is not None:
            return self.mcp.collection
        assert self.direct is not None
        return self.direct.collection

    async def _status(self) -> dict:
        if self.mcp is not None:
            return await self.mcp.status()
        assert self.direct is not None
        return self.direct.status()

    async def run(self) -> None:
        output.render_banner(await self._status())
        while True:
            try:
                text = (
                    await self.session.prompt_async(
                        self._prompt_message,
                        style=self._prompt_style,
                    )
                ).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text.startswith("/"):
                await self.commands.dispatch(text)
                if self.commands.should_exit:
                    break
            else:
                await self._handle_query(text)
        output.console.print("[dim]Goodbye.[/]")

    async def _handle_query(self, text: str) -> None:
        try:
            with output.StreamingDisplay() as stream:

                async def prog(p: float, tot: float | None, msg: str | None) -> None:
                    stream.update_progress(p, tot, msg)

                def verb(message: str, _level: str) -> None:
                    stream.update(message)

                if self.mcp is not None:
                    result = await self.mcp.query(
                        text,
                        progress_callback=prog,
                    )
                else:
                    assert self.direct is not None
                    result = await asyncio.to_thread(
                        self.direct.query,
                        text,
                        verbose_emitter=verb,
                    )
            output.render_query_result(result)
            self.history.add_turn(
                self.session_id,
                text,
                result,
                collection=self._collection_for_history(),
            )
        except Exception as e:
            output.render_error(str(e))
