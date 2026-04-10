"""Interactive REPL using prompt_toolkit."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from pinrag.config import get_collection_name
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from pinrag_cli import output
from pinrag_cli.backend import BackendClient
from pinrag_cli.commands import CommandDispatcher
from pinrag_cli.config import CLIConfig, load_config
from pinrag_cli.history import ConversationStore
from pinrag_cli.memory import ConversationMemory

if TYPE_CHECKING:
    from pinrag_cli.mcp_backend import MCPBackendClient


class REPLApp:
    """Read lines; slash commands or RAG queries."""

    def __init__(
        self,
        *,
        direct: BackendClient | None = None,
        mcp: MCPBackendClient | None = None,
        cli_config: CLIConfig,
        config_sources: dict[str, str],
        launch_cli_collection: str | None,
        launch_cli_server: str | None,
        launch_cli_response_style: str | None,
        resume_session_id: str | None = None,
    ) -> None:
        if (direct is None) == (mcp is None):
            raise ValueError("Provide exactly one of direct= or mcp=")
        self.direct = direct
        self.mcp = mcp
        self.cli_config = cli_config
        self.config_sources = dict(config_sources)
        self._launch_cli_collection = launch_cli_collection
        self._launch_cli_server = launch_cli_server
        self._launch_cli_response_style = launch_cli_response_style
        self.commands = CommandDispatcher(self)
        self.history = ConversationStore()
        self.memory = ConversationMemory(
            max_turns=cli_config.memory_turns,
            enabled=cli_config.memory_enabled,
        )
        self.session_id = self.history.new_session()
        self.focused_doc: str | None = None
        if resume_session_id:
            self._resume_session(resume_session_id)
        history_path = Path.home() / ".pinrag_cli_history"
        self.session = PromptSession(history=FileHistory(str(history_path)))
        self._prompt_style = Style.from_dict(
            {"pinrag-prompt": "bold ansibrightcyan"}
        )

    @property
    def _prompt_message(self) -> FormattedText:
        if self.focused_doc:
            label = f"pinrag[{self.focused_doc}]> "
        else:
            label = "pinrag> "
        return FormattedText([("class:pinrag-prompt", label)])

    def _resume_session(self, session_id: str) -> int:
        """Prime memory from a prior session's last N turns. Returns turn count loaded."""
        turns = self.history.get_session(session_id)
        if not turns:
            return 0
        self.memory.clear()
        cap = self.memory._window.maxlen or len(turns)
        to_load = turns[-cap:]
        for t in to_load:
            self.memory.add_turn(t.get("query", ""), t.get("answer", ""))
        return len(to_load)

    def _response_style_literal(self) -> Literal["thorough", "concise"]:
        return cast(
            Literal["thorough", "concise"],
            self.cli_config.response_style,
        )

    def reload_config_merged(self) -> None:
        """Reload TOML and env; reapply launcher CLI overrides; sync memory."""
        prev_turns = self.cli_config.memory_turns
        prev_mem_enabled = self.cli_config.memory_enabled

        repl_collection: str | None = None
        if self.config_sources.get("collection") == "repl":
            repl_collection = self.cli_config.collection

        cfg, src = load_config(
            cli_collection=self._launch_cli_collection,
            cli_server=self._launch_cli_server,
            cli_response_style=self._launch_cli_response_style,
        )

        if repl_collection is not None:
            persisting = frozenset({"cli", "env", "project", "user"})
            if src.get("collection") not in persisting:
                cfg.collection = repl_collection
                src["collection"] = "repl"

        self.cli_config = cfg
        self.config_sources = src
        if cfg.memory_turns != prev_turns or cfg.memory_enabled != prev_mem_enabled:
            self.memory = ConversationMemory(
                max_turns=cfg.memory_turns,
                enabled=cfg.memory_enabled,
            )
        self._sync_collection_to_backend()

    def _sync_collection_to_backend(self) -> None:
        c = self.cli_config.collection
        if self.mcp is not None:
            self.mcp.collection = c
        elif self.direct is not None:
            self.direct.collection = (
                c if c is not None else get_collection_name()
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
            prefix = self.memory.build_context_prefix()
            augmented = prefix + text if prefix else text
            rs = self._response_style_literal()
            with output.StreamingDisplay() as stream:

                async def prog(p: float, tot: float | None, msg: str | None) -> None:
                    stream.update_progress(p, tot, msg)

                def verb(message: str, _level: str) -> None:
                    stream.update(message)

                if self.mcp is not None:
                    result = await self.mcp.query(
                        augmented,
                        document_id=self.focused_doc,
                        progress_callback=prog,
                        response_style=rs,
                    )
                else:
                    assert self.direct is not None
                    result = await asyncio.to_thread(
                        self.direct.query,
                        augmented,
                        document_id=self.focused_doc,
                        verbose_emitter=verb,
                        response_style=rs,
                    )
            output.render_query_result(result)
            self.memory.add_turn(text, str(result.get("answer", "")))
            self.history.add_turn(
                self.session_id,
                text,
                result,
                collection=self._collection_for_history(),
            )
        except Exception as e:
            output.render_error(str(e))
