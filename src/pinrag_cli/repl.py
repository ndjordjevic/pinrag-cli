"""Interactive REPL using prompt_toolkit."""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from pinrag_cli.backend import BackendClient
from pinrag_cli.commands import CommandDispatcher
from pinrag_cli import output


class REPLApp:
    """Read lines; slash commands or RAG queries."""

    def __init__(self, client: BackendClient) -> None:
        self.client = client
        self.commands = CommandDispatcher(client)
        history_path = Path.home() / ".pinrag_cli_history"
        self.session = PromptSession(history=FileHistory(str(history_path)))

    def run(self) -> None:
        output.render_banner(self.client.status())
        while True:
            try:
                text = self.session.prompt("pinrag> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text.startswith("/"):
                self.commands.dispatch(text)
                if self.commands.should_exit:
                    break
            else:
                self._handle_query(text)
        output.console.print("[dim]Goodbye.[/]")

    def _handle_query(self, text: str) -> None:
        try:
            result = self.client.query(text)
            output.render_query_result(result)
        except Exception as e:
            output.render_error(str(e))
