"""Slash-command dispatcher for the PinRAG CLI REPL."""

from __future__ import annotations

from pinrag_cli import output
from pinrag_cli.backend import BackendClient


def _split_tag_args(args_str: str) -> tuple[str | None, str]:
    """Extract optional `--tag TAG` from tail; return (tag, remainder)."""
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


class CommandDispatcher:
    """Maps `/name` to handler methods."""

    def __init__(self, client: BackendClient) -> None:
        self.client = client
        self.should_exit = False

    def dispatch(self, text: str) -> None:
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
            handler(args_str)
        except Exception as e:
            output.render_error(str(e))

    def cmd_add(self, args_str: str) -> None:
        tag, path = _split_tag_args(args_str)
        if not path:
            output.render_error("Usage: /add <path> [--tag TAG]")
            return
        tags = [tag] if tag else None
        result = self.client.add([path], tags=tags)
        output.render_add_result(result)

    def cmd_list(self, args_str: str) -> None:
        tag, _ = _split_tag_args(args_str)
        result = self.client.list_documents(tag=tag if tag else None)
        output.render_documents_table(result)

    def cmd_remove(self, args_str: str) -> None:
        doc_id = args_str.strip()
        if not doc_id:
            output.render_error("Usage: /remove <document_id>")
            return
        result = self.client.remove(doc_id)
        output.render_remove_result(result)

    def cmd_status(self, args_str: str) -> None:
        _ = args_str
        output.render_status(self.client.status())

    def cmd_help(self, args_str: str) -> None:
        _ = args_str
        output.render_help()

    def cmd_exit(self, args_str: str) -> None:
        _ = args_str
        self.should_exit = True

    def cmd_quit(self, args_str: str) -> None:
        """Alias for /exit."""
        self.cmd_exit(args_str)
