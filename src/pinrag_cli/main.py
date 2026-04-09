"""Entry point for pinrag-cli."""

from __future__ import annotations

import argparse
import asyncio
import sys

from pinrag.env_validation import require_llm_api_key

from pinrag_cli.backend import BackendClient
from pinrag_cli.mcp_backend import MCPBackendClient
from pinrag_cli.repl import REPLApp


async def _async_main(
    *,
    server_url: str | None,
    persist_dir: str | None,
    collection: str | None,
) -> None:
    mcp_client: MCPBackendClient | None = None
    try:
        if server_url:
            mcp_client = MCPBackendClient(server_url)
            await mcp_client.connect()
            if collection:
                mcp_client.collection = collection
            app = REPLApp(mcp=mcp_client)
            await app.run()
        else:
            require_llm_api_key()
            client = BackendClient(persist_dir=persist_dir, collection=collection)
            app = REPLApp(direct=client)
            await app.run()
    finally:
        if mcp_client is not None:
            await mcp_client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PinRAG interactive CLI (query and manage the local index).",
    )
    parser.add_argument(
        "--persist-dir",
        default=None,
        help="Chroma persist directory (default: PINRAG_PERSIST_DIR or chroma_db)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Chroma collection name (default: PINRAG_COLLECTION_NAME or pinrag)",
    )
    parser.add_argument(
        "--server",
        default=None,
        metavar="URL",
        help=(
            "MCP streamable-http URL (e.g. http://127.0.0.1:8765/mcp). "
            "Use with `pinrag server`; skips local LLM key check."
        ),
    )
    args = parser.parse_args()

    try:
        asyncio.run(
            _async_main(
                server_url=args.server,
                persist_dir=args.persist_dir,
                collection=args.collection,
            )
        )
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    main()
