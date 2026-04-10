"""Entry point for pinrag-cli."""

from __future__ import annotations

import argparse
import asyncio
import sys

from pinrag.env_validation import require_llm_api_key

from pinrag_cli.backend import BackendClient
from pinrag_cli.config import CLIConfig, load_config
from pinrag_cli.mcp_backend import MCPBackendClient
from pinrag_cli.repl import REPLApp


async def _async_main(
    *,
    cli_config: CLIConfig,
    config_sources: dict[str, str],
    persist_dir: str | None,
    launch_cli_collection: str | None,
    launch_cli_server: str | None,
    launch_cli_response_style: str | None,
    resume_session_id: str | None,
) -> None:
    mcp_client: MCPBackendClient | None = None
    try:
        if cli_config.server_url:
            mcp_client = MCPBackendClient(cli_config.server_url)
            await mcp_client.connect()
            if cli_config.collection:
                mcp_client.collection = cli_config.collection
            app = REPLApp(
                mcp=mcp_client,
                cli_config=cli_config,
                config_sources=config_sources,
                launch_cli_collection=launch_cli_collection,
                launch_cli_server=launch_cli_server,
                launch_cli_response_style=launch_cli_response_style,
                resume_session_id=resume_session_id,
            )
            await app.run()
        else:
            require_llm_api_key()
            client = BackendClient(
                persist_dir=persist_dir,
                collection=cli_config.collection,
            )
            app = REPLApp(
                direct=client,
                cli_config=cli_config,
                config_sources=config_sources,
                launch_cli_collection=launch_cli_collection,
                launch_cli_server=launch_cli_server,
                launch_cli_response_style=launch_cli_response_style,
                resume_session_id=resume_session_id,
            )
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
    parser.add_argument(
        "--response-style",
        default=None,
        choices=("thorough", "concise"),
        help="RAG answer style (default: config / PINRAG_RESPONSE_STYLE / thorough).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="SESSION_ID",
        help="Resume a previous session (primes memory context; run /sessions to list IDs).",
    )
    args = parser.parse_args()

    cfg, src = load_config(
        cli_collection=args.collection,
        cli_server=args.server,
        cli_response_style=args.response_style,
    )

    try:
        asyncio.run(
            _async_main(
                cli_config=cfg,
                config_sources=src,
                persist_dir=args.persist_dir,
                launch_cli_collection=args.collection,
                launch_cli_server=args.server,
                launch_cli_response_style=args.response_style,
                resume_session_id=args.resume,
            )
        )
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    main()
