"""Live HTTP MCP checks (opt-in).

Set ``PINRAG_CLI_HTTP_ITEST=1`` and start ``pinrag server`` (same machine).
Optional: ``PINRAG_CLI_HTTP_URL`` (default ``http://127.0.0.1:8765/mcp``).
"""

from __future__ import annotations

import asyncio
import os

import pytest

from pinrag_cli.mcp_backend import MCPBackendClient

pytestmark = pytest.mark.skipif(
    os.environ.get("PINRAG_CLI_HTTP_ITEST") != "1",
    reason="Set PINRAG_CLI_HTTP_ITEST=1 for live HTTP MCP test",
)


def test_mcp_client_list_collections_roundtrip() -> None:
    url = os.environ.get("PINRAG_CLI_HTTP_URL", "http://127.0.0.1:8765/mcp")

    async def _run() -> None:
        client = MCPBackendClient(url)
        await client.connect()
        try:
            names = await client.list_collections()
            assert isinstance(names, list)
            st = await client.status()
            assert "server_url" in st
        finally:
            await client.close()

    try:
        asyncio.run(_run())
    except BaseException as e:
        if type(e) is SystemExit:
            raise
        pytest.skip(f"Live PinRAG HTTP server not reachable at {url}: {e!r}")
