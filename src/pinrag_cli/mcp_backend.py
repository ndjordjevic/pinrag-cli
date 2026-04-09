"""MCP streamable-http backend for pinrag-cli (Phase 2)."""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from typing import Any, Literal

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent

logger = logging.getLogger(__name__)

_BANNER_CONFIG_KEYS = (
    "PINRAG_COLLECTION_NAME",
    "PINRAG_LLM_PROVIDER",
    "PINRAG_LLM_MODEL",
)


def parse_pinrag_server_config_text(text: str) -> dict[str, str]:
    """Extract effective collection / LLM fields from ``pinrag://server-config`` body.

    The resource repeats keys under \"explicitly set\" and \"defaults\"; the first
    occurrence wins (same effective value).
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        for key in _BANNER_CONFIG_KEYS:
            if key in out:
                continue
            prefix = f"{key}:"
            if s.startswith(prefix):
                out[key] = s[len(prefix) :].strip()
                break
    return out


def _tool_result_dict(result: CallToolResult) -> dict[str, Any]:
    """Parse FastMCP / MCP tool result into a dict.

    FastMCP wraps the actual payload under a ``"result"`` key in
    ``structuredContent`` (e.g. ``{"result": {"collections": [...], ...}}``).
    We unwrap that envelope so callers always get the inner payload dict.
    The text-content fallback path already contains the unwrapped payload.
    """
    if result.isError:
        msg = "tool error"
        for block in result.content:
            if isinstance(block, TextContent):
                msg = block.text
                break
        raise RuntimeError(msg)
    if result.structuredContent:
        data = dict(result.structuredContent)
        # FastMCP envelopes the payload under a single "result" key.
        if list(data.keys()) == ["result"] and isinstance(data["result"], dict):
            return data["result"]
        return data
    for block in result.content:
        if isinstance(block, TextContent) and block.text:
            return json.loads(block.text)
    raise ValueError("Empty or unparseable tool result")


class MCPBackendClient:
    """Thin async client over PinRAG MCP (streamable-http)."""

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url.rstrip("/")
        self.collection: str | None = None
        self.persist_dir: str = ""
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    def _collection_arg(self) -> str:
        return (self.collection or "").strip()

    async def connect(self) -> None:
        """Open HTTP transport and initialize MCP session."""
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            read, write, _get_id = await stack.enter_async_context(
                streamable_http_client(self.server_url)
            )
            session = ClientSession(read, write)
            await stack.enter_async_context(session)
            await session.initialize()
            self._stack = stack
            self._session = session
        except Exception:
            await stack.__aexit__(None, None, None)
            raise

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(None, None, None)
        self._stack = None
        self._session = None

    async def _call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("MCP client not connected; call connect() first")
        result = await self._session.call_tool(
            name,
            arguments,
            progress_callback=progress_callback,
        )
        out = _tool_result_dict(result)
        if isinstance(out, dict) and "persist_directory" in out:
            self.persist_dir = str(out["persist_directory"])
        return out

    async def query(
        self,
        user_query: str,
        *,
        document_id: str | None = None,
        page_min: int | None = None,
        page_max: int | None = None,
        tag: str | None = None,
        document_type: str | None = None,
        response_style: Literal["thorough", "concise"] = "thorough",
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {
            "query": user_query,
            "response_style": response_style,
        }
        c = self._collection_arg()
        if c:
            args["collection"] = c
        if document_id:
            args["document_id"] = document_id
        if page_min is not None:
            args["page_min"] = page_min
        if page_max is not None:
            args["page_max"] = page_max
        if tag:
            args["tag"] = tag
        if document_type:
            args["document_type"] = document_type
        return await self._call_tool(
            "query_tool", args, progress_callback=progress_callback
        )

    async def add(
        self,
        paths: list[str],
        *,
        tags: list[str] | None = None,
        branch: str | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"paths": paths}
        c = self._collection_arg()
        if c:
            args["collection"] = c
        if tags is not None:
            args["tags"] = tags
        if branch:
            args["branch"] = branch
        if include_patterns:
            args["include_patterns"] = include_patterns
        if exclude_patterns:
            args["exclude_patterns"] = exclude_patterns
        return await self._call_tool(
            "add_document_tool", args, progress_callback=progress_callback
        )

    async def list_documents(
        self,
        tag: str | None = None,
        *,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {}
        c = self._collection_arg()
        if c:
            args["collection"] = c
        if tag:
            args["tag"] = tag
        return await self._call_tool(
            "list_documents_tool", args, progress_callback=progress_callback
        )

    async def remove(
        self,
        document_id: str,
        *,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"document_id": document_id}
        c = self._collection_arg()
        if c:
            args["collection"] = c
        return await self._call_tool(
            "remove_document_tool", args, progress_callback=progress_callback
        )

    async def set_document_tag(
        self,
        document_id: str,
        tag: str,
        *,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"document_id": document_id, "tag": tag}
        c = self._collection_arg()
        if c:
            args["collection"] = c
        return await self._call_tool(
            "set_document_tag_tool",
            args,
            progress_callback=progress_callback,
        )

    async def list_collections(
        self,
        *,
        progress_callback: Any | None = None,
    ) -> list[str]:
        args: dict[str, Any] = {}
        if self.persist_dir:
            args["persist_dir"] = self.persist_dir
        data = await self._call_tool(
            "list_collections_tool", args, progress_callback=progress_callback
        )
        return list(data.get("collections") or [])

    async def status(self) -> dict[str, Any]:
        """Probe server: list_collections + pinrag://server-config for banner fields."""
        data: dict[str, Any] = {}
        try:
            data = await self._call_tool("list_collections_tool", {})
        except Exception as e:
            logger.debug("status list_collections_tool failed: %s", e)
        ver = str(data.get("_server_version", "mcp")) if data else "mcp"

        cfg: dict[str, str] = {}
        if self._session is not None:
            try:
                from pydantic import AnyUrl

                rr = await self._session.read_resource(
                    AnyUrl("pinrag://server-config")
                )
                chunks: list[str] = []
                for block in rr.contents:
                    t = getattr(block, "text", None)
                    if t:
                        chunks.append(t)
                if chunks:
                    cfg = parse_pinrag_server_config_text("\n".join(chunks))
            except Exception as e:
                logger.debug("status read_resource pinrag://server-config failed: %s", e)

        coll = (
            (self.collection or "").strip()
            or cfg.get("PINRAG_COLLECTION_NAME")
            or "(server default)"
        )
        provider = cfg.get("PINRAG_LLM_PROVIDER") or "(unknown)"
        model = cfg.get("PINRAG_LLM_MODEL") or "(unknown)"

        return {
            "pinrag_version": ver,
            "persist_dir": self.persist_dir or "(server)",
            "collection": coll,
            "llm_provider": provider,
            "llm_model": model,
            "server_url": self.server_url,
        }
