"""Tests for parsing pinrag://server-config text in the MCP client."""

from __future__ import annotations

from pinrag_cli.mcp_backend import parse_pinrag_server_config_text


def test_parse_server_config_extracts_banner_keys() -> None:
    body = """
--- Explicitly set (runtime env) ---
  PINRAG_COLLECTION_NAME: mycoll
  PINRAG_LLM_PROVIDER: openrouter
--- Defaults (not set in env) ---
  PINRAG_LLM_MODEL: openrouter/free
"""
    out = parse_pinrag_server_config_text(body)
    assert out["PINRAG_COLLECTION_NAME"] == "mycoll"
    assert out["PINRAG_LLM_PROVIDER"] == "openrouter"
    assert out["PINRAG_LLM_MODEL"] == "openrouter/free"


def test_parse_server_config_first_occurrence_wins() -> None:
    body = """
  PINRAG_COLLECTION_NAME: first
  PINRAG_COLLECTION_NAME: second
"""
    out = parse_pinrag_server_config_text(body)
    assert out["PINRAG_COLLECTION_NAME"] == "first"
