"""Tests for CLI config merge (Phase 3a)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pinrag_cli.commands import CommandDispatcher
from pinrag_cli.config import (
    load_config,
    load_toml_file,
    project_config_path,
    read_user_config_dict,
    set_user_config_key,
)


def test_load_toml_missing_returns_empty(tmp_path: Path) -> None:
    assert load_toml_file(tmp_path / "nope.toml") == {}


def test_precedence_project_over_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = tmp_path / "u.toml"
    user.write_text(
        '[defaults]\ncollection = "from_user"\nserver_url = "http://u/mcp"\n',
        encoding="utf-8",
    )
    proj = tmp_path / "proj.toml"
    proj.write_text(
        '[defaults]\ncollection = "from_project"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    cfg, src = load_config(
        user_config_path=user,
        project_config_path_override=proj,
        env={},
    )
    assert cfg.collection == "from_project"
    assert src["collection"] == "project"
    assert cfg.server_url == "http://u/mcp"
    assert src["server_url"] == "user"


def test_precedence_cli_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg, src = load_config(
        cli_collection="cli_coll",
        cli_server="http://cli/mcp",
        cli_response_style="concise",
        env={
            "PINRAG_COLLECTION_NAME": "env_coll",
            "PINRAG_RESPONSE_STYLE": "thorough",
        },
        user_config_path=tmp_path / "none.toml",
    )
    assert cfg.collection == "cli_coll"
    assert src["collection"] == "cli"
    assert cfg.server_url == "http://cli/mcp"
    assert cfg.response_style == "concise"
    assert src["response_style"] == "cli"


def test_precedence_env_over_user_file(tmp_path: Path) -> None:
    user = tmp_path / "u.toml"
    user.write_text(
        '[defaults]\ncollection = "file"\n[memory]\nenabled = false\nturns = 9\n',
        encoding="utf-8",
    )
    cfg, src = load_config(
        user_config_path=user,
        project_config_path_override=tmp_path / ".pinrag-cli.no",
        env={"PINRAG_COLLECTION_NAME": "from_env", "PINRAG_CLI_MEMORY": "1"},
    )
    assert cfg.collection == "from_env"
    assert src["collection"] == "env"
    assert cfg.memory_enabled is True
    assert src["memory_enabled"] == "env"
    assert cfg.memory_turns == 9
    assert src["memory_turns"] == "user"


def test_user_file_memory_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    set_user_config_key("memory.turns", "3", path=p)
    set_user_config_key("memory.enabled", "false", path=p)
    d = read_user_config_dict(p)
    assert d["memory"]["turns"] == 3
    assert d["memory"]["enabled"] is False


def test_cmd_config_renders_table() -> None:
    from pinrag_cli.config import CLIConfig, effective_config_rows, initial_sources

    repl = MagicMock()
    repl.cli_config = CLIConfig(collection="c1", response_style="concise")
    repl.config_sources = initial_sources()
    repl.config_sources["collection"] = "cli"
    repl.reload_config_merged = MagicMock()
    repl._status = AsyncMock(return_value={"collection": "ignored"})
    d = CommandDispatcher(repl)
    with patch("pinrag_cli.commands.output.render_config_table") as mock_tbl:
        asyncio.run(d.cmd_config(""))
        mock_tbl.assert_called_once()
        rows = mock_tbl.call_args[0][0]
        assert rows == effective_config_rows(repl.cli_config, repl.config_sources)
        repl._status.assert_awaited_once()


def test_effective_config_rows_shows_runtime_collection() -> None:
    from pinrag_cli.config import CLIConfig, effective_config_rows, initial_sources

    cfg = CLIConfig()
    src = initial_sources()
    rows = effective_config_rows(
        cfg, src, runtime_collection="my_collection"
    )
    assert rows[0] == ("collection", "my_collection", "effective")


def test_cmd_config_set_writes_and_reloads(tmp_path: Path) -> None:
    from pinrag_cli.config import CLIConfig, initial_sources

    p = tmp_path / "config.toml"
    repl = MagicMock()
    repl.cli_config = CLIConfig()
    repl.config_sources = initial_sources()
    repl.reload_config_merged = MagicMock()

    d = CommandDispatcher(repl)
    with patch("pinrag_cli.config.USER_CONFIG_PATH", p):
        asyncio.run(d.cmd_config("set collection mycol"))
    assert p.is_file()
    data = load_toml_file(p)
    assert data["defaults"]["collection"] == "mycol"
    repl.reload_config_merged.assert_called_once()


def test_reload_config_merged_preserves_switch_collection() -> None:
    from pinrag_cli.config import CLIConfig, initial_sources
    from pinrag_cli.repl import REPLApp

    mcp = MagicMock()
    src = initial_sources()
    src["collection"] = "repl"
    repl = REPLApp(
        mcp=mcp,
        cli_config=CLIConfig(collection="mycol", memory_turns=5),
        config_sources=src,
        launch_cli_collection=None,
        launch_cli_server=None,
        launch_cli_response_style=None,
    )
    merged = CLIConfig(memory_turns=9, memory_enabled=False)
    merged_sources = initial_sources()

    with patch(
        "pinrag_cli.repl.load_config", return_value=(merged, dict(merged_sources))
    ):
        repl.reload_config_merged()

    assert repl.cli_config.memory_turns == 9
    assert repl.cli_config.memory_enabled is False
    assert repl.cli_config.collection == "mycol"
    assert repl.config_sources["collection"] == "repl"
    assert mcp.collection == "mycol"


def test_reload_config_merged_keeps_conversation_memory_when_unrelated() -> None:
    """Changing non-memory settings must not clear the rolling Q/A window."""
    from pinrag_cli.config import CLIConfig, initial_sources
    from pinrag_cli.repl import REPLApp

    mcp = MagicMock()
    repl = REPLApp(
        mcp=mcp,
        cli_config=CLIConfig(memory_turns=5, memory_enabled=True),
        config_sources=initial_sources(),
        launch_cli_collection=None,
        launch_cli_server=None,
        launch_cli_response_style=None,
    )
    mem_before = repl.memory
    repl.memory.add_turn("hello", "world")
    merged = CLIConfig(
        memory_turns=5,
        memory_enabled=True,
        response_style="concise",
    )
    merged_sources = initial_sources()
    merged_sources["response_style"] = "user"

    with patch(
        "pinrag_cli.repl.load_config", return_value=(merged, dict(merged_sources))
    ):
        repl.reload_config_merged()

    assert repl.memory is mem_before
    assert "hello" in repl.memory.build_context_prefix()


def test_reload_config_merged_recreates_memory_when_turns_change() -> None:
    from pinrag_cli.config import CLIConfig, initial_sources
    from pinrag_cli.repl import REPLApp

    mcp = MagicMock()
    repl = REPLApp(
        mcp=mcp,
        cli_config=CLIConfig(memory_turns=5),
        config_sources=initial_sources(),
        launch_cli_collection=None,
        launch_cli_server=None,
        launch_cli_response_style=None,
    )
    mem_before = repl.memory
    merged = CLIConfig(memory_turns=9)

    with patch(
        "pinrag_cli.repl.load_config", return_value=(merged, dict(initial_sources()))
    ):
        repl.reload_config_merged()

    assert repl.memory is not mem_before
    assert repl.cli_config.memory_turns == 9


def test_reload_config_merged_file_collection_overrides_switch() -> None:
    from pinrag_cli.config import CLIConfig, initial_sources
    from pinrag_cli.repl import REPLApp

    mcp = MagicMock()
    src = initial_sources()
    src["collection"] = "repl"
    repl = REPLApp(
        mcp=mcp,
        cli_config=CLIConfig(collection="mycol"),
        config_sources=src,
        launch_cli_collection=None,
        launch_cli_server=None,
        launch_cli_response_style=None,
    )
    merged = CLIConfig(collection="fromtoml")
    merged_sources = initial_sources()
    merged_sources["collection"] = "user"

    with patch(
        "pinrag_cli.repl.load_config", return_value=(merged, dict(merged_sources))
    ):
        repl.reload_config_merged()

    assert repl.cli_config.collection == "fromtoml"
    assert repl.config_sources["collection"] == "user"
    assert mcp.collection == "fromtoml"


def test_project_config_path_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert project_config_path() == tmp_path / ".pinrag-cli.toml"
