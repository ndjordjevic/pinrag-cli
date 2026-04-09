"""CLI configuration: TOML files, env, and CLI flags (Phase 3a).

Precedence (highest wins): CLI flags > env vars > project `.pinrag-cli.toml` >
user `~/.config/pinrag-cli/config.toml` > hardcoded defaults.

Only CLI-owned settings live here; persist_dir and LLM settings remain on the
pinrag server / ``pinrag.config`` env.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

USER_CONFIG_PATH = Path.home() / ".config" / "pinrag-cli" / "config.toml"
PROJECT_CONFIG_FILENAME = ".pinrag-cli.toml"

_VALID_RESPONSE_STYLES = frozenset({"thorough", "concise"})


@dataclass
class CLIConfig:
    """Effective CLI settings after merge."""

    collection: str | None = None
    server_url: str | None = None
    response_style: str = "thorough"
    memory_enabled: bool = True
    memory_turns: int = 5


def project_config_path(cwd: Path | None = None) -> Path:
    return (cwd or Path.cwd()) / PROJECT_CONFIG_FILENAME


def load_toml_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    import tomllib  # stdlib py3.11+

    with path.open("rb") as f:
        return tomllib.load(f)


def normalize_response_style(raw: str | None) -> str:
    if raw is None or not str(raw).strip():
        return "thorough"
    v = str(raw).strip().lower()
    return v if v in _VALID_RESPONSE_STYLES else "thorough"


def _truthy_env(raw: str | None, *, default: bool = True) -> bool:
    """Same semantics as ``memory._truthy_env`` for PINRAG_CLI_MEMORY."""
    if raw is None or not str(raw).strip():
        return default
    v = str(raw).strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return default


def _positive_int_env(raw: str | None, default: int) -> int:
    if raw is None or not str(raw).strip():
        return default
    try:
        n = int(str(raw).strip(), 10)
    except ValueError:
        return default
    return max(1, n)


def _norm_collection(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    return s if s else None


def _norm_server_url(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    return s if s else None


def _apply_toml_dict(
    cfg: CLIConfig,
    sources: dict[str, str],
    data: dict[str, Any],
    layer: str,
) -> None:
    defaults = data.get("defaults")
    if isinstance(defaults, dict):
        if "collection" in defaults:
            c = _norm_collection(defaults.get("collection"))
            if c is not None:
                cfg.collection = c
                sources["collection"] = layer
        if "server_url" in defaults:
            u = _norm_server_url(defaults.get("server_url"))
            cfg.server_url = u
            sources["server_url"] = layer
        if "response_style" in defaults:
            rs = normalize_response_style(str(defaults.get("response_style")))
            cfg.response_style = rs
            sources["response_style"] = layer

    mem = data.get("memory")
    if isinstance(mem, dict):
        if "enabled" in mem:
            v = mem["enabled"]
            if isinstance(v, bool):
                cfg.memory_enabled = v
            elif isinstance(v, (int, float)) and v in (0, 1):
                cfg.memory_enabled = bool(v)
            else:
                cfg.memory_enabled = _truthy_env(
                    str(v) if v is not None else None,
                    default=cfg.memory_enabled,
                )
            sources["memory_enabled"] = layer
        if "turns" in mem:
            try:
                cfg.memory_turns = max(1, int(mem["turns"]))
            except (TypeError, ValueError):
                pass
            else:
                sources["memory_turns"] = layer


def _apply_env(cfg: CLIConfig, sources: dict[str, str], env: Mapping[str, str]) -> None:
    raw = env.get("PINRAG_COLLECTION_NAME")
    if raw is not None and str(raw).strip():
        cfg.collection = str(raw).strip()
        sources["collection"] = "env"

    raw = env.get("PINRAG_RESPONSE_STYLE")
    if raw is not None and str(raw).strip():
        cfg.response_style = normalize_response_style(str(raw))
        sources["response_style"] = "env"

    if "PINRAG_CLI_MEMORY" in env:
        cfg.memory_enabled = _truthy_env(
            env.get("PINRAG_CLI_MEMORY"),
            default=True,
        )
        sources["memory_enabled"] = "env"

    if "PINRAG_CLI_MEMORY_TURNS" in env:
        cfg.memory_turns = _positive_int_env(
            env.get("PINRAG_CLI_MEMORY_TURNS"),
            cfg.memory_turns,
        )
        sources["memory_turns"] = "env"


def _apply_cli_flags(
    cfg: CLIConfig,
    sources: dict[str, str],
    *,
    collection: str | None,
    server: str | None,
    response_style: str | None = None,
) -> None:
    if collection is not None and str(collection).strip():
        cfg.collection = str(collection).strip()
        sources["collection"] = "cli"

    if server is not None:
        u = str(server).strip()
        cfg.server_url = u if u else None
        sources["server_url"] = "cli"

    if response_style is not None and str(response_style).strip():
        cfg.response_style = normalize_response_style(str(response_style))
        sources["response_style"] = "cli"


def initial_sources() -> dict[str, str]:
    return {
        "collection": "default",
        "server_url": "default",
        "response_style": "default",
        "memory_enabled": "default",
        "memory_turns": "default",
    }


def load_config(
    *,
    cli_collection: str | None = None,
    cli_server: str | None = None,
    cli_response_style: str | None = None,
    env: Mapping[str, str] | None = None,
    user_config_path: Path | None = None,
    project_config_path_override: Path | None = None,
    cwd: Path | None = None,
) -> tuple[CLIConfig, dict[str, str]]:
    """Merge config layers; return effective config and per-field source labels."""
    environ: Mapping[str, str] = env if env is not None else os.environ
    user_path = user_config_path or USER_CONFIG_PATH
    proj_path = project_config_path_override or project_config_path(cwd)

    cfg = CLIConfig()
    sources = initial_sources()

    _apply_toml_dict(cfg, sources, load_toml_file(user_path), "user")
    _apply_toml_dict(cfg, sources, load_toml_file(proj_path), "project")
    _apply_env(cfg, sources, environ)
    _apply_cli_flags(
        cfg,
        sources,
        collection=cli_collection,
        server=cli_server,
        response_style=cli_response_style,
    )
    return cfg, sources


def effective_config_rows(
    cfg: CLIConfig,
    sources: dict[str, str],
    *,
    runtime_collection: str | None = None,
) -> list[tuple[str, str, str]]:
    """Rows for Rich table: (key, value, source).

    ``runtime_collection`` is the active collection (e.g. from ``REPLApp._status()``),
    shown when there is no CLI-layer override in ``cfg.collection``.
    """
    if cfg.collection is not None:
        coll_val = cfg.collection
        coll_src = sources.get("collection", "default")
    elif runtime_collection:
        coll_val = runtime_collection
        coll_src = "effective"
    else:
        coll_val = "(server / env default)"
        coll_src = sources.get("collection", "default")
    return [
        (
            "collection",
            coll_val,
            coll_src,
        ),
        (
            "server_url",
            cfg.server_url if cfg.server_url else "(direct mode)",
            sources.get("server_url", "default"),
        ),
        (
            "response_style",
            cfg.response_style,
            sources.get("response_style", "default"),
        ),
        (
            "memory.enabled",
            str(cfg.memory_enabled).lower(),
            sources.get("memory_enabled", "default"),
        ),
        (
            "memory.turns",
            str(cfg.memory_turns),
            sources.get("memory_turns", "default"),
        ),
    ]


# --- User config file writes (/config set) ---


def _toml_string(s: str) -> str:
    """Double-quoted TOML string with minimal escaping."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _toml_format_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return _toml_string(v)
    return _toml_string(str(v))


def _deep_merge_dict(
    base: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = _deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def render_user_toml(data: dict[str, Any]) -> str:
    """Serialize a small nested dict as TOML (defaults + memory sections)."""
    lines: list[str] = []
    if "defaults" in data and isinstance(data["defaults"], dict):
        lines.append("[defaults]")
        for key in sorted(data["defaults"]):
            lines.append(f"{key} = {_toml_format_value(data['defaults'][key])}")
        lines.append("")
    if "memory" in data and isinstance(data["memory"], dict):
        lines.append("[memory]")
        for key in sorted(data["memory"]):
            lines.append(f"{key} = {_toml_format_value(data['memory'][key])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def read_user_config_dict(path: Path | None = None) -> dict[str, Any]:
    raw = load_toml_file(path or USER_CONFIG_PATH)
    out: dict[str, Any] = {}
    if isinstance(raw.get("defaults"), dict):
        out["defaults"] = dict(raw["defaults"])
    if isinstance(raw.get("memory"), dict):
        out["memory"] = dict(raw["memory"])
    return out


def set_user_config_key(
    key: str,
    value_str: str,
    *,
    path: Path | None = None,
) -> None:
    """Parse ``key`` and ``value_str`` and merge into the user config file.

    Supported keys: ``collection``, ``server_url``, ``response_style``,
    ``memory.enabled`` / ``memory_enabled``, ``memory.turns`` / ``memory_turns``.
    """
    path = path or USER_CONFIG_PATH
    key = key.strip()
    current = read_user_config_dict(path)

    if key in ("memory.enabled", "memory_enabled"):
        vlow = value_str.strip().lower()
        enabled = vlow not in ("0", "false", "no", "off")
        patch = {"memory": {"enabled": enabled}}
    elif key in ("memory.turns", "memory_turns"):
        n = int(value_str.strip(), 10)
        if n < 1:
            raise ValueError("memory turns must be >= 1")
        patch = {"memory": {"turns": n}}
    elif key == "collection":
        patch = {"defaults": {"collection": value_str.strip()}}
    elif key == "server_url":
        patch = {"defaults": {"server_url": value_str.strip()}}
    elif key == "response_style":
        rs = normalize_response_style(value_str)
        patch = {"defaults": {"response_style": rs}}
    else:
        raise ValueError(
            f"Unknown config key: {key!r}. "
            "Try: collection, server_url, response_style, "
            "memory.enabled, memory.turns"
        )

    merged = _deep_merge_dict(current, patch)
    text = render_user_toml(merged)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_set_args(args_str: str) -> tuple[str, str]:
    """``/config set KEY VALUE...`` → (key, value)."""
    s = args_str.strip()
    if not s.lower().startswith("set "):
        raise ValueError("Usage: /config set <key> <value>")
    rest = s[4:].strip()
    if not rest:
        raise ValueError("Usage: /config set <key> <value>")
    parts = rest.split(maxsplit=1)
    if len(parts) < 2:
        raise ValueError("Usage: /config set <key> <value>")
    return parts[0].strip(), parts[1].strip()
