"""Unit tests for slash-command parsing and dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pinrag_cli.commands import CommandDispatcher, _split_tag_args


@pytest.mark.parametrize(
    ("raw", "expected_tag", "expected_rest"),
    [
        ("", None, ""),
        ("/path/to/file", None, "/path/to/file"),
        ("--tag t /path", "t", "/path"),
        ("/path --tag mytag", "mytag", "/path"),
        ("  /a/b  --tag  x  ", "x", "/a/b"),
    ],
)
def test_split_tag_args(
    raw: str,
    expected_tag: str | None,
    expected_rest: str,
) -> None:
    tag, rest = _split_tag_args(raw)
    assert tag == expected_tag
    assert rest == expected_rest


def test_dispatch_unknown_command() -> None:
    client = MagicMock()
    d = CommandDispatcher(client)
    d.dispatch("/nosuchthing")
    client.assert_not_called()


def test_cmd_add_passes_tag() -> None:
    client = MagicMock()
    client.add.return_value = {
        "indexed": [],
        "failed": [],
        "total_indexed": 0,
        "total_failed": 0,
        "persist_directory": "/x",
        "collection_name": "y",
    }
    d = CommandDispatcher(client)
    d.cmd_add('/tmp/book.pdf --tag mine')
    client.add.assert_called_once_with(["/tmp/book.pdf"], tags=["mine"])


def test_cmd_exit_sets_flag() -> None:
    client = MagicMock()
    d = CommandDispatcher(client)
    assert not d.should_exit
    d.cmd_exit("")
    assert d.should_exit
