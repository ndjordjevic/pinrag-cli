"""Unit tests for slash-command parsing and dispatch."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from pinrag_cli.commands import CommandDispatcher, _split_ask_args, _split_tag_args


@pytest.mark.parametrize(
    ("raw", "expected_tag", "expected_rest"),
    [
        ("", None, ""),
        ("/path/to/file", None, "/path/to/file"),
        ("--tag t /path", "t", "/path"),
        ("/path --tag mytag", "mytag", "/path"),
        ("  /a/b  --tag  x  ", "x", "/a/b"),
        (
            "'/path/pCloud Drive/file.pdf' --tag AMIGA",
            "AMIGA",
            "/path/pCloud Drive/file.pdf",
        ),
        (
            '"/Volumes/My Disk/Book.pdf" --tag mine',
            "mine",
            "/Volumes/My Disk/Book.pdf",
        ),
        ("/unquoted pCloud Drive/x.pdf --tag t", "t", "/unquoted pCloud Drive/x.pdf"),
        ('--tag "two words" /tmp/a.pdf', "two words", "/tmp/a.pdf"),
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


def _mock_repl() -> MagicMock:
    repl = MagicMock()
    repl.direct = MagicMock()
    repl.mcp = None
    repl.history = MagicMock()
    repl.history.get_session.return_value = []
    repl.session_id = "s1"
    repl._collection_for_history = MagicMock(return_value="pinrag")
    return repl


@pytest.mark.parametrize(
    ("raw", "expected_sel", "expected_q"),
    [
        ("", None, None),
        ("book.pdf -- What is GPIO?", "book.pdf", "What is GPIO?"),
        (
            '"Bare-metal Amiga" -- Summarize chapter 2',
            "Bare-metal Amiga",
            "Summarize chapter 2",
        ),
        (
            "pico -- Compare with RP2040",
            "pico",
            "Compare with RP2040",
        ),
    ],
)
def test_split_ask_args(
    raw: str,
    expected_sel: str | None,
    expected_q: str | None,
) -> None:
    sel, q = _split_ask_args(raw)
    assert sel == expected_sel
    assert q == expected_q


def test_split_ask_args_missing_separator() -> None:
    assert _split_ask_args("only selector") == (None, None)
    assert _split_ask_args("a --") == (None, None)
    assert _split_ask_args("-- only question") == (None, None)


def test_dispatch_unknown_command() -> None:
    repl = _mock_repl()
    d = CommandDispatcher(repl)
    asyncio.run(d.dispatch("/nosuchthing"))
    repl.direct.assert_not_called()


def test_cmd_add_passes_tag() -> None:
    repl = _mock_repl()
    repl.direct.add.return_value = {
        "indexed": [],
        "failed": [],
        "total_indexed": 0,
        "total_failed": 0,
        "persist_directory": "/x",
        "collection_name": "y",
    }
    d = CommandDispatcher(repl)
    asyncio.run(d.cmd_add("/tmp/book.pdf --tag mine"))
    repl.direct.add.assert_called_once()
    call_kw = repl.direct.add.call_args
    assert call_kw[0][0] == ["/tmp/book.pdf"]
    assert call_kw[1]["tags"] == ["mine"]


def test_cmd_add_quoted_path_with_spaces() -> None:
    repl = _mock_repl()
    repl.direct.add.return_value = {
        "indexed": [],
        "failed": [],
        "total_indexed": 0,
        "total_failed": 0,
        "persist_directory": "/x",
        "collection_name": "y",
    }
    d = CommandDispatcher(repl)
    path = "/Users/x/pCloud Drive/Books/Amiga Intern 1992.pdf"
    asyncio.run(d.cmd_add(f"'{path}' --tag AMIGA"))
    repl.direct.add.assert_called_once()
    assert repl.direct.add.call_args[0][0] == [path]
    assert repl.direct.add.call_args[1]["tags"] == ["AMIGA"]


def test_cmd_tag_calls_direct_set_document_tag() -> None:
    repl = _mock_repl()
    repl.direct.set_document_tag.return_value = {
        "document_id": "book.pdf",
        "tag": "AMIGA",
        "updated_chunks": 3,
        "parents_updated": 0,
        "persist_directory": "/x",
        "collection_name": "y",
    }
    d = CommandDispatcher(repl)
    asyncio.run(d.cmd_tag('book.pdf --tag AMIGA'))
    repl.direct.set_document_tag.assert_called_once()
    assert repl.direct.set_document_tag.call_args[0][:2] == ("book.pdf", "AMIGA")


def test_cmd_remove_joins_unquoted_tokens_into_one_ref() -> None:
    repl = _mock_repl()
    repl.direct.remove.return_value = {
        "document_id": "a b c.pdf",
        "deleted_chunks": 3,
        "persist_directory": "/x",
        "collection_name": "y",
    }
    d = CommandDispatcher(repl)
    asyncio.run(d.cmd_remove("a b c.pdf"))
    repl.direct.remove.assert_called_once()
    assert repl.direct.remove.call_args[0][0] == "a b c.pdf"


def test_cmd_exit_sets_flag() -> None:
    repl = _mock_repl()
    d = CommandDispatcher(repl)
    assert not d.should_exit
    asyncio.run(d.cmd_exit(""))
    assert d.should_exit


@patch("pinrag_cli.commands.output.render_error")
def test_cmd_ask_shows_usage_when_no_separator(mock_err: MagicMock) -> None:
    repl = _mock_repl()
    d = CommandDispatcher(repl)
    asyncio.run(d.cmd_ask("missing dash"))
    mock_err.assert_called_once()
    repl.direct.query.assert_not_called()


def test_cmd_ask_calls_direct_query_with_document_id() -> None:
    repl = _mock_repl()
    repl.direct.query.return_value = {
        "answer": "ok",
        "sources": [{"document_id": "book.pdf", "page": 1}],
    }
    d = CommandDispatcher(repl)
    asyncio.run(d.cmd_ask("book.pdf -- What is up?"))
    repl.direct.query.assert_called_once()
    assert repl.direct.query.call_args[0][0] == "What is up?"
    assert repl.direct.query.call_args[1]["document_id"] == "book.pdf"
    repl.history.add_turn.assert_called_once()
    assert repl.history.add_turn.call_args[0][1] == "/ask book.pdf -- What is up?"
