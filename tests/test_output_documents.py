"""Tests for document list rendering helpers."""

from __future__ import annotations

from unittest.mock import patch

from pinrag_cli.output import (
    _document_extent_and_extra,
    _format_bytes_cell,
    _format_source_location_cell,
    _format_uploaded_cell,
    render_remove_result,
    render_set_tag_result,
)


def test_format_bytes_cell() -> None:
    assert _format_bytes_cell(None) == ""
    assert _format_bytes_cell(0) == "0 B"
    assert _format_bytes_cell(500) == "500 B"
    assert _format_bytes_cell(2048).endswith("KB")


def test_format_uploaded_cell_iso_to_utc_minute() -> None:
    assert (
        _format_uploaded_cell("2026-03-30T10:36:33.446693+00:00") == "2026-03-30 10:36"
    )
    assert _format_uploaded_cell("2026-03-30T12:00:00Z") == "2026-03-30 12:00"


def test_format_uploaded_cell_non_iso_passthrough_or_trim() -> None:
    assert _format_uploaded_cell("") == ""
    assert _format_uploaded_cell(None) == ""
    assert _format_uploaded_cell("not-a-date") == "not-a-date"


def test_render_set_tag_success_panel() -> None:
    with patch("pinrag_cli.output.console.print") as mock_print:
        render_set_tag_result(
            {
                "document_id": "a.pdf",
                "tag": "T",
                "updated_chunks": 2,
                "parents_updated": 1,
            }
        )
    mock_print.assert_called_once()
    panel = mock_print.call_args[0][0]
    assert "tag set" in str(panel.renderable).lower()


def test_render_set_tag_warns_when_zero_chunks() -> None:
    with patch("pinrag_cli.output.console.print") as mock_print:
        render_set_tag_result(
            {"document_id": "x", "tag": "T", "updated_chunks": 0}
        )
    mock_print.assert_called_once()
    assert "no chunks updated" in str(mock_print.call_args[0][0].title).lower()


def test_render_remove_result_warns_when_zero_chunks_deleted() -> None:
    with patch("pinrag_cli.output.console.print") as mock_print:
        render_remove_result({"document_id": "Amiga Intern 1992", "deleted_chunks": 0})
    mock_print.assert_called_once()
    panel = mock_print.call_args[0][0]
    text = str(panel.renderable)
    assert "No chunks matched" in text


def test_source_location_cell_pdf_pages() -> None:
    assert (
        _format_source_location_cell(
            [
                {"document_id": "a.pdf", "page": 2, "document_type": "pdf"},
                {"document_id": "a.pdf", "page": 1, "document_type": "pdf"},
            ]
        )
        == "1, 2"
    )


def test_source_location_cell_web_urls_not_zero() -> None:
    cell = _format_source_location_cell(
        [
            {
                "document_id": "picocomputer.github.io/",
                "page": 0,
                "document_type": "web",
                "source": "https://picocomputer.github.io/hardware.html",
            },
        ]
    )
    assert "picocomputer.github.io" in cell
    assert cell != "0"


def test_source_location_cell_youtube_timestamps() -> None:
    assert (
        _format_source_location_cell(
            [
                {"document_id": "v", "page": 0, "start": 83, "document_type": "youtube"},
                {"document_id": "v", "page": 0, "start": 125, "document_type": "youtube"},
            ]
        )
        == "1:23, 2:05"
    )


def test_extent_includes_unknown_keys() -> None:
    info = {
        "pages": 3,
        "segments": 10,
        "custom_key": "x",
        "document_type": "pdf",
    }
    s = _document_extent_and_extra(info)
    assert "pages=3" in s
    assert "segments=10" in s
    assert "custom_key=x" in s
    assert "document_type" not in s
