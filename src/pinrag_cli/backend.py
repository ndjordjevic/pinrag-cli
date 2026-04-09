"""Backend abstraction direct PinRAG core imports Phase 1."""

from __future__ import annotations

from typing import Any, Literal

from pinrag import __version__ as pinrag_version
from pinrag.config import (
    get_collection_name,
    get_llm_model,
    get_llm_provider,
    get_persist_dir,
)
from pinrag.core import (
    add_files as core_add_files,
    list_documents as core_list_documents,
    query as core_query,
    remove_document as core_remove_document,
)


class BackendClient:
    """Thin wrapper around pinrag.core operations with fixed store location."""

    def __init__(self, *, persist_dir: str | None = None, collection: str | None = None) -> None:
        self.persist_dir = get_persist_dir() if persist_dir is None else persist_dir
        self.collection = get_collection_name() if collection is None else collection

    def query(
        self,
        user_query: str = "",
        document_id: str | None = None,
        page_min: int | None = None,
        page_max: int | None = None,
        tag: str | None = None,
        document_type: str | None = None,
        response_style: Literal["thorough", "concise"] = "thorough",
        verbose_emitter: Any = None,
    ) -> dict[str, Any]:
        return core_query(
            user_query=user_query,
            document_id=document_id,
            page_min=page_min,
            page_max=page_max,
            tag=tag,
            document_type=document_type,
            response_style=response_style,
            persist_dir=self.persist_dir,
            collection=self.collection,
            verbose_emitter=verbose_emitter,
        )

    def add(
        self,
        paths: list[str],
        *,
        tags: list[str] | None = None,
        branch: str | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        verbose_emitter: Any = None,
    ) -> dict[str, Any]:
        return core_add_files(
            paths,
            persist_dir=self.persist_dir,
            collection=self.collection,
            tags=tags,
            branch=branch,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            verbose_emitter=verbose_emitter,
        )

    def list_documents(
        self,
        tag: str | None = None,
        verbose_emitter: Any = None,
    ) -> dict[str, Any]:
        return core_list_documents(
            persist_dir=self.persist_dir,
            collection=self.collection,
            tag=tag,
            verbose_emitter=verbose_emitter,
        )

    def remove(self, document_id: str, verbose_emitter: Any = None) -> dict[str, Any]:
        return core_remove_document(
            document_id,
            persist_dir=self.persist_dir,
            collection=self.collection,
            verbose_emitter=verbose_emitter,
        )

    def status(self) -> dict[str, Any]:
        return {
            "pinrag_version": pinrag_version,
            "persist_dir": self.persist_dir,
            "collection": self.collection,
            "llm_provider": get_llm_provider(),
            "llm_model": get_llm_model(),
        }
