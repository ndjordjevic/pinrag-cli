"""Entry point for pinrag-cli."""

from __future__ import annotations

import argparse
import sys

from pinrag.env_validation import require_llm_api_key

from pinrag_cli.backend import BackendClient
from pinrag_cli.repl import REPLApp


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
    args = parser.parse_args()

    require_llm_api_key()

    client = BackendClient(
        persist_dir=args.persist_dir,
        collection=args.collection,
    )
    app = REPLApp(client)
    try:
        app.run()
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    main()
