# pinrag-cli

Interactive REPL for [PinRAG](https://github.com/ndjordjevic/pinrag): query indexed documents, add paths, list and remove documents.

## Requirements

- Python 3.12+
- **Direct mode** (default): same environment variables as PinRAG (e.g. `OPENROUTER_API_KEY` when using OpenRouter)
- **Server mode** (`--server`): no local LLM key needed; run `pinrag server` in another terminal with keys set there
- Optional: `PINRAG_PERSIST_DIR`, `PINRAG_COLLECTION_NAME`

## Install

From this repo (uses editable sibling `../pinrag` via uv):

```bash
uv sync
uv run pinrag-cli
```

## Usage

- Plain text at the prompt runs a RAG **query** (with a short **Rich Live** progress panel).
- **Slash commands:** `/add`, `/list`, `/remove`, `/tag`, `/ask`, `/switch`, `/history`, `/status`, `/help`, `/exit`
- Input line history: `~/.pinrag_cli_history`
- Conversation turns (query + answer + sources): `~/.pinrag-cli/history/<session-id>.json`

### Direct (in-process PinRAG)

```bash
pinrag-cli [--persist-dir DIR] [--collection NAME]
```

### HTTP MCP (separate `pinrag server` process)

Terminal 1:

```bash
pinrag server --host 127.0.0.1 --port 8765
```

Terminal 2:

```bash
pinrag-cli --server http://127.0.0.1:8765/mcp [--collection NAME]
```

Use `/switch` to list Chroma collections or `/switch <name>` to target a collection (passed through to MCP tools).

### Live HTTP test (optional)

With a running server:

```bash
PINRAG_CLI_HTTP_ITEST=1 PINRAG_CLI_HTTP_URL=http://127.0.0.1:8765/mcp uv run pytest tests/test_mcp_http_optional.py -v
```

If the server is not up, the test **skips** instead of failing.
