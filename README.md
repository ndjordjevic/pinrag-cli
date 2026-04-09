# pinrag-cli

Interactive REPL for [PinRAG](https://github.com/ndjordjevic/pinrag): query indexed documents, add paths, list and remove documents.

## Requirements

- Python 3.12+
- Same environment variables as PinRAG (e.g. `OPENROUTER_API_KEY` when using OpenRouter)
- Optional: `PINRAG_PERSIST_DIR`, `PINRAG_COLLECTION_NAME`

## Install

From this repo (uses editable sibling `../pinrag` via uv):

```bash
uv sync
uv run pinrag-cli
```

## Usage

- Plain text at the prompt runs a RAG **query**.
- **Slash commands:** `/add`, `/list`, `/remove`, `/status`, `/help`, `/exit`
- History: `~/.pinrag_cli_history`

```bash
pinrag-cli [--persist-dir DIR] [--collection NAME]
```
