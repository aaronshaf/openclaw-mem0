# openclaw-mem0

A monorepo providing persistent memory for [openclaw](https://openclaw.dev) agents via Qdrant + Ollama.

## Architecture

```
openclaw-mem0/
├── plugin/          # openclaw plugin (TypeScript, Effect)
│   ├── index.ts     # Plugin entry — hooks into agent lifecycle
│   └── ...
├── server/          # Bun HTTP memory server (replaces Python server.py)
│   ├── index.ts     # HTTP server: /add, /search, /memories, /health
│   └── ...
└── shared/
    └── types.ts     # Shared TypeScript API types
```

```
openclaw agent
     │
     │  POST /add, POST /search
     ▼
openclaw-mem0-server  (Bun, port 7890)
     │
     ├── Ollama (LLM: fact extraction, embeddings)
     └── Qdrant (vector storage)
```

## Server Setup

### Prerequisites

- [Bun](https://bun.sh) — `curl -fsSL https://bun.sh/install | bash`
- [Qdrant](https://qdrant.tech) running on port 6333
- [Ollama](https://ollama.ai) running on port 11434 with models:
  - `ollama pull qwen2.5:3b` (LLM for fact extraction)
  - `ollama pull nomic-embed-text` (embeddings, 768 dims)

### Run the server

```bash
cd server
bun install
bun run index.ts
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant base URL |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_LLM_MODEL` | `qwen2.5:3b` | Model for fact extraction |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Model for embeddings |
| `EMBED_DIMS` | `768` | Embedding dimensions |
| `COLLECTION_NAME` | `memories` | Qdrant collection name |
| `PORT` | `7890` | HTTP server port |

### systemd (Linux)

```bash
cp server/openclaw-mem0-server.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now openclaw-mem0-server
```

## Plugin Setup

Install the plugin via openclaw:

```bash
openclaw plugin install ./plugin
```

Configure in your openclaw config:

```json
{
  "plugins": {
    "openclaw-mem0-plugin": {
      "url": "http://localhost:7890",
      "userId": "your-user-id",
      "autoRecall": true,
      "autoCapture": true,
      "topK": 5
    }
  }
}
```

## API Reference

### `GET /health`

Returns server health.

**Response:** `{ "status": "ok" }`

### `POST /add`

Extract facts from a conversation and store them as memories.

**Request:**
```json
{
  "messages": "User: I prefer TypeScript over Python.\nAssistant: Got it!",
  "user_id": "alice",
  "agent_id": "optional-agent-id"
}
```

**Response:**
```json
{
  "results": [
    { "id": "uuid", "memory": "Prefers TypeScript over Python", "event": "ADD" }
  ]
}
```

### `POST /search`

Semantic search over stored memories.

**Request:**
```json
{ "query": "programming language preferences", "user_id": "alice", "limit": 5 }
```

**Response:**
```json
{
  "results": [
    { "id": "uuid", "memory": "Prefers TypeScript over Python", "score": 0.92, "user_id": "alice" }
  ]
}
```

### `GET /memories?user_id=alice`

List all memories for a user.

**Response:**
```json
{
  "memories": [
    { "id": "uuid", "memory": "...", "user_id": "alice", "created_at": "2026-01-01T00:00:00.000Z" }
  ]
}
```

### `DELETE /memories/:id`

Delete a memory by ID.

**Response:** `{ "success": true }`
