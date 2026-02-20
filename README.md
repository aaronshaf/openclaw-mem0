# openclaw-mem0

A monorepo providing persistent memory for [openclaw](https://openclaw.dev) agents via Qdrant + an embedding provider.

Inspired by [mem0ai/mem0 server](https://github.com/mem0ai/mem0/tree/main/server).

## Architecture

```
openclaw-mem0/
├── plugin/          # openclaw plugin (TypeScript, Effect)
│   ├── index.ts     # Plugin entry — hooks into agent lifecycle
│   └── ...
├── server/          # Bun HTTP memory server
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
     ├── LLM (fact extraction) — Ollama or OpenAI-compatible
     └── Qdrant (vector storage)
```

## Memory Dimensions

Memories are tagged with up to three identifiers:

| Dimension | Field | Description |
|-----------|-------|-------------|
| User | `user_id` | Required. Shared across all agents for a user. |
| Agent | `agent_id` | Optional. Scopes memory to a specific bot (e.g. "wolverine"). |
| Session | `run_id` | Optional. Scopes memory to a single conversation run. |

At least one of `user_id` must be non-empty. Search and recall can filter by any combination.

## Server Setup

### Prerequisites

- [Bun](https://bun.sh) — `curl -fsSL https://bun.sh/install | bash`
- [Qdrant](https://qdrant.tech) running on port 6333
- An embedding provider (choose one):
  - **Ollama** (default): `ollama pull nomic-embed-text` + `ollama pull qwen2.5:3b`
  - **OpenAI-compatible**: set `EMBED_PROVIDER=openai` and `OPENAI_API_KEY`

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
| `EMBED_PROVIDER` | `ollama` | Embedding provider: `ollama` or `openai` |
| `OPENAI_API_KEY` | — | API key for OpenAI-compatible embedding provider |
| `OPENAI_BASE_URL` | `https://api.openai.com` | Base URL for OpenAI-compatible provider |
| `COLLECTION_NAME` | `memories` | Qdrant collection name |
| `PORT` | `7890` | HTTP server port |
| `API_KEY` | — | Optional bearer token to protect the server |

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

Configure in your openclaw config (`~/.openclaw/openclaw.json` or mounted config):

```json
{
  plugins: {
    slots: { memory: openclaw-mem0-rest },
    entries: {
      openclaw-mem0-rest: {
        enabled: true,
        config: {
          url: http://localhost:7890,
          userId: inst-bots,
          agentId: my-bot-name,
          autoRecall: true,
          autoCapture: true,
          topK: 5
        }
      }
    }
  }
}
```

### Plugin config options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `url` | ✓ | — | URL of the openclaw-mem0 server |
| `userId` | | `inst-bots` | User ID for memory scoping |
| `agentId` | | — | Agent name (e.g. `wolverine`). Enables agent-scoped memory. |
| `runId` | | — | Static session ID. Leave unset to use per-session IDs from events. |
| `autoRecall` | | `false` | Inject relevant memories before each agent turn |
| `autoCapture` | | `false` | Store conversation snippets after each agent run |
| `topK` | | `5` | Number of memories to inject on recall |

## API Reference

### `GET /health`

Returns server health.

**Response:** `{ status: ok }`

### `POST /add`

Extract facts from a conversation and store them as memories.

**Request:**
```json
{
  messages: User: I prefer TypeScript over Python.nAssistant: Got it!,
  user_id: inst-bots,
  agent_id: wolverine,
  run_id: optional-session-id
}
```

**Response:**
```json
{
  results: [
    { id: uuid, memory: Prefers TypeScript over Python, event: ADD }
  ]
}
```

### `POST /search`

Semantic search over stored memories.

**Request:**
```json
{
  query: programming language preferences,
  user_id: inst-bots,
  agent_id: wolverine,
  run_id: optional-session-id,
  limit: 5
}
```

**Response:**
```json
{
  results: [
    { id: uuid, memory: Prefers TypeScript over Python, score: 0.92, user_id: inst-bots, agent_id: wolverine }
  ]
}
```

### `GET /memories?user_id=inst-bots[&agent_id=wolverine][&run_id=x]`

List all memories, optionally filtered by agent or session.

### `GET /memories/count?user_id=inst-bots[&agent_id=wolverine][&run_id=x]`

Count memories matching the given filters.

### `DELETE /memories/:id`

Delete a memory by ID.

**Response:** `{ success: true }`
