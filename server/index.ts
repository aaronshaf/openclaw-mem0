import type {
  SearchRequest,
  SearchResponse,
  SearchResult,
  AddRequest,
  AddResponse,
  AddResult,
  DeleteResponse,
  HealthResponse,
  Memory,
} from "../shared/types"

// --- Config ---
const QDRANT_URL = process.env.QDRANT_URL ?? "http://localhost:6333"
const OLLAMA_URL = process.env.OLLAMA_URL ?? "http://localhost:11434"
const OLLAMA_LLM_MODEL = process.env.OLLAMA_LLM_MODEL ?? "qwen2.5:3b"
const OLLAMA_EMBED_MODEL = process.env.OLLAMA_EMBED_MODEL ?? "nomic-embed-text"
const EMBED_DIMS = parseInt(process.env.EMBED_DIMS ?? "768", 10)
const COLLECTION_NAME = process.env.COLLECTION_NAME ?? "memories"
const PORT = parseInt(process.env.PORT ?? "7890", 10)

// --- Qdrant helpers ---

async function ensureCollection(): Promise<void> {
  const checkRes = await fetch(`${QDRANT_URL}/collections/${COLLECTION_NAME}`)
  if (checkRes.ok) {
    console.log(`[qdrant] Collection '${COLLECTION_NAME}' already exists`)
    return
  }
  const res = await fetch(`${QDRANT_URL}/collections/${COLLECTION_NAME}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      vectors: {
        size: EMBED_DIMS,
        distance: "Cosine",
      },
    }),
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`Failed to create collection: ${res.status} ${body}`)
  }
  console.log(`[qdrant] Created collection '${COLLECTION_NAME}' with ${EMBED_DIMS} dims`)
}

async function upsertPoint(
  id: string,
  vector: number[],
  payload: Record<string, unknown>
): Promise<void> {
  const res = await fetch(`${QDRANT_URL}/collections/${COLLECTION_NAME}/points`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      points: [{ id, vector, payload }],
    }),
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`Qdrant upsert failed: ${res.status} ${body}`)
  }
}

async function searchPoints(
  vector: number[],
  userId: string,
  limit: number
): Promise<SearchResult[]> {
  const res = await fetch(`${QDRANT_URL}/collections/${COLLECTION_NAME}/points/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      vector,
      limit,
      with_payload: true,
      filter: {
        must: [{ key: "user_id", match: { value: userId } }],
      },
    }),
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`Qdrant search failed: ${res.status} ${body}`)
  }
  const data = (await res.json()) as { result: Array<{ id: string; score: number; payload: Record<string, unknown> }> }
  return data.result.map((r) => ({
    id: String(r.id),
    memory: String(r.payload.memory ?? ""),
    score: r.score,
    user_id: r.payload.user_id ? String(r.payload.user_id) : undefined,
    agent_id: r.payload.agent_id ? String(r.payload.agent_id) : undefined,
  }))
}

async function deletePoint(id: string): Promise<void> {
  const res = await fetch(`${QDRANT_URL}/collections/${COLLECTION_NAME}/points/delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points: [id] }),
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`Qdrant delete failed: ${res.status} ${body}`)
  }
}

async function scrollPoints(userId: string): Promise<Memory[]> {
  const res = await fetch(`${QDRANT_URL}/collections/${COLLECTION_NAME}/points/scroll`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      limit: 1000,
      with_payload: true,
      with_vector: false,
      filter: {
        must: [{ key: "user_id", match: { value: userId } }],
      },
    }),
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`Qdrant scroll failed: ${res.status} ${body}`)
  }
  const data = (await res.json()) as { result: { points: Array<{ id: string; payload: Record<string, unknown> }> } }
  return data.result.points.map((p) => ({
    id: String(p.id),
    memory: String(p.payload.memory ?? ""),
    user_id: p.payload.user_id ? String(p.payload.user_id) : undefined,
    agent_id: p.payload.agent_id ? String(p.payload.agent_id) : undefined,
    created_at: p.payload.created_at ? String(p.payload.created_at) : undefined,
  }))
}

// --- Ollama helpers ---

async function embed(text: string): Promise<number[]> {
  const res = await fetch(`${OLLAMA_URL}/api/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: OLLAMA_EMBED_MODEL, prompt: text }),
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`Ollama embed failed: ${res.status} ${body}`)
  }
  const data = (await res.json()) as { embedding: number[] }
  return data.embedding
}

async function extractFacts(messages: string): Promise<string[]> {
  const systemPrompt =
    "You are a memory extraction assistant. Extract discrete, reusable facts and preferences from conversations. Return only a JSON array of strings."
  const userPrompt = `Extract discrete, reusable facts and preferences from this conversation. Return a JSON array of strings, each a short factual statement. Focus on facts about the user, their preferences, skills, and context. Return [] if nothing memorable.

Conversation:
${messages}`

  const res = await fetch(`${OLLAMA_URL}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: OLLAMA_LLM_MODEL,
      system: systemPrompt,
      prompt: userPrompt,
      format: "json",
      stream: false,
    }),
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`Ollama generate failed: ${res.status} ${body}`)
  }
  const data = (await res.json()) as { response: string }
  try {
    const parsed = JSON.parse(data.response)
    if (Array.isArray(parsed)) {
      return parsed.filter((f) => typeof f === "string" && f.trim().length > 0)
    }
    // Some models wrap in an object
    const firstArray = Object.values(parsed).find(Array.isArray)
    if (firstArray) return (firstArray as unknown[]).filter((f): f is string => typeof f === "string")
    console.warn("[llm] Unexpected JSON shape, falling back to raw message")
    return [messages.slice(0, 500)]
  } catch (e) {
    console.warn(`[llm] JSON parse failed (${String(e)}), storing raw message as single memory`)
    return [messages.slice(0, 500)]
  }
}

// --- Request helpers ---

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function error(msg: string, status = 400): Response {
  return json({ error: msg }, status)
}

// --- Route handlers ---

async function handleAdd(req: Request): Promise<Response> {
  let body: AddRequest
  try {
    body = (await req.json()) as AddRequest
  } catch {
    return error("Invalid JSON body")
  }
  if (!body.messages || !body.user_id) {
    return error("Missing required fields: messages, user_id")
  }

  let facts: string[]
  try {
    facts = await extractFacts(body.messages)
  } catch (e) {
    console.error(`[add] extractFacts error: ${String(e)}`)
    facts = [body.messages.slice(0, 500)]
  }

  if (facts.length === 0) {
    return json({ results: [] } satisfies AddResponse)
  }

  const results: AddResult[] = []
  const now = new Date().toISOString()

  for (const fact of facts) {
    try {
      const vector = await embed(fact)
      const id = crypto.randomUUID()
      await upsertPoint(id, vector, {
        memory: fact,
        user_id: body.user_id,
        agent_id: body.agent_id ?? body.user_id,
        created_at: now,
      })
      results.push({ id, memory: fact, event: "ADD" })
      console.log(`[add] Stored memory ${id}: "${fact.slice(0, 60)}..."`)
    } catch (e) {
      console.error(`[add] Failed to store fact "${fact.slice(0, 40)}": ${String(e)}`)
    }
  }

  return json({ results } satisfies AddResponse)
}

async function handleSearch(req: Request): Promise<Response> {
  let body: SearchRequest
  try {
    body = (await req.json()) as SearchRequest
  } catch {
    return error("Invalid JSON body")
  }
  if (!body.query || !body.user_id) {
    return error("Missing required fields: query, user_id")
  }

  const limit = body.limit ?? 5

  let vector: number[]
  try {
    vector = await embed(body.query)
  } catch (e) {
    console.error(`[search] embed error: ${String(e)}`)
    return error("Failed to embed query", 500)
  }

  let results: SearchResult[]
  try {
    results = await searchPoints(vector, body.user_id, limit)
  } catch (e) {
    console.error(`[search] qdrant error: ${String(e)}`)
    return error("Search failed", 500)
  }

  console.log(`[search] query="${body.query.slice(0, 40)}" user=${body.user_id} -> ${results.length} results`)
  return json({ results } satisfies SearchResponse)
}

async function handleDelete(id: string): Promise<Response> {
  try {
    await deletePoint(id)
    console.log(`[delete] Deleted memory ${id}`)
    return json({ success: true } satisfies DeleteResponse)
  } catch (e) {
    console.error(`[delete] error: ${String(e)}`)
    return error("Delete failed", 500)
  }
}

async function handleListMemories(req: Request): Promise<Response> {
  const url = new URL(req.url)
  const userId = url.searchParams.get("user_id")
  if (!userId) return error("Missing query param: user_id")

  try {
    const memories = await scrollPoints(userId)
    console.log(`[memories] Listed ${memories.length} memories for user=${userId}`)
    return json({ memories })
  } catch (e) {
    console.error(`[memories] error: ${String(e)}`)
    return error("Failed to list memories", 500)
  }
}

// --- Server ---

async function init() {
  try {
    await ensureCollection()
  } catch (e) {
    console.error(`[init] Failed to ensure collection: ${String(e)}`)
    console.warn("[init] Server will start anyway — Qdrant may not be ready yet")
  }
}

await init()

const server = Bun.serve({
  port: PORT,
  async fetch(req: Request) {
    const url = new URL(req.url)
    const method = req.method.toUpperCase()
    const path = url.pathname

    console.log(`[http] ${method} ${path}`)

    if (method === "GET" && path === "/health") {
      return json({ status: "ok" } satisfies HealthResponse)
    }

    if (method === "POST" && path === "/add") {
      return handleAdd(req)
    }

    if (method === "POST" && path === "/search") {
      return handleSearch(req)
    }

    if (method === "DELETE" && path.startsWith("/memories/")) {
      const id = path.slice("/memories/".length)
      if (!id) return error("Missing memory ID")
      return handleDelete(id)
    }

    if (method === "GET" && path === "/memories") {
      return handleListMemories(req)
    }

    return error("Not found", 404)
  },
  error(err: Error) {
    console.error(`[server] Unhandled error: ${String(err)}`)
    return new Response("Internal server error", { status: 500 })
  },
})

console.log(`[server] openclaw-mem0-server listening on port ${PORT}`)
console.log(`[server] Qdrant: ${QDRANT_URL}, collection: ${COLLECTION_NAME}`)
console.log(`[server] Ollama LLM: ${OLLAMA_LLM_MODEL}, embed: ${OLLAMA_EMBED_MODEL}`)
