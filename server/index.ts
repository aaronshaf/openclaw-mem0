import { Config, Effect, Schema, ConfigProvider, pipe } from "effect"
import type {
  SearchResponse,
  AddResponse,
  AddResult,
  DeleteResponse,
  HealthResponse,
  Memory,
  SearchResult,
} from "../shared/types"

// --- Config ---

const AppConfig = Config.all({
  qdrantUrl: Config.withDefault(Config.string("QDRANT_URL"), "http://localhost:6333"),
  ollamaUrl: Config.withDefault(Config.string("OLLAMA_URL"), "http://localhost:11434"),
  llmModel: Config.withDefault(Config.string("OLLAMA_LLM_MODEL"), "qwen2.5:3b"),
  embedModel: Config.withDefault(Config.string("OLLAMA_EMBED_MODEL"), "nomic-embed-text"),
  embedDims: Config.withDefault(Config.integer("EMBED_DIMS"), 768),
  collectionName: Config.withDefault(Config.string("COLLECTION_NAME"), "memories"),
  port: Config.withDefault(Config.integer("PORT"), 7890),
})

// --- Tagged errors ---

class QdrantError extends Schema.TaggedError<QdrantError>()("QdrantError", {
  message: Schema.String,
}) {}

class OllamaError extends Schema.TaggedError<OllamaError>()("OllamaError", {
  message: Schema.String,
}) {}

class ValidationError extends Schema.TaggedError<ValidationError>()("ValidationError", {
  message: Schema.String,
}) {}

// --- Request schemas ---

const AddRequestSchema = Schema.Struct({
  messages: Schema.String,
  user_id: Schema.String,
  agent_id: Schema.optional(Schema.String),
})

const SearchRequestSchema = Schema.Struct({
  query: Schema.String,
  user_id: Schema.String,
  limit: Schema.optional(Schema.Number),
})

// --- External API response schemas ---

const OllamaEmbedResponseSchema = Schema.Struct({
  embedding: Schema.Array(Schema.Number),
})

const OllamaGenerateResponseSchema = Schema.Struct({
  response: Schema.String,
})

const QdrantSearchResultSchema = Schema.Struct({
  id: Schema.Union(Schema.String, Schema.Number),
  score: Schema.Number,
  payload: Schema.Record({ key: Schema.String, value: Schema.Unknown }),
})

const QdrantSearchResponseSchema = Schema.Struct({
  result: Schema.Array(QdrantSearchResultSchema),
})

const QdrantScrollResponseSchema = Schema.Struct({
  result: Schema.Struct({
    points: Schema.Array(
      Schema.Struct({
        id: Schema.Union(Schema.String, Schema.Number),
        payload: Schema.Record({ key: Schema.String, value: Schema.Unknown }),
      })
    ),
  }),
})

// --- JSON response helpers ---

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function errorResponse(msg: string, status = 400): Response {
  return json({ error: msg }, status)
}

// --- Effect-based fetch helpers ---

function fetchJson(url: string, init?: RequestInit): Effect.Effect<unknown, QdrantError, never> {
  return Effect.tryPromise({
    try: async () => {
      const res = await fetch(url, init)
      if (!res.ok) {
        const body = await res.text()
        throw new Error(`HTTP ${res.status}: ${body}`)
      }
      return res.json() as Promise<unknown>
    },
    catch: (e) => new QdrantError({ message: String(e) }),
  })
}

function fetchJsonOllama(url: string, init?: RequestInit): Effect.Effect<unknown, OllamaError, never> {
  return Effect.tryPromise({
    try: async () => {
      const res = await fetch(url, init)
      if (!res.ok) {
        const body = await res.text()
        throw new Error(`HTTP ${res.status}: ${body}`)
      }
      return res.json() as Promise<unknown>
    },
    catch: (e) => new OllamaError({ message: String(e) }),
  })
}

// --- Config-dependent helpers ---

function makeHelpers(cfg: {
  qdrantUrl: string
  ollamaUrl: string
  llmModel: string
  embedModel: string
  embedDims: number
  collectionName: string
}) {
  const { qdrantUrl, ollamaUrl, llmModel, embedModel, embedDims, collectionName } = cfg

  // Qdrant helpers

  function ensureCollection(): Effect.Effect<void, QdrantError, never> {
    return pipe(
      Effect.tryPromise({
        try: () => fetch(`${qdrantUrl}/collections/${collectionName}`),
        catch: (e) => new QdrantError({ message: String(e) }),
      }),
      Effect.flatMap((checkRes) => {
        if (checkRes.ok) {
          console.log(`[qdrant] Collection '${collectionName}' already exists`)
          return Effect.void
        }
        return pipe(
          fetchJson(`${qdrantUrl}/collections/${collectionName}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              vectors: { size: embedDims, distance: "Cosine" },
            }),
          }),
          Effect.map(() => {
            console.log(`[qdrant] Created collection '${collectionName}' with ${embedDims} dims`)
          })
        )
      })
    )
  }

  function upsertPoint(
    id: string,
    vector: number[],
    payload: Record<string, unknown>
  ): Effect.Effect<void, QdrantError, never> {
    return pipe(
      fetchJson(`${qdrantUrl}/collections/${collectionName}/points`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points: [{ id, vector, payload }] }),
      }),
      Effect.map(() => undefined)
    )
  }

  function searchPoints(
    vector: number[],
    userId: string,
    limit: number
  ): Effect.Effect<SearchResult[], QdrantError, never> {
    return pipe(
      fetchJson(`${qdrantUrl}/collections/${collectionName}/points/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          vector,
          limit,
          with_payload: true,
          filter: { must: [{ key: "user_id", match: { value: userId } }] },
        }),
      }),
      Effect.flatMap((data) =>
        pipe(
          Schema.decodeUnknown(QdrantSearchResponseSchema)(data),
          Effect.mapError((e) => new QdrantError({ message: `Response decode failed: ${String(e)}` }))
        )
      ),
      Effect.map((decoded) =>
        decoded.result.map((r) => ({
          id: String(r.id),
          memory: String(r.payload["memory"] ?? ""),
          score: r.score,
          user_id: r.payload["user_id"] ? String(r.payload["user_id"]) : undefined,
          agent_id: r.payload["agent_id"] ? String(r.payload["agent_id"]) : undefined,
        }))
      )
    )
  }

  function deletePoint(id: string): Effect.Effect<void, QdrantError, never> {
    return pipe(
      fetchJson(`${qdrantUrl}/collections/${collectionName}/points/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points: [id] }),
      }),
      Effect.map(() => undefined)
    )
  }

  function scrollPoints(userId: string): Effect.Effect<Memory[], QdrantError, never> {
    return pipe(
      fetchJson(`${qdrantUrl}/collections/${collectionName}/points/scroll`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          limit: 1000,
          with_payload: true,
          with_vector: false,
          filter: { must: [{ key: "user_id", match: { value: userId } }] },
        }),
      }),
      Effect.flatMap((data) =>
        pipe(
          Schema.decodeUnknown(QdrantScrollResponseSchema)(data),
          Effect.mapError((e) => new QdrantError({ message: `Response decode failed: ${String(e)}` }))
        )
      ),
      Effect.map((decoded) =>
        decoded.result.points.map((p) => ({
          id: String(p.id),
          memory: String(p.payload["memory"] ?? ""),
          user_id: p.payload["user_id"] ? String(p.payload["user_id"]) : undefined,
          agent_id: p.payload["agent_id"] ? String(p.payload["agent_id"]) : undefined,
          created_at: p.payload["created_at"] ? String(p.payload["created_at"]) : undefined,
        }))
      )
    )
  }

  // Ollama helpers

  function embed(text: string): Effect.Effect<number[], OllamaError, never> {
    return pipe(
      fetchJsonOllama(`${ollamaUrl}/api/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: embedModel, prompt: text }),
      }),
      Effect.flatMap((data) =>
        pipe(
          Schema.decodeUnknown(OllamaEmbedResponseSchema)(data),
          Effect.mapError((e) => new OllamaError({ message: `Embed response decode failed: ${String(e)}` }))
        )
      ),
      Effect.map((decoded) => [...decoded.embedding])
    )
  }

  function extractFacts(messages: string): Effect.Effect<string[], never, never> {
    const systemPrompt =
      "You are a memory extraction assistant. Extract discrete, reusable facts and preferences from conversations. Return only a JSON array of strings."
    const userPrompt = `Extract discrete, reusable facts and preferences from this conversation. Return a JSON array of strings, each a short factual statement. Focus on facts about the user, their preferences, skills, and context. Return [] if nothing memorable.

Conversation:
${messages}`

    const llmEffect = pipe(
      fetchJsonOllama(`${ollamaUrl}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: llmModel,
          system: systemPrompt,
          prompt: userPrompt,
          format: "json",
          stream: false,
        }),
      }),
      Effect.flatMap((data) =>
        pipe(
          Schema.decodeUnknown(OllamaGenerateResponseSchema)(data),
          Effect.mapError((e) => new OllamaError({ message: `Generate response decode failed: ${String(e)}` }))
        )
      ),
      Effect.flatMap((decoded) =>
        Effect.try({
          try: () => {
            const parsed = JSON.parse(decoded.response)
            if (Array.isArray(parsed)) {
              return parsed.filter((f): f is string => typeof f === "string" && f.trim().length > 0)
            }
            const firstArray = Object.values(parsed as Record<string, unknown>).find(Array.isArray)
            if (firstArray) {
              return (firstArray as unknown[]).filter((f): f is string => typeof f === "string")
            }
            console.warn("[llm] Unexpected JSON shape, falling back to raw message")
            return [messages.slice(0, 500)]
          },
          catch: (e) => new OllamaError({ message: `JSON parse failed: ${String(e)}` }),
        })
      )
    )

    return pipe(
      llmEffect,
      Effect.catchAll((e) => {
        console.warn(`[llm] extractFacts failed (${e.message}), storing raw message`)
        return Effect.succeed([messages.slice(0, 500)])
      })
    )
  }

  // --- Route handlers (all return Effect<Response, never, never>) ---

  function handleAdd(req: Request): Effect.Effect<Response, never, never> {
    const parseBody = pipe(
      Effect.tryPromise({
        try: () => req.json() as Promise<unknown>,
        catch: () => new ValidationError({ message: "Invalid JSON body" }),
      }),
      Effect.flatMap((body) =>
        pipe(
          Schema.decodeUnknown(AddRequestSchema)(body),
          Effect.mapError((e) => new ValidationError({ message: `Validation failed: ${String(e)}` }))
        )
      )
    )

    const handler = pipe(
      parseBody,
      Effect.flatMap((body) =>
        pipe(
          extractFacts(body.messages),
          Effect.flatMap((facts) => {
            if (facts.length === 0) {
              return Effect.succeed(json({ results: [] } satisfies AddResponse))
            }

            const now = new Date().toISOString()

            return pipe(
              Effect.forEach(
                facts,
                (fact) =>
                  pipe(
                    embed(fact),
                    Effect.flatMap((vector) => {
                      const id = crypto.randomUUID()
                      return pipe(
                        upsertPoint(id, vector, {
                          memory: fact,
                          user_id: body.user_id,
                          agent_id: body.agent_id ?? body.user_id,
                          created_at: now,
                        }),
                        Effect.map((): AddResult => {
                          console.log(`[add] Stored memory ${id}: "${fact.slice(0, 60)}..."`)
                          return { id, memory: fact, event: "ADD" }
                        })
                      )
                    }),
                    Effect.catchAll((e) => {
                      console.error(`[add] Failed to store fact "${fact.slice(0, 40)}": ${e.message}`)
                      return Effect.succeed(null as AddResult | null)
                    })
                  ),
                { concurrency: "inherit" }
              ),
              Effect.map((results) => {
                const stored = results.filter((r): r is AddResult => r !== null)
                return json({ results: stored } satisfies AddResponse)
              })
            )
          })
        )
      )
    )

    return pipe(
      handler,
      Effect.catchAll((e) => {
        if (e._tag === "ValidationError") {
          return Effect.succeed(errorResponse(e.message, 400))
        }
        console.error(`[add] Unhandled error: ${e.message}`)
        return Effect.succeed(errorResponse("Internal server error", 500))
      })
    )
  }

  function handleSearch(req: Request): Effect.Effect<Response, never, never> {
    const parseBody = pipe(
      Effect.tryPromise({
        try: () => req.json() as Promise<unknown>,
        catch: () => new ValidationError({ message: "Invalid JSON body" }),
      }),
      Effect.flatMap((body) =>
        pipe(
          Schema.decodeUnknown(SearchRequestSchema)(body),
          Effect.mapError((e) => new ValidationError({ message: `Validation failed: ${String(e)}` }))
        )
      )
    )

    const handler = pipe(
      parseBody,
      Effect.flatMap((body) => {
        const limit = body.limit ?? 5
        return pipe(
          embed(body.query),
          Effect.mapError((e): ValidationError | OllamaError | QdrantError => e),
          Effect.flatMap((vector) => searchPoints(vector, body.user_id, limit)),
          Effect.map((results) => {
            console.log(
              `[search] query="${body.query.slice(0, 40)}" user=${body.user_id} -> ${results.length} results`
            )
            return json({ results } satisfies SearchResponse)
          })
        )
      })
    )

    return pipe(
      handler,
      Effect.catchAll((e) => {
        if (e._tag === "ValidationError") {
          return Effect.succeed(errorResponse(e.message, 400))
        }
        console.error(`[search] error: ${e.message}`)
        return Effect.succeed(errorResponse("Search failed", 500))
      })
    )
  }

  function handleDelete(id: string): Effect.Effect<Response, never, never> {
    return pipe(
      deletePoint(id),
      Effect.map(() => {
        console.log(`[delete] Deleted memory ${id}`)
        return json({ success: true } satisfies DeleteResponse)
      }),
      Effect.catchAll((e) => {
        console.error(`[delete] error: ${e.message}`)
        return Effect.succeed(errorResponse("Delete failed", 500))
      })
    )
  }

  function handleListMemories(req: Request): Effect.Effect<Response, never, never> {
    const url = new URL(req.url)
    const userId = url.searchParams.get("user_id")
    if (!userId) return Effect.succeed(errorResponse("Missing query param: user_id"))

    return pipe(
      scrollPoints(userId),
      Effect.map((memories) => {
        console.log(`[memories] Listed ${memories.length} memories for user=${userId}`)
        return json({ memories })
      }),
      Effect.catchAll((e) => {
        console.error(`[memories] error: ${e.message}`)
        return Effect.succeed(errorResponse("Failed to list memories", 500))
      })
    )
  }

  return { ensureCollection, handleAdd, handleSearch, handleDelete, handleListMemories }
}

// --- Server bootstrap ---

const program = pipe(
  ConfigProvider.fromEnv().load(AppConfig),
  Effect.flatMap((cfg) => {
    const helpers = makeHelpers(cfg)

    return pipe(
      helpers.ensureCollection(),
      Effect.catchAll((e) => {
        console.error(`[init] Failed to ensure collection: ${e.message}`)
        console.warn("[init] Server will start anyway — Qdrant may not be ready yet")
        return Effect.void
      }),
      Effect.map(() => {
        const server = Bun.serve({
          port: cfg.port,
          async fetch(req: Request) {
            const url = new URL(req.url)
            const method = req.method.toUpperCase()
            const path = url.pathname

            console.log(`[http] ${method} ${path}`)

            if (method === "GET" && path === "/health") {
              return json({ status: "ok" } satisfies HealthResponse)
            }

            if (method === "POST" && path === "/add") {
              return Effect.runPromise(helpers.handleAdd(req))
            }

            if (method === "POST" && path === "/search") {
              return Effect.runPromise(helpers.handleSearch(req))
            }

            if (method === "DELETE" && path.startsWith("/memories/")) {
              const id = path.slice("/memories/".length)
              if (!id) return errorResponse("Missing memory ID")
              return Effect.runPromise(helpers.handleDelete(id))
            }

            if (method === "GET" && path === "/memories") {
              return Effect.runPromise(helpers.handleListMemories(req))
            }

            return errorResponse("Not found", 404)
          },
          error(err: Error) {
            console.error(`[server] Unhandled error: ${String(err)}`)
            return new Response("Internal server error", { status: 500 })
          },
        })

        console.log(`[server] openclaw-mem0-server listening on port ${cfg.port}`)
        console.log(`[server] Qdrant: ${cfg.qdrantUrl}, collection: ${cfg.collectionName}`)
        console.log(`[server] Ollama LLM: ${cfg.llmModel}, embed: ${cfg.embedModel}`)

        return server
      })
    )
  })
)

Effect.runPromise(program).catch((e) => {
  console.error("[fatal] Failed to start server:", e)
  process.exit(1)
})
