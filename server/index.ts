import { Config, Context, Effect, Layer, Schema, pipe } from "effect"
import type {
  SearchResponse,
  AddResponse,
  AddResult,
  DeleteResponse,
  HealthResponse,
  Memory,
  SearchResult,
  CountResponse,
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
  apiKey: Config.option(Config.string("API_KEY")),
})

type AppConfigType = Effect.Effect.Success<typeof AppConfig>

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

// --- Request schemas (with validation constraints) ---

const AddRequestSchema = Schema.Struct({
  messages: Schema.String.pipe(Schema.maxLength(50_000)),
  user_id: Schema.String,
})

const SearchRequestSchema = Schema.Struct({
  query: Schema.String.pipe(Schema.maxLength(1000)),
  user_id: Schema.String,
  limit: Schema.optional(Schema.Number.pipe(Schema.between(1, 100))),
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
    next_page_offset: Schema.optional(Schema.Union(Schema.String, Schema.Number, Schema.Null)),
  }),
})

const QdrantCountResponseSchema = Schema.Struct({
  result: Schema.Struct({
    count: Schema.Number,
  }),
})

// --- UUID validation ---

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

function isValidUUID(s: string): boolean {
  return UUID_RE.test(s)
}

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

// --- Effect-based fetch helper (generic, parameterized by error constructor) ---

function fetchJsonAs<E>(
  url: string,
  init: RequestInit | undefined,
  mkError: (msg: string) => E
): Effect.Effect<unknown, E, never> {
  return Effect.tryPromise({
    try: async () => {
      const res = await fetch(url, init)
      if (!res.ok) {
        const body = await res.text()
        throw new Error(`HTTP ${res.status}: ${body}`)
      }
      return res.json() as Promise<unknown>
    },
    catch: (e) => mkError(String(e)),
  })
}

// --- Effect Service interfaces ---

interface QdrantClientService {
  ensureCollection: () => Effect.Effect<void, QdrantError, never>
  upsertPoint: (id: string, vector: number[], payload: Record<string, unknown>) => Effect.Effect<void, QdrantError, never>
  searchPoints: (vector: number[], userId: string, limit: number) => Effect.Effect<SearchResult[], QdrantError, never>
  deletePoint: (id: string) => Effect.Effect<void, QdrantError, never>
  scrollPoints: (userId: string) => Effect.Effect<Memory[], QdrantError, never>
  countPoints: (userId: string) => Effect.Effect<number, QdrantError, never>
}

interface OllamaClientService {
  embed: (text: string) => Effect.Effect<number[], OllamaError, never>
  extractFacts: (messages: string) => Effect.Effect<string[], never, never>
}

class QdrantClient extends Context.Tag("QdrantClient")<QdrantClient, QdrantClientService>() {}
class OllamaClient extends Context.Tag("OllamaClient")<OllamaClient, OllamaClientService>() {}

// --- Layer factories ---

function makeQdrantClientLive(cfg: {
  qdrantUrl: string
  collectionName: string
  embedDims: number
}): Layer.Layer<QdrantClient, never, never> {
  const { qdrantUrl, collectionName, embedDims } = cfg
  const mkErr = (msg: string) => new QdrantError({ message: msg })

  const impl: QdrantClientService = {
    ensureCollection: () =>
      pipe(
        Effect.tryPromise({
          try: () => fetch(`${qdrantUrl}/collections/${collectionName}`),
          catch: (e) => mkErr(String(e)),
        }),
        Effect.flatMap((checkRes) => {
          if (checkRes.ok) {
            console.log(`[qdrant] Collection '${collectionName}' already exists`)
            return Effect.void
          }
          return pipe(
            fetchJsonAs(`${qdrantUrl}/collections/${collectionName}`, {
              method: "PUT",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ vectors: { size: embedDims, distance: "Cosine" } }),
            }, mkErr),
            Effect.map(() => {
              console.log(`[qdrant] Created collection '${collectionName}' with ${embedDims} dims`)
            })
          )
        })
      ),

    upsertPoint: (id, vector, payload) =>
      pipe(
        fetchJsonAs(`${qdrantUrl}/collections/${collectionName}/points`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ points: [{ id, vector, payload }] }),
        }, mkErr),
        Effect.map(() => undefined)
      ),

    searchPoints: (vector, userId, limit) =>
      pipe(
        fetchJsonAs(`${qdrantUrl}/collections/${collectionName}/points/search`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            vector,
            limit,
            with_payload: true,
            filter: { must: [{ key: "user_id", match: { value: userId } }] },
          }),
        }, mkErr),
        Effect.flatMap((data) =>
          pipe(
            Schema.decodeUnknown(QdrantSearchResponseSchema)(data),
            Effect.mapError((e) => mkErr(`Response decode failed: ${String(e)}`))
          )
        ),
        Effect.map((decoded) =>
          decoded.result.map((r) => ({
            id: String(r.id),
            memory: String(r.payload["memory"] ?? ""),
            score: r.score,
            user_id: r.payload["user_id"] ? String(r.payload["user_id"]) : undefined,
          }))
        )
      ),

    deletePoint: (id) =>
      pipe(
        fetchJsonAs(`${qdrantUrl}/collections/${collectionName}/points/delete`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ points: [id] }),
        }, mkErr),
        Effect.map(() => undefined)
      ),

    scrollPoints: (userId) => {
      const loop = (
        offset: string | number | null,
        acc: Memory[]
      ): Effect.Effect<Memory[], QdrantError, never> =>
        pipe(
          fetchJsonAs(`${qdrantUrl}/collections/${collectionName}/points/scroll`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              limit: 250,
              with_payload: true,
              with_vector: false,
              offset: offset ?? undefined,
              filter: { must: [{ key: "user_id", match: { value: userId } }] },
            }),
          }, mkErr),
          Effect.flatMap((data) =>
            pipe(
              Schema.decodeUnknown(QdrantScrollResponseSchema)(data),
              Effect.mapError((e) => mkErr(`Response decode failed: ${String(e)}`))
            )
          ),
          Effect.flatMap((decoded) => {
            const newPoints: Memory[] = decoded.result.points.map((p) => ({
              id: String(p.id),
              memory: String(p.payload["memory"] ?? ""),
              user_id: p.payload["user_id"] ? String(p.payload["user_id"]) : undefined,
              created_at: p.payload["created_at"] ? String(p.payload["created_at"]) : undefined,
            }))
            const all = [...acc, ...newPoints]
            const nextOffset = decoded.result.next_page_offset
            if (nextOffset == null) return Effect.succeed(all)
            return loop(nextOffset as string | number, all)
          })
        )
      return loop(null, [])
    },

    countPoints: (userId) =>
      pipe(
        fetchJsonAs(`${qdrantUrl}/collections/${collectionName}/points/count`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            filter: { must: [{ key: "user_id", match: { value: userId } }] },
            exact: true,
          }),
        }, mkErr),
        Effect.flatMap((data) =>
          pipe(
            Schema.decodeUnknown(QdrantCountResponseSchema)(data),
            Effect.mapError((e) => mkErr(`Response decode failed: ${String(e)}`))
          )
        ),
        Effect.map((decoded) => decoded.result.count)
      ),
  }

  return Layer.succeed(QdrantClient, impl)
}

function makeOllamaClientLive(cfg: {
  ollamaUrl: string
  llmModel: string
  embedModel: string
}): Layer.Layer<OllamaClient, never, never> {
  const { ollamaUrl, llmModel, embedModel } = cfg
  const mkErr = (msg: string) => new OllamaError({ message: msg })

  const impl: OllamaClientService = {
    embed: (text) =>
      pipe(
        fetchJsonAs(`${ollamaUrl}/api/embeddings`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: embedModel, prompt: text }),
        }, mkErr),
        Effect.flatMap((data) =>
          pipe(
            Schema.decodeUnknown(OllamaEmbedResponseSchema)(data),
            Effect.mapError((e) => mkErr(`Embed response decode failed: ${String(e)}`))
          )
        ),
        Effect.map((decoded) => [...decoded.embedding])
      ),

    extractFacts: (messages) => {
      const systemPrompt =
        "You are a memory extraction assistant. Extract discrete, reusable facts and preferences from conversations. Return only a JSON array of strings."
      const userPrompt = `Extract discrete, reusable facts and preferences from this conversation. Return a JSON array of strings, each a short factual statement. Focus on facts about the user, their preferences, skills, and context. Return [] if nothing memorable.

Conversation:
${messages}`

      const llmEffect = pipe(
        fetchJsonAs(`${ollamaUrl}/api/generate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: llmModel,
            system: systemPrompt,
            prompt: userPrompt,
            format: "json",
            stream: false,
          }),
        }, mkErr),
        Effect.flatMap((data) =>
          pipe(
            Schema.decodeUnknown(OllamaGenerateResponseSchema)(data),
            Effect.mapError((e) => mkErr(`Generate response decode failed: ${String(e)}`))
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
            catch: (e) => mkErr(`JSON parse failed: ${String(e)}`),
          })
        )
      )

      return pipe(
        llmEffect,
        Effect.catchAll((e) => {
          console.warn(`[extractFacts] LLM unreachable, storing raw message as fallback (${e.message})`)
          return Effect.succeed([messages.slice(0, 500)])
        })
      )
    },
  }

  return Layer.succeed(OllamaClient, impl)
}

// --- Route handlers using Effect services ---

function handleAdd(req: Request): Effect.Effect<Response, never, QdrantClient | OllamaClient> {
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
        Effect.flatMap(OllamaClient, (ollama) => ollama.extractFacts(body.messages)),
        Effect.flatMap((facts) => {
          if (facts.length === 0) {
            return Effect.succeed(json({ results: [], failed: 0 } satisfies AddResponse))
          }

          const now = new Date().toISOString()

          return pipe(
            Effect.flatMap(QdrantClient, (qdrant) =>
              Effect.flatMap(OllamaClient, (ollama) =>
                pipe(
                  Effect.forEach(
                    facts,
                    (fact) =>
                      pipe(
                        ollama.embed(fact),
                        Effect.flatMap((vector) => {
                          const id = crypto.randomUUID()
                          return pipe(
                            qdrant.upsertPoint(id, vector, {
                              memory: fact,
                              user_id: body.user_id,
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
                    const failed = results.length - stored.length
                    return json({ results: stored, ...(failed > 0 ? { failed } : {}) } satisfies AddResponse)
                  })
                )
              )
            )
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

function handleSearch(req: Request): Effect.Effect<Response, never, QdrantClient | OllamaClient> {
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
        Effect.flatMap(OllamaClient, (ollama) => ollama.embed(body.query)),
        Effect.mapError((e): ValidationError | OllamaError | QdrantError => e),
        Effect.flatMap((vector) =>
          Effect.flatMap(QdrantClient, (qdrant) => qdrant.searchPoints(vector, body.user_id, limit))
        ),
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

function handleDelete(id: string): Effect.Effect<Response, never, QdrantClient> {
  if (!isValidUUID(id)) {
    return Effect.succeed(errorResponse("Invalid memory ID: must be a valid UUID", 400))
  }

  return pipe(
    Effect.flatMap(QdrantClient, (qdrant) => qdrant.deletePoint(id)),
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

function handleListMemories(req: Request): Effect.Effect<Response, never, QdrantClient> {
  const url = new URL(req.url)
  const userId = url.searchParams.get("user_id")
  if (!userId) return Effect.succeed(errorResponse("Missing query param: user_id"))

  return pipe(
    Effect.flatMap(QdrantClient, (qdrant) => qdrant.scrollPoints(userId)),
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

function handleCountMemories(req: Request): Effect.Effect<Response, never, QdrantClient> {
  const url = new URL(req.url)
  const userId = url.searchParams.get("user_id")
  if (!userId) return Effect.succeed(errorResponse("Missing query param: user_id"))

  return pipe(
    Effect.flatMap(QdrantClient, (qdrant) => qdrant.countPoints(userId)),
    Effect.map((count) => {
      console.log(`[count] user=${userId} -> ${count} memories`)
      return json({ count } satisfies CountResponse)
    }),
    Effect.catchAll((e) => {
      console.error(`[count] error: ${e.message}`)
      return Effect.succeed(errorResponse("Failed to count memories", 500))
    })
  )
}

// --- Exports for testing ---

export {
  AppConfig,
  QdrantClient,
  OllamaClient,
  QdrantError,
  OllamaError,
  ValidationError,
  AddRequestSchema,
  SearchRequestSchema,
  handleAdd,
  handleSearch,
  handleDelete,
  handleListMemories,
  handleCountMemories,
  makeQdrantClientLive,
  makeOllamaClientLive,
  json,
  errorResponse,
  isValidUUID,
}
export type { QdrantClientService, OllamaClientService }

// --- Server bootstrap ---

export let qdrantReady = false

function startServer() {
  const program = pipe(
    AppConfig,
    Effect.flatMap((cfg) => {
      const qdrantLayer = makeQdrantClientLive({
        qdrantUrl: cfg.qdrantUrl,
        collectionName: cfg.collectionName,
        embedDims: cfg.embedDims,
      })
      const ollamaLayer = makeOllamaClientLive({
        ollamaUrl: cfg.ollamaUrl,
        llmModel: cfg.llmModel,
        embedModel: cfg.embedModel,
      })
      const fullLayer = Layer.merge(qdrantLayer, ollamaLayer)

      const apiKeyValue = cfg.apiKey._tag === "Some" ? cfg.apiKey.value : null

      const ensureInit = pipe(
        Effect.flatMap(QdrantClient, (qdrant) => qdrant.ensureCollection()),
        Effect.map(() => {
          qdrantReady = true
        }),
        Effect.catchAll((e) => {
          console.error(`[init] Failed to ensure collection: ${e.message}`)
          console.warn("[init] Server will start anyway — Qdrant may not be ready yet")
          qdrantReady = false
          return Effect.void
        }),
        Effect.provide(fullLayer)
      )

      return pipe(
        ensureInit,
        Effect.map(() => {
          function checkAuth(req: Request): Response | null {
            if (!apiKeyValue) return null
            const auth = req.headers.get("authorization")
            if (!auth || auth !== `Bearer ${apiKeyValue}`) {
              return errorResponse("Unauthorized", 401)
            }
            return null
          }

          const server = Bun.serve({
            port: cfg.port,
            async fetch(req: Request) {
              const url = new URL(req.url)
              const method = req.method.toUpperCase()
              const path = url.pathname

              console.log(`[http] ${method} ${path}`)

              // /health is exempt from auth
              if (method === "GET" && path === "/health") {
                if (!qdrantReady) {
                  return json({ status: "degraded", reason: "qdrant unreachable" } satisfies HealthResponse, 503)
                }
                return json({ status: "ok" } satisfies HealthResponse)
              }

              // Auth check for all other routes
              const authErr = checkAuth(req)
              if (authErr) return authErr

              if (method === "POST" && path === "/add") {
                return Effect.runPromise(Effect.provide(handleAdd(req), fullLayer))
              }

              if (method === "POST" && path === "/search") {
                return Effect.runPromise(Effect.provide(handleSearch(req), fullLayer))
              }

              if (method === "DELETE" && path.startsWith("/memories/")) {
                const id = path.slice("/memories/".length)
                if (!id) return errorResponse("Missing memory ID")
                return Effect.runPromise(Effect.provide(handleDelete(id), fullLayer))
              }

              if (method === "GET" && path === "/memories/count") {
                return Effect.runPromise(Effect.provide(handleCountMemories(req), fullLayer))
              }

              if (method === "GET" && path === "/memories") {
                return Effect.runPromise(Effect.provide(handleListMemories(req), fullLayer))
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
          if (apiKeyValue) console.log(`[server] API key auth enabled`)

          return server
        })
      )
    })
  )

  Effect.runPromise(program).catch((e) => {
    console.error("[fatal] Failed to start server:", e)
    process.exit(1)
  })
}

if (import.meta.main) {
  startServer()
}
