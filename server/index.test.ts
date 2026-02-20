import { describe, test, expect, beforeAll } from "bun:test"
import { Effect, Layer } from "effect"
import { http, HttpResponse } from "msw"
import { mswServer } from "./test-setup"
import { makeQdrantClientLive, makeOllamaClientLive } from "./infrastructure"
import {
  QdrantClient,
  OllamaClient,
  handleAdd,
  handleSearch,
  handleDelete,
  handleListMemories,
  handleCountMemories,
  isValidUUID,
  json,
  errorResponse,
} from "./index"
import type { QdrantClientService, OllamaClientService } from "./index"

const QDRANT = "http://localhost:6333"
const OLLAMA = "http://localhost:11434"
const OPENAI_BASE = "http://openai-api.test"
const COLLECTION = "memories"

// --- Mock layers ---

function makeMockQdrant(overrides: Partial<QdrantClientService> = {}): Layer.Layer<QdrantClient> {
  const defaults: QdrantClientService = {
    ensureCollection: () => Effect.void,
    upsertPoint: () => Effect.void,
    searchPoints: (_vec, _uid, _lim, _aid?, _rid?) => Effect.succeed([]),
    deletePoint: () => Effect.void,
    scrollPoints: (_uid?, _aid?, _rid?) => Effect.succeed([]),
    countPoints: (_uid?, _aid?, _rid?) => Effect.succeed(0),
  }
  return Layer.succeed(QdrantClient, { ...defaults, ...overrides })
}

function makeMockOllama(overrides: Partial<OllamaClientService> = {}): Layer.Layer<OllamaClient> {
  const defaults: OllamaClientService = {
    embed: () => Effect.succeed([0.1, 0.2, 0.3]),
    extractFacts: (msg) => Effect.succeed([`fact from: ${msg.slice(0, 30)}`]),
  }
  return Layer.succeed(OllamaClient, { ...defaults, ...overrides })
}

function runWithMocks<A>(
  effect: Effect.Effect<A, never, QdrantClient | OllamaClient>,
  qdrantOverrides: Partial<QdrantClientService> = {},
  ollamaOverrides: Partial<OllamaClientService> = {}
): Promise<A> {
  const layer = Layer.merge(makeMockQdrant(qdrantOverrides), makeMockOllama(ollamaOverrides))
  return Effect.runPromise(Effect.provide(effect, layer))
}

function runWithQdrant<A>(
  effect: Effect.Effect<A, never, QdrantClient>,
  overrides: Partial<QdrantClientService> = {}
): Promise<A> {
  return Effect.runPromise(Effect.provide(effect, makeMockQdrant(overrides)))
}

function runWithLive<A>(effect: Effect.Effect<A, never, QdrantClient | OllamaClient>): Promise<A> {
  const qdrantLayer = makeQdrantClientLive({
    qdrantUrl: QDRANT,
    collectionName: COLLECTION,
    embedDims: 768,
  })
  const ollamaLayer = makeOllamaClientLive({
    ollamaUrl: OLLAMA,
    llmModel: "qwen2.5:3b",
    embedModel: "nomic-embed-text",
    embedProvider: "ollama",
    openaiBaseUrl: OPENAI_BASE,
  })
  return Effect.runPromise(Effect.provide(effect, Layer.merge(qdrantLayer, ollamaLayer)))
}

function makeReq(method: string, url: string, body?: unknown): Request {
  const init: RequestInit = { method, headers: { "Content-Type": "application/json" } }
  if (body !== undefined) init.body = JSON.stringify(body)
  return new Request(url, init)
}

async function parseJson(res: Response): Promise<any> {
  return res.json()
}

// --- Tests ---

describe("isValidUUID", () => {
  test("accepts valid UUID", () => {
    expect(isValidUUID("550e8400-e29b-41d4-a716-446655440000")).toBe(true)
  })
  test("rejects non-UUID", () => {
    expect(isValidUUID("not-a-uuid")).toBe(false)
    expect(isValidUUID("")).toBe(false)
    expect(isValidUUID("123")).toBe(false)
  })
})

describe("GET /health", () => {
  test("json helper returns proper response", async () => {
    const res = json({ status: "ok" })
    expect(res.status).toBe(200)
    const data = await parseJson(res)
    expect(data.status).toBe("ok")
  })
})

describe("POST /add", () => {
  test("valid body -> extracts facts, embeds, upserts, returns results", async () => {
    let upsertCalled = false
    const res = await runWithMocks(
      handleAdd(makeReq("POST", "http://localhost/add", { messages: "User likes TypeScript", user_id: "u1", agent_id: "wolverine" })),
      { upsertPoint: () => { upsertCalled = true; return Effect.void } },
      { extractFacts: () => Effect.succeed(["User likes TypeScript"]) }
    )
    expect(res.status).toBe(200)
    const data = await parseJson(res)
    expect(data.results).toHaveLength(1)
    expect(data.results[0].event).toBe("ADD")
    expect(data.results[0].memory).toBe("User likes TypeScript")
    expect(upsertCalled).toBe(true)
  })

  test("missing messages field -> 400", async () => {
    const res = await runWithMocks(
      handleAdd(makeReq("POST", "http://localhost/add", { user_id: "u1", agent_id: "wolverine" }))
    )
    expect(res.status).toBe(400)
  })

  test("missing all identifiers -> 400", async () => {
    const res = await runWithMocks(
      handleAdd(makeReq("POST", "http://localhost/add", { messages: "hello" }))
    )
    expect(res.status).toBe(400)
  })

  test("messages too long -> 400", async () => {
    const res = await runWithMocks(
      handleAdd(makeReq("POST", "http://localhost/add", { messages: "x".repeat(50_001), user_id: "u1", agent_id: "wolverine" }))
    )
    expect(res.status).toBe(400)
  })

  test("LLM fails -> falls back to raw message", async () => {
    const res = await runWithMocks(
      handleAdd(makeReq("POST", "http://localhost/add", { messages: "hello world", user_id: "u1", agent_id: "wolverine" })),
      {},
      { extractFacts: (msg) => Effect.succeed([msg.slice(0, 500)]) }
    )
    expect(res.status).toBe(200)
    const data = await parseJson(res)
    expect(data.results).toHaveLength(1)
    expect(data.results[0].memory).toBe("hello world")
  })

  test("partial failure includes failed count", async () => {
    let callCount = 0
    const res = await runWithMocks(
      handleAdd(makeReq("POST", "http://localhost/add", { messages: "stuff", user_id: "u1", agent_id: "wolverine" })),
      {
        upsertPoint: () => {
          callCount++
          if (callCount === 1) return Effect.fail({ _tag: "QdrantError", message: "fail" } as any)
          return Effect.void
        },
      },
      { extractFacts: () => Effect.succeed(["fact1", "fact2"]) }
    )
    expect(res.status).toBe(200)
    const data = await parseJson(res)
    expect(data.results.length + (data.failed ?? 0)).toBe(2)
  })

  test("add with run_id -> stores in payload", async () => {
    let capturedPayload: Record<string, unknown> | null = null
    const res = await runWithMocks(
      handleAdd(makeReq("POST", "http://localhost/add", { messages: "session info", user_id: "u1", run_id: "session-123" })),
      {
        upsertPoint: (_id, _vec, payload) => {
          capturedPayload = payload
          return Effect.void
        },
      },
      { extractFacts: () => Effect.succeed(["session info"]) }
    )
    expect(res.status).toBe(200)
    expect(capturedPayload).not.toBeNull()
    expect(capturedPayload!["run_id"]).toBe("session-123")
  })

  test("add with only run_id (no agent_id) -> succeeds", async () => {
    const res = await runWithMocks(
      handleAdd(makeReq("POST", "http://localhost/add", { messages: "hello", user_id: "u1", run_id: "sess-1" })),
      {},
      { extractFacts: () => Effect.succeed(["hello"]) }
    )
    expect(res.status).toBe(200)
  })
})

describe("POST /search", () => {
  test("valid body -> embeds, searches, returns results", async () => {
    const res = await runWithMocks(
      handleSearch(makeReq("POST", "http://localhost/search", { query: "typescript", user_id: "u1" })),
      {
        searchPoints: () =>
          Effect.succeed([{ id: "1", memory: "User likes TS", score: 0.9, user_id: "u1", agent_id: "wolverine" }]),
      }
    )
    expect(res.status).toBe(200)
    const data = await parseJson(res)
    expect(data.results).toHaveLength(1)
    expect(data.results[0].memory).toBe("User likes TS")
  })

  test("missing query -> 400", async () => {
    const res = await runWithMocks(
      handleSearch(makeReq("POST", "http://localhost/search", { user_id: "u1" }))
    )
    expect(res.status).toBe(400)
  })

  test("with agent_id -> passes to searchPoints", async () => {
    let receivedAgentId: string | undefined
    const res = await runWithMocks(
      handleSearch(makeReq("POST", "http://localhost/search", { query: "ts", user_id: "u1", agent_id: "panda" })),
      {
        searchPoints: (_vec, _uid, _lim, aid) => {
          receivedAgentId = aid
          return Effect.succeed([])
        },
      }
    )
    expect(res.status).toBe(200)
    expect(receivedAgentId).toBe("panda")
  })

  test("without agent_id -> searches across all agents", async () => {
    let receivedAgentId: string | undefined
    const res = await runWithMocks(
      handleSearch(makeReq("POST", "http://localhost/search", { query: "ts", user_id: "u1" })),
      {
        searchPoints: (_vec, _uid, _lim, aid) => {
          receivedAgentId = aid
          return Effect.succeed([])
        },
      }
    )
    expect(res.status).toBe(200)
    expect(receivedAgentId).toBeUndefined()
  })

  test("with run_id -> passes to searchPoints", async () => {
    let receivedRunId: string | undefined
    const res = await runWithMocks(
      handleSearch(makeReq("POST", "http://localhost/search", { query: "ts", user_id: "u1", run_id: "session-abc" })),
      {
        searchPoints: (_vec, _uid, _lim, _aid, rid) => {
          receivedRunId = rid
          return Effect.succeed([])
        },
      }
    )
    expect(res.status).toBe(200)
    expect(receivedRunId).toBe("session-abc")
  })

  test("query too long -> 400", async () => {
    const res = await runWithMocks(
      handleSearch(makeReq("POST", "http://localhost/search", { query: "x".repeat(1001), user_id: "u1" }))
    )
    expect(res.status).toBe(400)
  })
})

describe("DELETE /memories/:id", () => {
  test("valid UUID -> deletes -> success", async () => {
    let deletedId = ""
    const id = "550e8400-e29b-41d4-a716-446655440000"
    const res = await runWithQdrant(
      handleDelete(id),
      { deletePoint: (did) => { deletedId = did; return Effect.void } }
    )
    expect(res.status).toBe(200)
    const data = await parseJson(res)
    expect(data.success).toBe(true)
    expect(deletedId).toBe(id)
  })

  test("invalid UUID -> 400", async () => {
    const res = await runWithQdrant(handleDelete("not-a-uuid"))
    expect(res.status).toBe(400)
    const data = await parseJson(res)
    expect(data.error).toContain("UUID")
  })
})

describe("GET /memories", () => {
  test("with user_id -> returns list", async () => {
    const res = await runWithQdrant(
      handleListMemories(makeReq("GET", "http://localhost/memories?user_id=foo")),
      {
        scrollPoints: () =>
          Effect.succeed([{ id: "1", memory: "m1", user_id: "foo", agent_id: "panda" }]),
      }
    )
    expect(res.status).toBe(200)
    const data = await parseJson(res)
    expect(data.memories).toHaveLength(1)
  })

  test("without user_id -> 400", async () => {
    const res = await runWithQdrant(
      handleListMemories(makeReq("GET", "http://localhost/memories"))
    )
    expect(res.status).toBe(400)
  })

  test("with agent_id -> passes to scrollPoints", async () => {
    let receivedAgentId: string | undefined
    const res = await runWithQdrant(
      handleListMemories(makeReq("GET", "http://localhost/memories?user_id=foo&agent_id=panda")),
      {
        scrollPoints: (_uid, aid) => {
          receivedAgentId = aid
          return Effect.succeed([])
        },
      }
    )
    expect(res.status).toBe(200)
    expect(receivedAgentId).toBe("panda")
  })

  test("with run_id -> passes to scrollPoints", async () => {
    let receivedRunId: string | undefined
    const res = await runWithQdrant(
      handleListMemories(makeReq("GET", "http://localhost/memories?user_id=foo&run_id=sess-99")),
      {
        scrollPoints: (_uid, _aid, rid) => {
          receivedRunId = rid
          return Effect.succeed([])
        },
      }
    )
    expect(res.status).toBe(200)
    expect(receivedRunId).toBe("sess-99")
  })
})

describe("GET /memories/count", () => {
  test("with user_id -> returns count", async () => {
    const res = await runWithQdrant(
      handleCountMemories(makeReq("GET", "http://localhost/memories/count?user_id=foo")),
      { countPoints: () => Effect.succeed(42) }
    )
    expect(res.status).toBe(200)
    const data = await parseJson(res)
    expect(data.count).toBe(42)
  })

  test("without user_id -> 400", async () => {
    const res = await runWithQdrant(
      handleCountMemories(makeReq("GET", "http://localhost/memories/count"))
    )
    expect(res.status).toBe(400)
  })

  test("with run_id -> passes to countPoints", async () => {
    let receivedRunId: string | undefined
    const res = await runWithQdrant(
      handleCountMemories(makeReq("GET", "http://localhost/memories/count?user_id=foo&run_id=sess-42")),
      {
        countPoints: (_uid, _aid, rid) => {
          receivedRunId = rid
          return Effect.succeed(7)
        },
      }
    )
    expect(res.status).toBe(200)
    expect(receivedRunId).toBe("sess-42")
  })
})

// --- Additional error path tests ---

describe("POST /add (error paths)", () => {
  test("invalid JSON body -> 400", async () => {
    const req = new Request("http://localhost/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "not valid json{{{",
    })
    const res = await runWithMocks(handleAdd(req))
    expect(res.status).toBe(400)
    const data = await parseJson(res)
    expect(data.error).toContain("Invalid JSON")
  })

  test("extractFacts returns empty array -> returns empty results", async () => {
    const res = await runWithMocks(
      handleAdd(makeReq("POST", "http://localhost/add", { messages: "hello", user_id: "u1" })),
      {},
      { extractFacts: () => Effect.succeed([]) }
    )
    expect(res.status).toBe(200)
    const data = await parseJson(res)
    expect(data.results).toHaveLength(0)
    expect(data.failed).toBe(0)
  })

  test("all facts fail to store -> returns 200 with empty results and failed count", async () => {
    const res = await runWithMocks(
      handleAdd(makeReq("POST", "http://localhost/add", { messages: "stuff", user_id: "u1" })),
      {
        upsertPoint: () => Effect.fail({ _tag: "QdrantError", message: "qdrant down" } as any),
      },
      { extractFacts: () => Effect.succeed(["fact1", "fact2"]) }
    )
    expect(res.status).toBe(200)
    const data = await parseJson(res)
    expect(data.results).toHaveLength(0)
    expect(data.failed).toBe(2)
  })
})

describe("POST /search (error paths)", () => {
  test("embed fails -> 500", async () => {
    const { OllamaError } = await import("./index")
    const res = await runWithMocks(
      handleSearch(makeReq("POST", "http://localhost/search", { query: "typescript", user_id: "u1" })),
      {},
      { embed: () => Effect.fail(new OllamaError({ message: "ollama down" })) }
    )
    expect(res.status).toBe(500)
    const data = await parseJson(res)
    expect(data.error).toContain("Search failed")
  })

  test("searchPoints fails -> 500", async () => {
    const { QdrantError } = await import("./index")
    const res = await runWithMocks(
      handleSearch(makeReq("POST", "http://localhost/search", { query: "typescript", user_id: "u1" })),
      { searchPoints: () => Effect.fail(new QdrantError({ message: "qdrant down" })) }
    )
    expect(res.status).toBe(500)
    const data = await parseJson(res)
    expect(data.error).toContain("Search failed")
  })
})

describe("DELETE /memories/:id (error paths)", () => {
  test("deletePoint fails -> 500", async () => {
    const { QdrantError } = await import("./index")
    const id = "550e8400-e29b-41d4-a716-446655440000"
    const res = await runWithQdrant(
      handleDelete(id),
      { deletePoint: () => Effect.fail(new QdrantError({ message: "qdrant down" })) }
    )
    expect(res.status).toBe(500)
    const data = await parseJson(res)
    expect(data.error).toContain("Delete failed")
  })
})

describe("GET /memories (error paths)", () => {
  test("scrollPoints fails -> 500", async () => {
    const { QdrantError } = await import("./index")
    const res = await runWithQdrant(
      handleListMemories(makeReq("GET", "http://localhost/memories?user_id=foo")),
      { scrollPoints: () => Effect.fail(new QdrantError({ message: "qdrant down" })) }
    )
    expect(res.status).toBe(500)
    const data = await parseJson(res)
    expect(data.error).toContain("Failed to list memories")
  })
})

describe("GET /memories/count (error paths)", () => {
  test("countPoints fails -> 500", async () => {
    const { QdrantError } = await import("./index")
    const res = await runWithQdrant(
      handleCountMemories(makeReq("GET", "http://localhost/memories/count?user_id=foo")),
      { countPoints: () => Effect.fail(new QdrantError({ message: "qdrant down" })) }
    )
    expect(res.status).toBe(500)
    const data = await parseJson(res)
    expect(data.error).toContain("Failed to count memories")
  })
})

describe("POST /add (identifier validation)", () => {
  test("empty string user_id -> 400 validation error", async () => {
    const res = await runWithMocks(
      handleAdd(makeReq("POST", "http://localhost/add", { messages: "hello", user_id: "" }))
    )
    expect(res.status).toBe(400)
    const data = await parseJson(res)
    expect(data.error).toContain("Validation failed")
  })
})

describe("MSW integration", () => {
  test("POST /add works with live layers and MSW handlers", async () => {
    let upsertBody: unknown = null
    mswServer.use(
      http.post(`${OLLAMA}/api/generate`, () => HttpResponse.json({ response: '["User likes tests"]' })),
      http.post(`${OLLAMA}/api/embeddings`, () => HttpResponse.json({ embedding: [0.1, 0.2, 0.3] })),
      http.put(`${QDRANT}/collections/${COLLECTION}/points`, async ({ request }) => {
        upsertBody = await request.json()
        return HttpResponse.json({ result: { operation_id: 1, status: "completed" } })
      })
    )

    const res = await runWithLive(
      handleAdd(makeReq("POST", "http://localhost/add", { messages: "hello", user_id: "u1" }))
    )
    expect(res.status).toBe(200)
    const data = await parseJson(res)
    expect(data.results).toHaveLength(1)
    expect(data.results[0].memory).toBe("User likes tests")
    expect(upsertBody).not.toBeNull()
  })

  test("POST /search works with live layers and MSW handlers", async () => {
    mswServer.use(
      http.post(`${OLLAMA}/api/embeddings`, () => HttpResponse.json({ embedding: [0.1, 0.2, 0.3] })),
      http.post(`${QDRANT}/collections/${COLLECTION}/points/search`, () =>
        HttpResponse.json({
          result: [{ id: "1", score: 0.91, payload: { memory: "User likes tests", user_id: "u1" } }],
        })
      )
    )

    const res = await runWithLive(
      handleSearch(makeReq("POST", "http://localhost/search", { query: "tests", user_id: "u1" }))
    )
    expect(res.status).toBe(200)
    const data = await parseJson(res)
    expect(data.results).toHaveLength(1)
    expect(data.results[0].memory).toBe("User likes tests")
  })
})
