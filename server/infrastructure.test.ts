import { describe, test, expect } from "bun:test"
import { Effect } from "effect"
import { http, HttpResponse } from "msw"
import { mswServer as server } from "./test-setup"
import { makeQdrantClientLive, makeOllamaClientLive } from "./infrastructure"
import { QdrantClient, OllamaClient } from "./index"

const QDRANT = "http://localhost:6333"
const OLLAMA = "http://localhost:11434"
const OPENAI_BASE = "http://openai-api.test"
const COL = "memories"

// --- Helper factories ---

function runQdrant<A>(eff: Effect.Effect<A, never, QdrantClient>): Promise<A> {
  return Effect.runPromise(
    Effect.provide(eff, makeQdrantClientLive({ qdrantUrl: QDRANT, collectionName: COL, embedDims: 768 }))
  )
}

function runOllama<A>(
  eff: Effect.Effect<A, never, OllamaClient>,
  opts: { embedProvider?: string; openaiApiKey?: string; openaiBaseUrl?: string } = {}
): Promise<A> {
  return Effect.runPromise(
    Effect.provide(
      eff,
      makeOllamaClientLive({
        ollamaUrl: OLLAMA,
        llmModel: "qwen2.5:3b",
        embedModel: "nomic-embed-text",
        embedProvider: opts.embedProvider ?? "ollama",
        openaiApiKey: opts.openaiApiKey,
        openaiBaseUrl: opts.openaiBaseUrl ?? OPENAI_BASE,
      })
    )
  )
}

// --- makeQdrantClientLive tests ---

describe("makeQdrantClientLive — ensureCollection", () => {
  test("collection exists: GET 200 -> no PUT issued", async () => {
    let putCalled = false
    server.use(
      http.get(`${QDRANT}/collections/${COL}`, () => HttpResponse.json({ result: {} }, { status: 200 })),
      http.put(`${QDRANT}/collections/${COL}`, () => {
        putCalled = true
        return HttpResponse.json({}, { status: 200 })
      })
    )
    await runQdrant(Effect.flatMap(QdrantClient, (q) => q.ensureCollection()))
    expect(putCalled).toBe(false)
  })

  test("collection not found: GET 404 -> PUT called", async () => {
    let putCalled = false
    server.use(
      http.get(`${QDRANT}/collections/${COL}`, () => HttpResponse.json({}, { status: 404 })),
      http.put(`${QDRANT}/collections/${COL}`, () => {
        putCalled = true
        return HttpResponse.json({ result: true }, { status: 200 })
      })
    )
    await runQdrant(Effect.flatMap(QdrantClient, (q) => q.ensureCollection()))
    expect(putCalled).toBe(true)
  })

  test("network error -> Left/QdrantError", async () => {
    server.use(
      http.get(`${QDRANT}/collections/${COL}`, () => HttpResponse.error())
    )
    const result = await Effect.runPromise(
      Effect.either(
        Effect.provide(
          Effect.flatMap(QdrantClient, (q) => q.ensureCollection()),
          makeQdrantClientLive({ qdrantUrl: QDRANT, collectionName: COL, embedDims: 768 })
        )
      )
    )
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") {
      expect(result.left._tag).toBe("QdrantError")
    }
  })
})

describe("makeQdrantClientLive — upsertPoint", () => {
  test("success -> resolves void", async () => {
    server.use(
      http.put(`${QDRANT}/collections/${COL}/points`, () => HttpResponse.json({ result: { operation_id: 1, status: "completed" } }))
    )
    await runQdrant(Effect.flatMap(QdrantClient, (q) => q.upsertPoint("id-1", [0.1, 0.2], { memory: "test" })))
  })

  test("HTTP error -> Left/QdrantError", async () => {
    server.use(
      http.put(`${QDRANT}/collections/${COL}/points`, () => HttpResponse.json({ error: "bad" }, { status: 500 }))
    )
    const result = await Effect.runPromise(
      Effect.either(
        Effect.provide(
          Effect.flatMap(QdrantClient, (q) => q.upsertPoint("id-1", [0.1], { memory: "test" })),
          makeQdrantClientLive({ qdrantUrl: QDRANT, collectionName: COL, embedDims: 768 })
        )
      )
    )
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") {
      expect(result.left._tag).toBe("QdrantError")
    }
  })
})

describe("makeQdrantClientLive — searchPoints", () => {
  test("no filters -> returns result array", async () => {
    server.use(
      http.post(`${QDRANT}/collections/${COL}/points/search`, () =>
        HttpResponse.json({
          result: [{ id: "abc-123", score: 0.95, payload: { memory: "test fact", user_id: "u1" } }],
        })
      )
    )
    const results = await runQdrant(Effect.flatMap(QdrantClient, (q) => q.searchPoints([0.1, 0.2], "u1", 5)))
    expect(results).toHaveLength(1)
    expect(results[0].id).toBe("abc-123")
    expect(results[0].score).toBe(0.95)
    expect(results[0].memory).toBe("test fact")
  })

  test("with agentId and runId -> body includes filter.must entries", async () => {
    let capturedBody: unknown = null
    server.use(
      http.post(`${QDRANT}/collections/${COL}/points/search`, async ({ request }) => {
        capturedBody = await request.json()
        return HttpResponse.json({ result: [] })
      })
    )
    await runQdrant(Effect.flatMap(QdrantClient, (q) => q.searchPoints([0.1], "u1", 5, "agent-x", "run-y")))
    expect(capturedBody).not.toBeNull()
    const body = capturedBody as { filter: { must: Array<{ key: string; match: { value: string } }> } }
    const keys = body.filter.must.map((m) => m.key)
    expect(keys).toContain("agent_id")
    expect(keys).toContain("run_id")
  })

  test("bad schema response -> Left", async () => {
    server.use(
      http.post(`${QDRANT}/collections/${COL}/points/search`, () =>
        HttpResponse.json({ wrong_key: [] })
      )
    )
    const result = await Effect.runPromise(
      Effect.either(
        Effect.provide(
          Effect.flatMap(QdrantClient, (q) => q.searchPoints([0.1], "u1", 5)),
          makeQdrantClientLive({ qdrantUrl: QDRANT, collectionName: COL, embedDims: 768 })
        )
      )
    )
    expect(result._tag).toBe("Left")
  })
})

describe("makeQdrantClientLive — deletePoint", () => {
  test("success -> resolves void", async () => {
    server.use(
      http.post(`${QDRANT}/collections/${COL}/points/delete`, () => HttpResponse.json({ result: { operation_id: 1, status: "completed" } }))
    )
    await runQdrant(Effect.flatMap(QdrantClient, (q) => q.deletePoint("some-id")))
  })

  test("HTTP error -> Left/QdrantError", async () => {
    server.use(
      http.post(`${QDRANT}/collections/${COL}/points/delete`, () => HttpResponse.json({ error: "bad" }, { status: 400 }))
    )
    const result = await Effect.runPromise(
      Effect.either(
        Effect.provide(
          Effect.flatMap(QdrantClient, (q) => q.deletePoint("some-id")),
          makeQdrantClientLive({ qdrantUrl: QDRANT, collectionName: COL, embedDims: 768 })
        )
      )
    )
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") {
      expect(result.left._tag).toBe("QdrantError")
    }
  })
})

describe("makeQdrantClientLive — scrollPoints", () => {
  test("single page (next_page_offset: null) -> returns all memories", async () => {
    server.use(
      http.post(`${QDRANT}/collections/${COL}/points/scroll`, () =>
        HttpResponse.json({
          result: {
            points: [
              { id: "1", payload: { memory: "m1", user_id: "u1" } },
              { id: "2", payload: { memory: "m2", user_id: "u1" } },
            ],
            next_page_offset: null,
          },
        })
      )
    )
    const memories = await runQdrant(Effect.flatMap(QdrantClient, (q) => q.scrollPoints("u1")))
    expect(memories).toHaveLength(2)
    expect(memories[0].memory).toBe("m1")
    expect(memories[1].memory).toBe("m2")
  })

  test("two pages -> fetches both pages, returns all memories", async () => {
    let callCount = 0
    server.use(
      http.post(`${QDRANT}/collections/${COL}/points/scroll`, () => {
        callCount++
        if (callCount === 1) {
          return HttpResponse.json({
            result: {
              points: [{ id: "1", payload: { memory: "m1", user_id: "u1" } }],
              next_page_offset: 42,
            },
          })
        }
        return HttpResponse.json({
          result: {
            points: [{ id: "2", payload: { memory: "m2", user_id: "u1" } }],
            next_page_offset: null,
          },
        })
      })
    )
    const memories = await runQdrant(Effect.flatMap(QdrantClient, (q) => q.scrollPoints("u1")))
    expect(callCount).toBe(2)
    expect(memories).toHaveLength(2)
  })
})

describe("makeQdrantClientLive — countPoints", () => {
  test("success -> returns count", async () => {
    server.use(
      http.post(`${QDRANT}/collections/${COL}/points/count`, () =>
        HttpResponse.json({ result: { count: 17 } })
      )
    )
    const count = await runQdrant(Effect.flatMap(QdrantClient, (q) => q.countPoints("u1")))
    expect(count).toBe(17)
  })

  test("bad schema -> Left", async () => {
    server.use(
      http.post(`${QDRANT}/collections/${COL}/points/count`, () =>
        HttpResponse.json({ result: { not_count: "x" } })
      )
    )
    const result = await Effect.runPromise(
      Effect.either(
        Effect.provide(
          Effect.flatMap(QdrantClient, (q) => q.countPoints("u1")),
          makeQdrantClientLive({ qdrantUrl: QDRANT, collectionName: COL, embedDims: 768 })
        )
      )
    )
    expect(result._tag).toBe("Left")
  })
})

// --- makeOllamaClientLive tests ---

describe("makeOllamaClientLive — embed", () => {
  test("ollama provider: POST /api/embeddings with prompt field", async () => {
    let capturedBody: unknown = null
    server.use(
      http.post(`${OLLAMA}/api/embeddings`, async ({ request }) => {
        capturedBody = await request.json()
        return HttpResponse.json({ embedding: [0.1, 0.2, 0.3] })
      })
    )
    const result = await runOllama(Effect.flatMap(OllamaClient, (o) => o.embed("hello")))
    expect(result).toEqual([0.1, 0.2, 0.3])
    const body = capturedBody as { prompt?: string; input?: string }
    expect(body.prompt).toBe("hello")
    expect(body.input).toBeUndefined()
  })

  test("openai provider: POST /v1/embeddings with input field and Auth header", async () => {
    let capturedBody: unknown = null
    let capturedAuth: string | null = null
    server.use(
      http.post(`${OPENAI_BASE}/v1/embeddings`, async ({ request }) => {
        capturedBody = await request.json()
        capturedAuth = request.headers.get("Authorization")
        return HttpResponse.json({ data: [{ embedding: [0.4, 0.5, 0.6] }] })
      })
    )
    const result = await runOllama(
      Effect.flatMap(OllamaClient, (o) => o.embed("hello")),
      { embedProvider: "openai", openaiApiKey: "sk-test-key" }
    )
    expect(result).toEqual([0.4, 0.5, 0.6])
    const body = capturedBody as { input?: string; prompt?: string }
    expect(body.input).toBe("hello")
    expect(body.prompt).toBeUndefined()
    expect(capturedAuth).toBe("Bearer sk-test-key")
  })

  test("embed HTTP error (ollama) -> Left/OllamaError", async () => {
    server.use(
      http.post(`${OLLAMA}/api/embeddings`, () => HttpResponse.json({ error: "not found" }, { status: 404 }))
    )
    const result = await Effect.runPromise(
      Effect.either(
        Effect.provide(
          Effect.flatMap(OllamaClient, (o) => o.embed("hello")),
          makeOllamaClientLive({ ollamaUrl: OLLAMA, llmModel: "qwen2.5:3b", embedModel: "nomic-embed-text", embedProvider: "ollama", openaiBaseUrl: OPENAI_BASE })
        )
      )
    )
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") {
      expect(result.left._tag).toBe("OllamaError")
    }
  })

  test("embed bad schema decode -> Left", async () => {
    server.use(
      http.post(`${OLLAMA}/api/embeddings`, () => HttpResponse.json({ not_embedding: "x" }))
    )
    const result = await Effect.runPromise(
      Effect.either(
        Effect.provide(
          Effect.flatMap(OllamaClient, (o) => o.embed("hello")),
          makeOllamaClientLive({ ollamaUrl: OLLAMA, llmModel: "qwen2.5:3b", embedModel: "nomic-embed-text", embedProvider: "ollama", openaiBaseUrl: OPENAI_BASE })
        )
      )
    )
    expect(result._tag).toBe("Left")
  })
})

describe("makeOllamaClientLive — extractFacts", () => {
  test("JSON array response -> returns array of facts", async () => {
    server.use(
      http.post(`${OLLAMA}/api/generate`, () =>
        HttpResponse.json({ response: '["fact1","fact2"]' })
      )
    )
    const facts = await runOllama(Effect.flatMap(OllamaClient, (o) => o.extractFacts("some messages")))
    expect(facts).toEqual(["fact1", "fact2"])
  })

  test("nested object with facts array -> extracts array", async () => {
    server.use(
      http.post(`${OLLAMA}/api/generate`, () =>
        HttpResponse.json({ response: '{"facts":["fact"]}' })
      )
    )
    const facts = await runOllama(Effect.flatMap(OllamaClient, (o) => o.extractFacts("some messages")))
    expect(facts).toEqual(["fact"])
  })

  test("unexpected JSON shape -> 1-element fallback slice", async () => {
    server.use(
      http.post(`${OLLAMA}/api/generate`, () =>
        HttpResponse.json({ response: '{"key":"not array"}' })
      )
    )
    const facts = await runOllama(Effect.flatMap(OllamaClient, (o) => o.extractFacts("some messages")))
    expect(facts).toHaveLength(1)
    expect(facts[0]).toContain("some messages")
  })

  test("network error -> 1-element fallback with raw message", async () => {
    server.use(
      http.post(`${OLLAMA}/api/generate`, () => HttpResponse.error())
    )
    const facts = await runOllama(Effect.flatMap(OllamaClient, (o) => o.extractFacts("fallback content")))
    expect(facts).toHaveLength(1)
    expect(facts[0]).toContain("fallback content")
  })

  test("non-JSON LLM response -> fallback (JSON parse error caught)", async () => {
    server.use(
      http.post(`${OLLAMA}/api/generate`, () =>
        HttpResponse.json({ response: "not valid json {{{{" })
      )
    )
    const facts = await runOllama(Effect.flatMap(OllamaClient, (o) => o.extractFacts("some messages")))
    expect(facts).toHaveLength(1)
  })
})

describe("makeQdrantClientLive — ensureCollection (error paths)", () => {
  test("GET 404 then PUT fails -> Left/QdrantError", async () => {
    server.use(
      http.get(`${QDRANT}/collections/${COL}`, () => HttpResponse.json({}, { status: 404 })),
      http.put(`${QDRANT}/collections/${COL}`, () => HttpResponse.json({ error: "internal" }, { status: 500 }))
    )
    const result = await Effect.runPromise(
      Effect.either(
        Effect.provide(
          Effect.flatMap(QdrantClient, (q) => q.ensureCollection()),
          makeQdrantClientLive({ qdrantUrl: QDRANT, collectionName: COL, embedDims: 768 })
        )
      )
    )
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") {
      expect(result.left._tag).toBe("QdrantError")
    }
  })
})

describe("makeOllamaClientLive — embed (openai edge cases)", () => {
  test("openai empty data array -> Left/OllamaError", async () => {
    server.use(
      http.post(`${OPENAI_BASE}/v1/embeddings`, () =>
        HttpResponse.json({ data: [] })
      )
    )
    const result = await Effect.runPromise(
      Effect.either(
        Effect.provide(
          Effect.flatMap(OllamaClient, (o) => o.embed("hello")),
          makeOllamaClientLive({ ollamaUrl: OLLAMA, llmModel: "qwen2.5:3b", embedModel: "nomic-embed-text", embedProvider: "openai", openaiBaseUrl: OPENAI_BASE })
        )
      )
    )
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") {
      expect(result.left._tag).toBe("OllamaError")
      expect(result.left.message).toContain("empty")
    }
  })

  test("openai schema decode failure -> Left", async () => {
    server.use(
      http.post(`${OPENAI_BASE}/v1/embeddings`, () =>
        HttpResponse.json({ wrong: "shape" })
      )
    )
    const result = await Effect.runPromise(
      Effect.either(
        Effect.provide(
          Effect.flatMap(OllamaClient, (o) => o.embed("hello")),
          makeOllamaClientLive({ ollamaUrl: OLLAMA, llmModel: "qwen2.5:3b", embedModel: "nomic-embed-text", embedProvider: "openai", openaiBaseUrl: OPENAI_BASE })
        )
      )
    )
    expect(result._tag).toBe("Left")
  })

  test("openai without api key -> no Authorization header sent", async () => {
    let capturedAuth: string | null = "present"
    server.use(
      http.post(`${OPENAI_BASE}/v1/embeddings`, async ({ request }) => {
        capturedAuth = request.headers.get("Authorization")
        return HttpResponse.json({ data: [{ embedding: [0.1] }] })
      })
    )
    await runOllama(
      Effect.flatMap(OllamaClient, (o) => o.embed("hello")),
      { embedProvider: "openai" }
    )
    expect(capturedAuth).toBeNull()
  })
})

describe("makeQdrantClientLive — scrollPoints (pagination body)", () => {
  test("second page request includes offset in body", async () => {
    let secondCallBody: unknown = null
    let callCount = 0
    server.use(
      http.post(`${QDRANT}/collections/${COL}/points/scroll`, async ({ request }) => {
        callCount++
        const body = await request.json()
        if (callCount === 1) {
          return HttpResponse.json({
            result: {
              points: [{ id: "1", payload: { memory: "m1", user_id: "u1" } }],
              next_page_offset: 99,
            },
          })
        }
        secondCallBody = body
        return HttpResponse.json({
          result: { points: [], next_page_offset: null },
        })
      })
    )
    await runQdrant(Effect.flatMap(QdrantClient, (q) => q.scrollPoints("u1")))
    expect(callCount).toBe(2)
    const b = secondCallBody as { offset?: number }
    expect(b.offset).toBe(99)
  })
})
