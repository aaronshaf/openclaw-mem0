import { describe, it, expect, mock, beforeEach, afterEach } from "bun:test"
import { decodeConfig, decodeSearchResponse, decodeAddResponse, decodeHealthResponse, decodeCountResponse, mem0Search, mem0Add, mem0Count } from "./index"
import plugin from "./index"

const originalFetch = globalThis.fetch

// ---------------------------------------------------------------------------
// decodeConfig
// ---------------------------------------------------------------------------
describe("decodeConfig", () => {
  it("accepts a valid config with only url (all other fields optional)", () => {
    const cfg = decodeConfig({ url: "http://localhost:8000" })
    expect(cfg.url).toBe("http://localhost:8000")
  })

  it("accepts userId, agentId, runId as optional fields", () => {
    const cfg = decodeConfig({ url: "http://x", userId: "u", agentId: "agent1", runId: "run1" })
    expect(cfg.userId).toBe("u")
    expect(cfg.agentId).toBe("agent1")
    expect(cfg.runId).toBe("run1")
  })

  it("accepts boolean/number optional fields", () => {
    const cfg = decodeConfig({ url: "http://x", autoCapture: true, autoRecall: false, topK: 10 })
    expect(cfg.autoCapture).toBe(true)
    expect(cfg.autoRecall).toBe(false)
    expect(cfg.topK).toBe(10)
  })

  it("throws a human-readable error when url is missing", () => {
    expect(() => decodeConfig({ userId: "u" })).toThrow("invalid plugin config")
  })

  it("throws when url is not a string", () => {
    expect(() => decodeConfig({ url: 123 })).toThrow("invalid plugin config")
  })

  it("throws when userId is not a string", () => {
    expect(() => decodeConfig({ url: "http://x", userId: 42 })).toThrow("invalid plugin config")
  })

  it("throws when topK is not a number", () => {
    expect(() => decodeConfig({ url: "http://x", topK: "five" })).toThrow("invalid plugin config")
  })
})

// ---------------------------------------------------------------------------
// decodeSearchResponse
// ---------------------------------------------------------------------------
describe("decodeSearchResponse", () => {
  it("parses a valid search response", () => {
    const raw = { results: [{ id: "1", memory: "foo", score: 0.9, user_id: "u" }] }
    const res = decodeSearchResponse(raw)
    expect(res.results).toHaveLength(1)
    expect(res.results[0].memory).toBe("foo")
    expect(res.results[0].score).toBe(0.9)
  })

  it("parses a response with empty results array", () => {
    const res = decodeSearchResponse({ results: [] })
    expect(res.results).toHaveLength(0)
  })

  it("throws on invalid search response (missing results)", () => {
    expect(() => decodeSearchResponse({})).toThrow()
  })

  it("throws when result is missing required fields", () => {
    expect(() => decodeSearchResponse({ results: [{ id: "1" }] })).toThrow()
  })

  it("throws when score is not a number", () => {
    expect(() => decodeSearchResponse({ results: [{ id: "1", memory: "m", score: "high" }] })).toThrow()
  })
})

// ---------------------------------------------------------------------------
// decodeAddResponse
// ---------------------------------------------------------------------------
describe("decodeAddResponse", () => {
  it("parses a valid add response", () => {
    const raw = { results: [{ id: "1", memory: "bar", event: "ADD" }] }
    const res = decodeAddResponse(raw)
    expect(res.results[0].event).toBe("ADD")
  })

  it("throws on missing event field", () => {
    expect(() => decodeAddResponse({ results: [{ id: "1", memory: "m" }] })).toThrow()
  })

  it("throws on invalid add response (no results key)", () => {
    expect(() => decodeAddResponse({ data: [] })).toThrow()
  })
})

// ---------------------------------------------------------------------------
// decodeHealthResponse
// ---------------------------------------------------------------------------
describe("decodeHealthResponse", () => {
  it("parses a valid health response", () => {
    const res = decodeHealthResponse({ status: "ok" })
    expect(res.status).toBe("ok")
  })

  it("throws when status is missing", () => {
    expect(() => decodeHealthResponse({})).toThrow()
  })

  it("throws when status is not a string", () => {
    expect(() => decodeHealthResponse({ status: 1 })).toThrow()
  })
})

// ---------------------------------------------------------------------------
// decodeCountResponse
// ---------------------------------------------------------------------------
describe("decodeCountResponse", () => {
  it("parses a valid count response", () => {
    const res = decodeCountResponse({ count: 42 })
    expect(res.count).toBe(42)
  })

  it("throws when count is missing", () => {
    expect(() => decodeCountResponse({})).toThrow()
  })

  it("throws when count is not a number", () => {
    expect(() => decodeCountResponse({ count: "many" })).toThrow()
  })
})

// ---------------------------------------------------------------------------
// mem0Search
// ---------------------------------------------------------------------------
describe("mem0Search", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  it("POSTs to /search and returns results", async () => {
    const mockFetch = mock(() =>
      Promise.resolve({
        ok: true,
        json: async () => ({ results: [{ id: "1", memory: "remember this", score: 0.95 }] }),
      } as Response)
    )
    globalThis.fetch = mockFetch as any

    const results = await mem0Search("http://localhost:8000", "test query", "user1", 5)
    expect(mockFetch.mock.calls).toHaveLength(1)
    const [url, opts] = mockFetch.mock.calls[0] as [string, RequestInit]
    expect(url).toBe("http://localhost:8000/search")
    expect(opts.method).toBe("POST")
    expect(JSON.parse(opts.body as string)).toEqual({ query: "test query", user_id: "user1", limit: 5 })
    expect(results).toHaveLength(1)
    expect(results[0].memory).toBe("remember this")
  })

  it("includes agentId in body when provided", async () => {
    const mockFetch = mock(() =>
      Promise.resolve({ ok: true, json: async () => ({ results: [] }) } as Response)
    )
    globalThis.fetch = mockFetch as any
    await mem0Search("http://localhost:8000", "q", "u", 5, "agent1")
    const body = JSON.parse((mockFetch.mock.calls[0] as [string, RequestInit])[1].body as string)
    expect(body.agent_id).toBe("agent1")
  })

  it("throws on non-ok HTTP response", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({ ok: false, status: 500, statusText: "Server Error" } as Response)
    ) as any
    await expect(mem0Search("http://localhost:8000", "q", "u", 5)).rejects.toThrow("HTTP 500")
  })

  it("propagates fetch network errors", async () => {
    globalThis.fetch = mock(() => Promise.reject(new Error("Network failure"))) as any
    await expect(mem0Search("http://localhost:8000", "q", "u", 5)).rejects.toThrow("Network failure")
  })
})

// ---------------------------------------------------------------------------
// mem0Add
// ---------------------------------------------------------------------------
describe("mem0Add", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  it("POSTs to /add with user_id", async () => {
    const mockFetch = mock(() =>
      Promise.resolve({
        ok: true,
        json: async () => ({ results: [{ id: "2", memory: "stored", event: "ADD" }] }),
      } as Response)
    )
    globalThis.fetch = mockFetch as any

    const results = await mem0Add("http://localhost:8000", "User: hello\nAssistant: hi", "user1")
    const [url, opts] = mockFetch.mock.calls[0] as [string, RequestInit]
    expect(url).toBe("http://localhost:8000/add")
    const body = JSON.parse(opts.body as string)
    expect(body.messages).toBe("User: hello\nAssistant: hi")
    expect(body.user_id).toBe("user1")
    expect(body.agent_id).toBeUndefined()
    expect(results[0].event).toBe("ADD")
  })

  it("includes agent_id and run_id when provided", async () => {
    const mockFetch = mock(() =>
      Promise.resolve({
        ok: true,
        json: async () => ({ results: [{ id: "3", memory: "x", event: "ADD" }] }),
      } as Response)
    )
    globalThis.fetch = mockFetch as any
    await mem0Add("http://localhost:8000", "msg", "user1", "agent1", "run1")
    const body = JSON.parse((mockFetch.mock.calls[0] as [string, RequestInit])[1].body as string)
    expect(body.agent_id).toBe("agent1")
    expect(body.run_id).toBe("run1")
  })

  it("throws on non-ok HTTP response", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({ ok: false, status: 401, statusText: "Unauthorized" } as Response)
    ) as any
    await expect(mem0Add("http://localhost:8000", "msg", "u")).rejects.toThrow("HTTP 401")
  })
})

// ---------------------------------------------------------------------------
// mem0Count
// ---------------------------------------------------------------------------
describe("mem0Count", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  it("GETs /memories/count and returns count", async () => {
    const mockFetch = mock(() =>
      Promise.resolve({ ok: true, json: async () => ({ count: 42 }) } as Response)
    )
    globalThis.fetch = mockFetch as any

    const count = await mem0Count("http://localhost:8000", "user1")
    const [url] = mockFetch.mock.calls[0] as [string]
    expect(url).toBe("http://localhost:8000/memories/count?user_id=user1")
    expect(count).toBe(42)
  })

  it("includes agent_id in query string when provided", async () => {
    const mockFetch = mock(() =>
      Promise.resolve({ ok: true, json: async () => ({ count: 5 }) } as Response)
    )
    globalThis.fetch = mockFetch as any
    await mem0Count("http://localhost:8000", "user1", "agent1")
    const [url] = mockFetch.mock.calls[0] as [string]
    expect(url).toContain("agent_id=agent1")
  })

  it("throws on non-ok response", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({ ok: false, status: 500, statusText: "Error" } as Response)
    ) as any
    await expect(mem0Count("http://localhost:8000", "u")).rejects.toThrow("HTTP 500")
  })
})

// ---------------------------------------------------------------------------
// Plugin initialization
// ---------------------------------------------------------------------------
describe("plugin initialization", () => {
  it("logs initialization message", async () => {
    const mockApi = {
      pluginConfig: { url: "http://localhost:8000", userId: "testuser", agentId: "agent1" },
      logger: { info: mock(() => {}), warn: mock(() => {}) },
      on: mock(() => {}),
      registerCli: mock(() => {}),
    }
    await plugin(mockApi as any)
    expect(mockApi.logger.info.mock.calls.length).toBeGreaterThan(0)
    const infoMsg = (mockApi.logger.info.mock.calls[0] as [string])[0]
    expect(infoMsg).toContain("openclaw-mem0: initialized")
    expect(infoMsg).toContain("http://localhost:8000")
  })

  it("throws with human-readable error if url is missing", async () => {
    const mockApi = {
      pluginConfig: { agentId: "agent1" }, // missing url
      logger: { info: mock(() => {}), warn: mock(() => {}) },
      on: mock(() => {}),
      registerCli: mock(() => {}),
    }
    await expect(plugin(mockApi as any)).rejects.toThrow("invalid plugin config")
  })

  it("does not register before_agent_start when autoRecall is false", async () => {
    const mockApi = {
      pluginConfig: { url: "http://x", autoRecall: false },
      logger: { info: mock(() => {}), warn: mock(() => {}) },
      on: mock(() => {}),
      registerCli: mock(() => {}),
    }
    await plugin(mockApi as any)
    const eventNames = (mockApi.on.mock.calls as [string, unknown][]).map((c) => c[0])
    expect(eventNames).not.toContain("before_agent_start")
  })

  it("registers before_agent_start when autoRecall is true", async () => {
    const mockApi = {
      pluginConfig: { url: "http://x", autoRecall: true },
      logger: { info: mock(() => {}), warn: mock(() => {}) },
      on: mock(() => {}),
      registerCli: mock(() => {}),
    }
    await plugin(mockApi as any)
    const eventNames = (mockApi.on.mock.calls as [string, unknown][]).map((c) => c[0])
    expect(eventNames).toContain("before_agent_start")
  })

  it("registers agent_end when autoCapture is true", async () => {
    const mockApi = {
      pluginConfig: { url: "http://x", autoCapture: true },
      logger: { info: mock(() => {}), warn: mock(() => {}) },
      on: mock(() => {}),
      registerCli: mock(() => {}),
    }
    await plugin(mockApi as any)
    const eventNames = (mockApi.on.mock.calls as [string, unknown][]).map((c) => c[0])
    expect(eventNames).toContain("agent_end")
  })
})

// ---------------------------------------------------------------------------
// autoRecall handler (before_agent_start)
// ---------------------------------------------------------------------------
describe("autoRecall handler", () => {
  let mockApi: any
  let recallHandler: (event: any, ctx: any) => Promise<any>

  beforeEach(async () => {
    mockApi = {
      pluginConfig: { url: "http://localhost:8000", userId: "u", autoRecall: true, topK: 3 },
      logger: { info: mock(() => {}), warn: mock(() => {}) },
      on: mock(() => {}),
      registerCli: mock(() => {}),
    }
    await plugin(mockApi as any)
    const call = (mockApi.on.mock.calls as [string, Function][]).find((c) => c[0] === "before_agent_start")
    recallHandler = call![1]
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  it("returns {} when prompt is too short (< 5 chars)", async () => {
    const result = await recallHandler({ prompt: "hi" }, {})
    expect(result).toEqual({})
  })

  it("returns {} when prompt is missing", async () => {
    const result = await recallHandler({}, {})
    expect(result).toEqual({})
  })

  it("returns prependContext when memories are found", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({
        ok: true,
        json: async () => ({
          results: [
            { id: "1", memory: "User likes TypeScript", score: 0.9 },
            { id: "2", memory: "User prefers short responses", score: 0.8 },
          ],
        }),
      } as Response)
    ) as any

    const result = await recallHandler({ prompt: "what do I like?" }, {})
    expect(result).toHaveProperty("prependContext")
    expect(result.prependContext).toContain("User likes TypeScript")
    expect(result.prependContext).toContain("User prefers short responses")
    expect(result.prependContext).toContain("<relevant-memories>")
    const infoMessages = (mockApi.logger.info.mock.calls as [string][]).map((c) => c[0])
    expect(infoMessages.some((m) => m.includes("injecting 2 memories"))).toBe(true)
  })

  it("returns {} when search returns no results", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve({ ok: true, json: async () => ({ results: [] }) } as Response)
    ) as any
    const result = await recallHandler({ prompt: "what do I like?" }, {})
    expect(result).toEqual({})
  })

  it("logs warn and returns {} when fetch fails", async () => {
    globalThis.fetch = mock(() => Promise.reject(new Error("network error"))) as any
    const result = await recallHandler({ prompt: "what do I like?" }, {})
    expect(result).toEqual({})
    const warnMessages = (mockApi.logger.warn.mock.calls as [string][]).map((c) => c[0])
    expect(warnMessages.some((m) => m.includes("recall failed"))).toBe(true)
  })

  it("uses topK from config when searching", async () => {
    const fetchMock = mock(() =>
      Promise.resolve({ ok: true, json: async () => ({ results: [] }) } as Response)
    )
    globalThis.fetch = fetchMock as any
    await recallHandler({ prompt: "test prompt here" }, {})
    const body = JSON.parse((fetchMock.mock.calls[0] as [string, RequestInit])[1].body as string)
    expect(body.limit).toBe(3)
  })
})

// ---------------------------------------------------------------------------
// autoCapture handler (agent_end)
// ---------------------------------------------------------------------------
describe("autoCapture handler", () => {
  let mockApi: any
  let captureHandler: (event: any, ctx: any) => Promise<void>

  beforeEach(async () => {
    mockApi = {
      pluginConfig: { url: "http://localhost:8000", userId: "u", autoCapture: true },
      logger: { info: mock(() => {}), warn: mock(() => {}) },
      on: mock(() => {}),
      registerCli: mock(() => {}),
    }
    await plugin(mockApi as any)
    const call = (mockApi.on.mock.calls as [string, Function][]).find((c) => c[0] === "agent_end")
    captureHandler = call![1]
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  it("captures messages on success", async () => {
    const fetchMock = mock(() =>
      Promise.resolve({
        ok: true,
        json: async () => ({ results: [{ id: "1", memory: "captured", event: "ADD" }] }),
      } as Response)
    )
    globalThis.fetch = fetchMock as any
    await captureHandler({
      success: true,
      messages: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi there" },
      ],
    }, {})
    expect(fetchMock.mock.calls).toHaveLength(1)
    const [url, opts] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe("http://localhost:8000/add")
    expect(opts.method).toBe("POST")
    const infoMessages = (mockApi.logger.info.mock.calls as [string][]).map((c) => c[0])
    expect(infoMessages.some((m) => m.includes("captured 1 memory"))).toBe(true)
  })

  it("skips when success is false", async () => {
    const fetchMock = mock(() => Promise.resolve({} as Response))
    globalThis.fetch = fetchMock as any
    await captureHandler({ success: false, messages: [{ role: "user", content: "hi" }] }, {})
    expect(fetchMock.mock.calls).toHaveLength(0)
  })

  it("skips when messages array is empty", async () => {
    const fetchMock = mock(() => Promise.resolve({} as Response))
    globalThis.fetch = fetchMock as any
    await captureHandler({ success: true, messages: [] }, {})
    expect(fetchMock.mock.calls).toHaveLength(0)
  })

  it("skips when messages is missing", async () => {
    const fetchMock = mock(() => Promise.resolve({} as Response))
    globalThis.fetch = fetchMock as any
    await captureHandler({ success: true }, {})
    expect(fetchMock.mock.calls).toHaveLength(0)
  })

  it("skips when all messages are system role (not user or assistant)", async () => {
    const fetchMock = mock(() => Promise.resolve({} as Response))
    globalThis.fetch = fetchMock as any
    await captureHandler({
      success: true,
      messages: [{ role: "system", content: "You are a helpful assistant" }],
    }, {})
    expect(fetchMock.mock.calls).toHaveLength(0)
  })

  it("logs warn and does not throw when add fails", async () => {
    globalThis.fetch = mock(() => Promise.reject(new Error("add error"))) as any
    const p = captureHandler({
      success: true,
      messages: [{ role: "user", content: "Hello" }],
    }, {})
    await expect(p).resolves.toBeUndefined()
    const warnMessages = (mockApi.logger.warn.mock.calls as [string][]).map((c) => c[0])
    expect(warnMessages.some((m) => m.includes("capture failed"))).toBe(true)
  })

  it("only sends last 6 messages", async () => {
    const fetchMock = mock(() =>
      Promise.resolve({
        ok: true,
        json: async () => ({ results: [] }),
      } as Response)
    )
    globalThis.fetch = fetchMock as any
    const messages = Array.from({ length: 15 }, (_, i) => ({
      role: i % 2 === 0 ? "user" : "assistant",
      content: `message ${i}`,
    }))
    await captureHandler({ success: true, messages }, {})
    const body = JSON.parse((fetchMock.mock.calls[0] as [string, RequestInit])[1].body as string)
    // Last 6 messages: indices 9-14
    expect(body.messages).toContain("message 14")
    expect(body.messages).not.toContain("message 4")
  })
})
