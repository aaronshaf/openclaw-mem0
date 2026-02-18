import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { decodeConfig, decodeSearchResponse, decodeAddResponse, decodeHealthResponse, decodeCountResponse, mem0Search, mem0Add, mem0Count } from "./index"
import plugin from "./index"

// ---------------------------------------------------------------------------
// decodeConfig
// ---------------------------------------------------------------------------
describe("decodeConfig", () => {
  it("accepts a valid config with required fields", () => {
    const cfg = decodeConfig({ url: "http://localhost:8000", userId: "user1" })
    expect(cfg.url).toBe("http://localhost:8000")
    expect(cfg.userId).toBe("user1")
  })

  it("accepts optional fields", () => {
    const cfg = decodeConfig({ url: "http://x", userId: "u", autoCapture: true, autoRecall: false, topK: 10 })
    expect(cfg.autoCapture).toBe(true)
    expect(cfg.autoRecall).toBe(false)
    expect(cfg.topK).toBe(10)
  })

  it("throws a human-readable error when url is missing", () => {
    expect(() => decodeConfig({ userId: "u" })).toThrow("invalid plugin config")
  })

  it("throws a human-readable error when userId is missing", () => {
    expect(() => decodeConfig({ url: "http://x" })).toThrow("invalid plugin config")
  })

  it("throws when url is not a string", () => {
    expect(() => decodeConfig({ url: 123, userId: "u" })).toThrow("invalid plugin config")
  })

  it("throws when userId is not a string", () => {
    expect(() => decodeConfig({ url: "http://x", userId: 42 })).toThrow("invalid plugin config")
  })

  it("throws when topK is not a number", () => {
    expect(() => decodeConfig({ url: "http://x", userId: "u", topK: "five" })).toThrow("invalid plugin config")
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
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn())
  })
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("POSTs to /search and returns results", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ results: [{ id: "1", memory: "remember this", score: 0.95 }] }),
    })
    vi.stubGlobal("fetch", mockFetch)

    const results = await mem0Search("http://localhost:8000", "test query", "user1", 5)
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/search",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: "test query", user_id: "user1", limit: 5 }),
      })
    )
    expect(results).toHaveLength(1)
    expect(results[0].memory).toBe("remember this")
  })

  it("throws on non-ok HTTP response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 500, statusText: "Server Error" }))
    await expect(mem0Search("http://localhost:8000", "q", "u", 5)).rejects.toThrow("HTTP 500")
  })

  it("propagates fetch network errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("Network failure")))
    await expect(mem0Search("http://localhost:8000", "q", "u", 5)).rejects.toThrow("Network failure")
  })
})

// ---------------------------------------------------------------------------
// mem0Add
// ---------------------------------------------------------------------------
describe("mem0Add", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn())
  })
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("POSTs to /add without agent_id and returns results", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ results: [{ id: "2", memory: "stored", event: "ADD" }] }),
    })
    vi.stubGlobal("fetch", mockFetch)

    const results = await mem0Add("http://localhost:8000", "User: hello\nAssistant: hi", "user1")
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/add",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ messages: "User: hello\nAssistant: hi", user_id: "user1" }),
      })
    )
    expect(results[0].event).toBe("ADD")
  })

  it("throws on non-ok HTTP response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 401, statusText: "Unauthorized" }))
    await expect(mem0Add("http://localhost:8000", "msg", "u")).rejects.toThrow("HTTP 401")
  })
})

// ---------------------------------------------------------------------------
// mem0Count
// ---------------------------------------------------------------------------
describe("mem0Count", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn())
  })
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("GETs /memories/count and returns count", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ count: 42 }),
    })
    vi.stubGlobal("fetch", mockFetch)

    const count = await mem0Count("http://localhost:8000", "user1")
    expect(mockFetch).toHaveBeenCalledWith("http://localhost:8000/memories/count?user_id=user1")
    expect(count).toBe(42)
  })

  it("throws on non-ok response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 500, statusText: "Error" }))
    await expect(mem0Count("http://localhost:8000", "u")).rejects.toThrow("HTTP 500")
  })
})

// ---------------------------------------------------------------------------
// Plugin initialization
// ---------------------------------------------------------------------------
describe("plugin initialization", () => {
  it("logs initialization message and calls decodeConfig", async () => {
    const mockApi = {
      pluginConfig: { url: "http://localhost:8000", userId: "testuser" },
      logger: { info: vi.fn(), warn: vi.fn() },
      on: vi.fn(),
      registerCli: vi.fn(),
    }
    await plugin(mockApi as any)
    expect(mockApi.logger.info).toHaveBeenCalledWith(
      expect.stringContaining("openclaw-mem0: initialized")
    )
    expect(mockApi.logger.info).toHaveBeenCalledWith(
      expect.stringContaining("http://localhost:8000")
    )
    expect(mockApi.logger.info).toHaveBeenCalledWith(
      expect.stringContaining("testuser")
    )
  })

  it("throws with human-readable error if config is invalid", async () => {
    const mockApi = {
      pluginConfig: { url: "http://localhost:8000" }, // missing userId
      logger: { info: vi.fn(), warn: vi.fn() },
      on: vi.fn(),
      registerCli: vi.fn(),
    }
    await expect(plugin(mockApi as any)).rejects.toThrow("invalid plugin config")
  })

  it("does not register before_agent_start when autoRecall is false", async () => {
    const mockApi = {
      pluginConfig: { url: "http://x", userId: "u", autoRecall: false },
      logger: { info: vi.fn(), warn: vi.fn() },
      on: vi.fn(),
      registerCli: vi.fn(),
    }
    await plugin(mockApi as any)
    const eventNames = mockApi.on.mock.calls.map((c: any[]) => c[0])
    expect(eventNames).not.toContain("before_agent_start")
  })

  it("registers before_agent_start when autoRecall is true", async () => {
    const mockApi = {
      pluginConfig: { url: "http://x", userId: "u", autoRecall: true },
      logger: { info: vi.fn(), warn: vi.fn() },
      on: vi.fn(),
      registerCli: vi.fn(),
    }
    await plugin(mockApi as any)
    const eventNames = mockApi.on.mock.calls.map((c: any[]) => c[0])
    expect(eventNames).toContain("before_agent_start")
  })

  it("registers agent_end when autoCapture is true", async () => {
    const mockApi = {
      pluginConfig: { url: "http://x", userId: "u", autoCapture: true },
      logger: { info: vi.fn(), warn: vi.fn() },
      on: vi.fn(),
      registerCli: vi.fn(),
    }
    await plugin(mockApi as any)
    const eventNames = mockApi.on.mock.calls.map((c: any[]) => c[0])
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
    vi.stubGlobal("fetch", vi.fn())
    mockApi = {
      pluginConfig: { url: "http://localhost:8000", userId: "u", autoRecall: true, topK: 3 },
      logger: { info: vi.fn(), warn: vi.fn() },
      on: vi.fn(),
      registerCli: vi.fn(),
    }
    await plugin(mockApi as any)
    const call = mockApi.on.mock.calls.find((c: any[]) => c[0] === "before_agent_start")
    recallHandler = call[1]
  })

  afterEach(() => {
    vi.unstubAllGlobals()
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
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        results: [
          { id: "1", memory: "User likes TypeScript", score: 0.9 },
          { id: "2", memory: "User prefers short responses", score: 0.8 },
        ],
      }),
    }))
    const result = await recallHandler({ prompt: "what do I like?" }, {})
    expect(result).toHaveProperty("prependContext")
    expect(result.prependContext).toContain("User likes TypeScript")
    expect(result.prependContext).toContain("User prefers short responses")
    expect(result.prependContext).toContain("<relevant-memories>")
    expect(mockApi.logger.info).toHaveBeenCalledWith(
      expect.stringContaining("injecting 2 memories")
    )
  })

  it("returns {} when search returns no results", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ results: [] }),
    }))
    const result = await recallHandler({ prompt: "what do I like?" }, {})
    expect(result).toEqual({})
  })

  it("logs warn and returns {} when fetch fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network error")))
    const result = await recallHandler({ prompt: "what do I like?" }, {})
    expect(result).toEqual({})
    expect(mockApi.logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("recall failed")
    )
  })

  it("uses topK from config when searching", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ results: [] }),
    })
    vi.stubGlobal("fetch", fetchMock)
    await recallHandler({ prompt: "test prompt here" }, {})
    const body = JSON.parse(fetchMock.mock.calls[0][1].body)
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
    vi.stubGlobal("fetch", vi.fn())
    mockApi = {
      pluginConfig: { url: "http://localhost:8000", userId: "u", autoCapture: true },
      logger: { info: vi.fn(), warn: vi.fn() },
      on: vi.fn(),
      registerCli: vi.fn(),
    }
    await plugin(mockApi as any)
    const call = mockApi.on.mock.calls.find((c: any[]) => c[0] === "agent_end")
    captureHandler = call[1]
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("captures messages on success", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ results: [{ id: "1", memory: "captured", event: "ADD" }] }),
    })
    vi.stubGlobal("fetch", fetchMock)
    await captureHandler({
      success: true,
      messages: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi there" },
      ],
    }, {})
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/add",
      expect.objectContaining({ method: "POST" })
    )
    // Verify no agent_id is sent
    const body = JSON.parse(fetchMock.mock.calls[0][1].body)
    expect(body).not.toHaveProperty("agent_id")
    expect(mockApi.logger.info).toHaveBeenCalledWith(
      expect.stringContaining("captured 1 memory")
    )
  })

  it("skips when success is false", async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal("fetch", fetchMock)
    await captureHandler({ success: false, messages: [{ role: "user", content: "hi" }] }, {})
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("skips when messages array is empty", async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal("fetch", fetchMock)
    await captureHandler({ success: true, messages: [] }, {})
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("skips when messages is missing", async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal("fetch", fetchMock)
    await captureHandler({ success: true }, {})
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("skips when all messages are system role (not user or assistant)", async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal("fetch", fetchMock)
    await captureHandler({
      success: true,
      messages: [{ role: "system", content: "You are a helpful assistant" }],
    }, {})
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("logs warn and does not throw when add fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("add error")))
    await expect(captureHandler({
      success: true,
      messages: [{ role: "user", content: "Hello" }],
    }, {})).resolves.not.toThrow()
    expect(mockApi.logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("capture failed")
    )
  })

  it("only sends last 10 messages", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ results: [] }),
    })
    vi.stubGlobal("fetch", fetchMock)
    const messages = Array.from({ length: 15 }, (_, i) => ({
      role: i % 2 === 0 ? "user" : "assistant",
      content: `message ${i}`,
    }))
    await captureHandler({ success: true, messages }, {})
    const body = JSON.parse(fetchMock.mock.calls[0][1].body)
    // Last 10 messages: indices 5-14
    expect(body.messages).toContain("message 14")
    expect(body.messages).not.toContain("message 4")
  })
})
