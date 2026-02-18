import { Schema } from "effect"
import type { OpenClawPluginApi } from "openclaw/plugin-sdk"

// --- Config schema ---

const PluginConfig = Schema.Struct({
  url: Schema.String,
  userId: Schema.String,
  autoCapture: Schema.optional(Schema.Boolean),
  autoRecall: Schema.optional(Schema.Boolean),
  topK: Schema.optional(Schema.Number),
})
type PluginConfig = Schema.Schema.Type<typeof PluginConfig>

function decodeConfig(raw: unknown): PluginConfig {
  try {
    return Schema.decodeUnknownSync(PluginConfig)(raw)
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e)
    throw new Error(
      `openclaw-mem0: invalid plugin config. Check that 'url' (string) and 'userId' (string) are set. Details: ${msg}`
    )
  }
}

// --- API response schemas ---

const SearchResult = Schema.Struct({
  id: Schema.String,
  memory: Schema.String,
  score: Schema.Number,
  user_id: Schema.optional(Schema.String),
})

const SearchResponse = Schema.Struct({
  results: Schema.Array(SearchResult),
})

const AddResult = Schema.Struct({
  id: Schema.String,
  memory: Schema.String,
  event: Schema.String,
})

const AddResponse = Schema.Struct({
  results: Schema.Array(AddResult),
})

const HealthResponse = Schema.Struct({
  status: Schema.String,
})

const CountResponse = Schema.Struct({
  count: Schema.Number,
})

const decodeSearchResponse = Schema.decodeUnknownSync(SearchResponse)
const decodeAddResponse = Schema.decodeUnknownSync(AddResponse)
const decodeHealthResponse = Schema.decodeUnknownSync(HealthResponse)
const decodeCountResponse = Schema.decodeUnknownSync(CountResponse)

// --- HTTP helpers ---

async function postJson(url: string, body: unknown): Promise<unknown> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`)
  return res.json()
}

async function mem0Search(baseUrl: string, query: string, userId: string, limit: number) {
  const raw = await postJson(`${baseUrl}/search`, { query, user_id: userId, limit })
  return decodeSearchResponse(raw).results
}

async function mem0Add(baseUrl: string, messages: string, userId: string) {
  const raw = await postJson(`${baseUrl}/add`, { messages, user_id: userId })
  return decodeAddResponse(raw).results
}

async function mem0Count(baseUrl: string, userId: string): Promise<number> {
  const res = await fetch(`${baseUrl}/memories/count?user_id=${encodeURIComponent(userId)}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`)
  const raw = await res.json()
  return decodeCountResponse(raw).count
}

// --- Plugin ---

export default async function plugin(api: OpenClawPluginApi): Promise<void> {
  const cfg = decodeConfig(api.pluginConfig)
  const topK = cfg.topK ?? 5

  api.logger.info(
    `openclaw-mem0: initialized (url: ${cfg.url}, user: ${cfg.userId}, autoRecall: ${cfg.autoRecall ?? false}, autoCapture: ${cfg.autoCapture ?? false})`
  )

  if (cfg.autoRecall) {
    api.on("before_agent_start", async (event, _ctx) => {
      const prompt = (event as { prompt?: string }).prompt ?? ""
      if (prompt.length < 5) return {}
      try {
        const results = await mem0Search(cfg.url, prompt, cfg.userId, topK)
        if (results.length === 0) return {}
        const memories = results.map((r) => `- ${r.memory}`).join("\n")
        api.logger.info(`openclaw-mem0: injecting ${results.length} memories`)
        return { prependContext: `<relevant-memories>\n${memories}\n</relevant-memories>` }
      } catch (err) {
        api.logger.warn(`openclaw-mem0: recall failed: ${String(err)}`)
        return {}
      }
    })
  }

  if (cfg.autoCapture) {
    api.on("agent_end", async (event, _ctx) => {
      const ev = event as { success?: boolean; messages?: Array<{ role: string; content: unknown }> }
      if (!ev.success || !ev.messages?.length) return
      const recent = ev.messages.slice(-6).filter((m) => m.role === "user" || m.role === "assistant")
      if (recent.length === 0) return
      const contentToString = (content: unknown): string => {
        if (typeof content === "string") return content
        if (Array.isArray(content)) {
          return content
            .map((block: unknown) => {
              if (typeof block === "string") return block
              if (typeof block === "object" && block !== null) {
                const b = block as Record<string, unknown>
                if (b.type === "text" && typeof b.text === "string") return b.text
                if (b.type === "tool_result" || b.type === "tool_use") return "" // skip tool calls
              }
              return ""
            })
            .filter(Boolean)
            .join(" ")
        }
        return String(content)
      }
      const MAX_CONTENT = 500
      const formatted = recent
        .map((m) => {
          const text = contentToString(m.content).trim().slice(0, MAX_CONTENT)
          if (!text) return null
          return `${m.role === "assistant" ? "Assistant" : "User"}: ${text}`
        })
        .filter(Boolean)
        .join("\n")
      try {
        const results = await mem0Add(cfg.url, formatted, cfg.userId)
        api.logger.info(`openclaw-mem0: captured ${results.length} memory entries`)
      } catch (err) {
        api.logger.warn(`openclaw-mem0: capture failed: ${String(err)}`)
      }
    })
  }

  api.registerCli(
    ({ program }: { program: any }) => {
      const mem0 = program.command("mem0").description("mem0 memory commands")
      mem0
        .command("search")
        .description("Search memories")
        .argument("<query>", "Search query")
        .option("-k, --top-k <n>", "Max results", String(topK))
        .action(async (query: string, opts: { topK?: string }) => {
          const limit = opts.topK ? parseInt(opts.topK, 10) : topK
          const results = await mem0Search(cfg.url, query, cfg.userId, limit).catch((e) => {
            console.error(`Error: ${String(e)}`)
            process.exit(1)
          })
          if (!results.length) { console.log("No memories found."); return }
          results.forEach((r, i) => console.log(`${i + 1}. [${r.score.toFixed(3)}] ${r.memory}`))
        })

      mem0
        .command("add")
        .description("Manually store a memory")
        .argument("<text>", "Text to store as memory")
        .action(async (text: string) => {
          const results = await mem0Add(cfg.url, text, cfg.userId).catch((e) => {
            console.error(`Error: ${String(e)}`)
            process.exit(1)
          })
          if (!results.length) { console.log("No memories extracted."); return }
          results.forEach((r, i) => console.log(`${i + 1}. stored: ${r.memory}`))
        })

      mem0
        .command("delete")
        .description("Delete a memory by ID")
        .argument("<id>", "Memory ID")
        .action(async (id: string) => {
          const res = await fetch(`${cfg.url}/memories/${id}`, { method: "DELETE" }).catch((e) => {
            console.error(`Error: ${String(e)}`); process.exit(1)
          })
          const data = await res.json() as { success?: boolean; error?: string }
          if (data.success) { console.log(`Deleted ${id}`) } else { console.error(`Error: ${data.error}`); process.exit(1) }
        })

      mem0
        .command("list")
        .description("List all memories")
        .action(async () => {
          const res = await fetch(`${cfg.url}/memories?user_id=${cfg.userId}`).catch((e) => {
            console.error(`Error: ${String(e)}`); process.exit(1)
          })
          const data = await res.json() as { memories?: Array<{ id: string; memory: string }> }
          if (!data.memories?.length) { console.log("No memories."); return }
          data.memories.forEach((m, i) => console.log(`${i + 1}. [${m.id}] ${m.memory}`))
        })

      mem0
        .command("stats")
        .description("Health check and memory count")
        .action(async () => {
          const res = await fetch(`${cfg.url}/health`).catch((e) => {
            console.error(`Error: ${String(e)}`); process.exit(1)
          })
          const health = decodeHealthResponse(await res.json())
          console.log(`Status: ${health.status}`)
          try {
            const count = await mem0Count(cfg.url, cfg.userId)
            console.log(`Memories: ${count}`)
          } catch (e) {
            console.error(`Failed to get count: ${String(e)}`)
          }
        })
    },
    { commands: ["mem0"] }
  )
}

export { mem0Search, mem0Add, mem0Count, decodeConfig, decodeSearchResponse, decodeAddResponse, decodeHealthResponse, decodeCountResponse }
