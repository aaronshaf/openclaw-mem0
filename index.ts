import type { OpenClawPluginApi } from "openclaw/plugin-sdk";

interface PluginConfig {
  url: string;
  userId: string;
  autoCapture?: boolean;
  autoRecall?: boolean;
  topK?: number;
}

interface SearchResult {
  id: string;
  memory: string;
  score: number;
  user_id?: string;
  agent_id?: string;
}

interface AddResult {
  id: string;
  memory: string;
  event: string;
}

async function mem0Search(
  baseUrl: string,
  query: string,
  userId: string,
  limit: number
): Promise<SearchResult[]> {
  const res = await fetch(`${baseUrl}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, user_id: userId, limit }),
  });
  if (!res.ok) {
    throw new Error(`mem0 search failed: ${res.status} ${res.statusText}`);
  }
  const data = (await res.json()) as { results: SearchResult[] };
  return data.results ?? [];
}

async function mem0Add(
  baseUrl: string,
  messages: string,
  userId: string
): Promise<AddResult[]> {
  const res = await fetch(`${baseUrl}/add`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, agent_id: userId, user_id: userId }),
  });
  if (!res.ok) {
    throw new Error(`mem0 add failed: ${res.status} ${res.statusText}`);
  }
  const data = (await res.json()) as { results: AddResult[] };
  return data.results ?? [];
}

export default async function plugin(api: OpenClawPluginApi): Promise<void> {
  const cfg = api.pluginConfig as PluginConfig;
  const topK = cfg.topK ?? 5;

  if (cfg.autoRecall) {
    api.on("before_agent_start", async (event, _ctx) => {
      const prompt =
        (event as { prompt?: string }).prompt ?? "";
      if (!prompt) return {};

      try {
        const results = await mem0Search(cfg.url, prompt, cfg.userId, topK);
        if (results.length === 0) return {};

        const memories = results
          .map((r, i) => `${i + 1}. ${r.memory}`)
          .join("\n");
        const prependContext = `Relevant memories from past conversations:\n${memories}`;
        api.logger.info(`mem0: injected ${results.length} memories`);
        return { prependContext };
      } catch (err) {
        api.logger.warn(`mem0 recall failed: ${String(err)}`);
        return {};
      }
    });
  }

  if (cfg.autoCapture) {
    api.on("agent_end", async (event, _ctx) => {
      const messages: Array<{ role: string; content: string }> =
        (event as { messages?: Array<{ role: string; content: string }> })
          .messages ?? [];

      const last10 = messages.slice(-10);
      if (last10.length === 0) return;

      const formatted = last10
        .map((m) => {
          const role = m.role === "assistant" ? "Assistant" : "User";
          return `${role}: ${m.content}`;
        })
        .join("\n");

      try {
        const results = await mem0Add(cfg.url, formatted, cfg.userId);
        api.logger.info(`mem0: captured ${results.length} memory entries`);
      } catch (err) {
        api.logger.warn(`mem0 capture failed: ${String(err)}`);
      }
    });
  }

  api.registerCli(
    { name: "mem0", description: "Interact with mem0 memory store" },
    (program) => {
      program
        .command("search")
        .description("Search memories")
        .argument("<query>", "Search query")
        .option("-k, --top-k <number>", "Number of results", String(topK))
        .action(async (query: string, opts: { topK?: string }) => {
          const limit = opts.topK ? parseInt(opts.topK, 10) : topK;
          try {
            const results = await mem0Search(cfg.url, query, cfg.userId, limit);
            if (results.length === 0) {
              console.log("No memories found.");
              return;
            }
            results.forEach((r, i) => {
              const score = r.score.toFixed(3);
              console.log(`${i + 1}. [score: ${score}] ${r.memory}`);
            });
          } catch (err) {
            console.error(`Error: ${String(err)}`);
            process.exit(1);
          }
        });

      program
        .command("stats")
        .description("Show mem0 health and memory stats")
        .action(async () => {
          try {
            const healthRes = await fetch(`${cfg.url}/health`);
            if (!healthRes.ok) {
              console.error(
                `Health check failed: ${healthRes.status} ${healthRes.statusText}`
              );
              process.exit(1);
            }
            const health = (await healthRes.json()) as { status: string };
            console.log(`Status: ${health.status}`);

            // Get a rough count via a broad search
            const results = await mem0Search(cfg.url, " ", cfg.userId, 100);
            console.log(`Memories (approx): ${results.length}`);
          } catch (err) {
            console.error(`Error: ${String(err)}`);
            process.exit(1);
          }
        });
    }
  );
}
