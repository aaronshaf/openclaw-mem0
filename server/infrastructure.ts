// Infrastructure code (layer factories + server bootstrap).
// Layer factories are tested in infrastructure.test.ts via MSW.

import { Config, Effect, Layer, Schema, pipe } from "effect"
import type { HealthResponse } from "../shared/types"
import {
  AppConfig,
  QdrantClient,
  OllamaClient,
  QdrantError,
  OllamaError,
  OllamaEmbedResponseSchema,
  OllamaGenerateResponseSchema,
  OpenAIEmbedResponseSchema,
  QdrantSearchResponseSchema,
  QdrantScrollResponseSchema,
  QdrantCountResponseSchema,
  handleAdd,
  handleSearch,
  handleDelete,
  handleListMemories,
  handleCountMemories,
  json,
  errorResponse,
} from "./index"
import type { QdrantClientService, OllamaClientService } from "./index"

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

// --- Helpers ---

function buildMustFilter(userId: string, agentId?: string, runId?: string): Array<Record<string, unknown>> {
  const must: Array<Record<string, unknown>> = [{ key: "user_id", match: { value: userId } }]
  if (agentId) must.push({ key: "agent_id", match: { value: agentId } })
  if (runId) must.push({ key: "run_id", match: { value: runId } })
  return must
}

// --- Layer factories ---

export function makeQdrantClientLive(cfg: {
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

    searchPoints: (vector, userId, limit, agentId, runId) => {
      const must = buildMustFilter(userId, agentId, runId)
      return pipe(
        fetchJsonAs(`${qdrantUrl}/collections/${collectionName}/points/search`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            vector,
            limit,
            with_payload: true,
            filter: { must },
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
            agent_id: r.payload["agent_id"] ? String(r.payload["agent_id"]) : undefined,
            run_id: r.payload["run_id"] ? String(r.payload["run_id"]) : undefined,
          }))
        )
      )
    },

    deletePoint: (id) =>
      pipe(
        fetchJsonAs(`${qdrantUrl}/collections/${collectionName}/points/delete`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ points: [id] }),
        }, mkErr),
        Effect.map(() => undefined)
      ),

    scrollPoints: (userId, agentId, runId) => {
      const must = buildMustFilter(userId, agentId, runId)
      const loop = (
        offset: string | number | null,
        acc: Memory[],
        page: number
      ): Effect.Effect<Memory[], QdrantError, never> => {
        if (page > 100) {
          console.warn("[qdrant] scrollPoints hit 100-page limit, stopping")
          return Effect.succeed(acc)
        }
        return pipe(
          fetchJsonAs(`${qdrantUrl}/collections/${collectionName}/points/scroll`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              limit: 250,
              with_payload: true,
              with_vector: false,
              offset: offset ?? undefined,
              filter: { must },
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
              agent_id: p.payload["agent_id"] ? String(p.payload["agent_id"]) : undefined,
              run_id: p.payload["run_id"] ? String(p.payload["run_id"]) : undefined,
              created_at: p.payload["created_at"] ? String(p.payload["created_at"]) : undefined,
            }))
            const all = [...acc, ...newPoints]
            const nextOffset = decoded.result.next_page_offset
            if (nextOffset == null) return Effect.succeed(all)
            return loop(nextOffset as string | number, all, page + 1)
          })
        )
      }
      return loop(null, [], 0)
    },

    countPoints: (userId, agentId, runId) => {
      const must = buildMustFilter(userId, agentId, runId)
      return pipe(
        fetchJsonAs(`${qdrantUrl}/collections/${collectionName}/points/count`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            filter: { must },
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
      )
    },
  }

  return Layer.succeed(QdrantClient, impl)
}

export function makeOllamaClientLive(cfg: {
  ollamaUrl: string
  llmModel: string
  embedModel: string
  embedProvider: string
  openaiApiKey?: string
  openaiBaseUrl?: string
}): Layer.Layer<OllamaClient, never, never> {
  const { ollamaUrl, llmModel, embedModel, embedProvider, openaiApiKey, openaiBaseUrl } = cfg
  const mkErr = (msg: string) => new OllamaError({ message: msg })

  const impl: OllamaClientService = {
    embed: (text) => {
      const isOpenAI = embedProvider === "openai"
      const baseUrl = isOpenAI ? (openaiBaseUrl ?? "https://api.openai.com") : ollamaUrl
      const url = isOpenAI ? `${baseUrl}/v1/embeddings` : `${baseUrl}/api/embeddings`
      const body = isOpenAI
        ? JSON.stringify({ model: embedModel, input: text })
        : JSON.stringify({ model: embedModel, prompt: text })
      const headers: Record<string, string> = { "Content-Type": "application/json" }
      if (isOpenAI && openaiApiKey) headers["Authorization"] = `Bearer ${openaiApiKey}`

      if (isOpenAI) {
        return pipe(
          fetchJsonAs(url, { method: "POST", headers, body }, mkErr),
          Effect.flatMap((data) => pipe(
            Schema.decodeUnknown(OpenAIEmbedResponseSchema)(data),
            Effect.mapError((e) => mkErr(`Embed response decode failed: ${String(e)}`))
          )),
          Effect.flatMap((decoded) => {
            if (decoded.data.length === 0) {
              return Effect.fail(mkErr("OpenAI returned empty embeddings array"))
            }
            return Effect.succeed([...decoded.data[0].embedding])
          })
        )
      }

      return pipe(
        fetchJsonAs(url, { method: "POST", headers, body }, mkErr),
        Effect.flatMap((data) =>
          pipe(
            Schema.decodeUnknown(OllamaEmbedResponseSchema)(data),
            Effect.mapError((e) => mkErr(`Embed response decode failed: ${String(e)}`))
          )
        ),
        Effect.map((decoded) => [...decoded.embedding])
      )
    },

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

// --- Server bootstrap ---
