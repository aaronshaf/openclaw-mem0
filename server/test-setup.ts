// Global MSW setup — preloaded before every test file via bunfig.toml.
// onUnhandledRequest: "error" ensures any fetch that escapes the mock
// handlers fails the test rather than silently hitting a real service.

import { setupServer } from "msw/node"
import { beforeAll, afterEach, afterAll } from "bun:test"

export const mswServer = setupServer()

beforeAll(() => mswServer.listen({ onUnhandledRequest: "error" }))
afterEach(() => mswServer.resetHandlers())
afterAll(() => mswServer.close())
