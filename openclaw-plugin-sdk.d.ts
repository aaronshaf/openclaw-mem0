declare module "openclaw/plugin-sdk" {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  type AnyProgram = any;

  export interface OpenClawPluginApi {
    pluginConfig: unknown;
    logger: {
      info(msg: string): void;
      warn(msg: string): void;
      error(msg: string): void;
    };
    on(
      event: "before_agent_start",
      handler: (
        event: Record<string, unknown>,
        ctx: Record<string, unknown>
      ) => Promise<{ prependContext?: string }>
    ): void;
    on(
      event: "agent_end",
      handler: (
        event: Record<string, unknown>,
        ctx: Record<string, unknown>
      ) => Promise<void>
    ): void;
    registerCli(
      meta: { name: string; description: string },
      setup: (program: AnyProgram) => void
    ): void;
  }
}
