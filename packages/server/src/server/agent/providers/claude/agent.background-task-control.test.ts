import { describe, expect, test, vi } from "vitest";
import type { Query, SDKMessage, SDKTaskStartedMessage } from "@anthropic-ai/claude-agent-sdk";
import { ClaudeAgentClient } from "./agent.js";
import { createTestLogger } from "../../../../test-utils/test-logger.js";
import { asInternals } from "../../../test-utils/class-mocks.js";
import {
  AgentTimelineEntryPayloadSchema,
  type AgentBackgroundTaskTarget,
} from "@getpaseo/protocol/messages";
import type { AgentMetadata, AgentStreamEvent } from "../../agent-sdk-types.js";
import { projectTimelineRows } from "../../timeline-projection.js";
import { ClaudeTaskProtocolSource } from "./subagents/live-source.js";

function taskStarted(overrides: Partial<SDKTaskStartedMessage> = {}): SDKTaskStartedMessage {
  return {
    type: "system",
    subtype: "task_started",
    task_id: "native-task-a",
    tool_use_id: "tool-a",
    description: "Synthetic task",
    task_type: "local_bash",
    is_backgrounded: false,
    session_id: "session-a",
    uuid: "00000000-0000-4000-8000-000000000001",
    ...overrides,
  };
}

const target: AgentBackgroundTaskTarget = {
  expectedTaskId: "native-task-a",
  expectedNativeSessionId: "session-a",
  expectedTurnId: "turn-a",
  toolUseId: "tool-a",
};

async function fixture(registration: "foreground" | "unknown" | "absent" = "foreground") {
  const client = new ClaudeAgentClient({
    logger: createTestLogger(),
    queryFactory: vi.fn(() => {
      throw new Error("No SDK process may start in this test");
    }),
    resolveBinary: async () => "/synthetic/claude",
  });
  const session = await client.createSession({ provider: "claude", cwd: "/tmp" });
  const backgroundTasks = vi.fn<Query["backgroundTasks"]>(async () => true);
  const query: Partial<Pick<Query, "backgroundTasks">> = { backgroundTasks };
  const internal = asInternals<{
    query: typeof query;
    activeForegroundQuery: typeof query;
    activeForegroundTurnId: string | null;
    claudeSessionId: string;
    compacting: boolean;
    taskProtocolSource: ClaudeTaskProtocolSource;
    translateMessageToEvents(message: SDKMessage): Promise<AgentStreamEvent[]>;
    toolUseCache: Map<
      string,
      {
        id: string;
        name: string;
        server: string;
        classification: string;
        started: boolean;
        turnId: string | null;
        input?: AgentMetadata;
      }
    >;
  }>(session);
  internal.query = query;
  internal.activeForegroundQuery = query;
  internal.activeForegroundTurnId = target.expectedTurnId;
  internal.claudeSessionId = target.expectedNativeSessionId;
  internal.toolUseCache.set(target.toolUseId, {
    id: target.toolUseId,
    name: "Bash",
    server: "Bash",
    classification: "command",
    started: true,
    turnId: target.expectedTurnId,
  });
  if (registration !== "absent")
    internal.taskProtocolSource.observeNativeTask(
      taskStarted(registration === "unknown" ? { is_backgrounded: undefined } : {}),
    );
  return { session, internal, backgroundTasks };
}

describe("targeted background control", () => {
  test("requests exactly the active tool and leaves its state intact", async () => {
    const { session, internal, backgroundTasks } = await fixture();
    expect(await session.backgroundTask!(target)).toEqual({ status: "requested" });
    expect(backgroundTasks.mock.calls).toEqual([["tool-a"]]);
    expect(internal.activeForegroundTurnId).toBe("turn-a");
    expect(internal.toolUseCache.get("tool-a")?.started).toBe(true);
  });

  test.each([
    { ...target, expectedTaskId: "different-native-task" },
    { ...target, expectedNativeSessionId: "session-b" },
    { ...target, expectedTurnId: "turn-b" },
    { ...target, toolUseId: "tool-b" },
    { ...target, toolUseId: "" },
  ])("refuses a stale or missing target %j", async (request) => {
    const { session, backgroundTasks } = await fixture();
    expect(await session.backgroundTask!(request)).toEqual({ status: "stale" });
    expect(backgroundTasks.mock.calls).toEqual([]);
  });

  test("reports an unsupported native query without a fallback", async () => {
    const { session, internal } = await fixture();
    delete internal.query.backgroundTasks;
    expect(await session.backgroundTask!(target)).toEqual({ status: "unsupported" });
  });

  test("does not dispatch while compacting", async () => {
    const { session, internal, backgroundTasks } = await fixture();
    internal.compacting = true;
    expect(await session.backgroundTask!(target)).toEqual({ status: "stale" });
    expect(backgroundTasks.mock.calls).toEqual([]);
  });

  test("does not dispatch an arbitrary tool as a Bash or subagent task", async () => {
    const { session, internal, backgroundTasks } = await fixture();
    internal.toolUseCache.set(target.toolUseId, {
      id: target.toolUseId,
      name: "Read",
      server: "Read",
      classification: "generic",
      started: true,
      turnId: target.expectedTurnId,
    });
    expect(await session.backgroundTask!(target)).toEqual({ status: "stale" });
    expect(backgroundTasks.mock.calls).toEqual([]);
  });

  test("does not apply the current turn guard to an earlier turn's unfinished tool", async () => {
    const { session, internal, backgroundTasks } = await fixture();
    const entry = internal.toolUseCache.get(target.toolUseId);
    if (!entry) throw new Error("Missing synthetic tool");
    internal.toolUseCache.set(target.toolUseId, { ...entry, turnId: "earlier-turn" });
    expect(await session.backgroundTask!(target)).toEqual({ status: "stale" });
    expect(backgroundTasks.mock.calls).toEqual([]);
  });

  test("refuses a completed tool", async () => {
    const { session, internal, backgroundTasks } = await fixture();
    internal.toolUseCache.delete(target.toolUseId);
    expect(await session.backgroundTask!(target)).toEqual({ status: "stale" });
    expect(backgroundTasks.mock.calls).toEqual([]);
  });

  test("keeps a native refusal distinct from an unconfirmed reply", async () => {
    const { session, backgroundTasks } = await fixture();
    backgroundTasks.mockResolvedValueOnce(false);
    expect(await session.backgroundTask!(target)).toEqual({ status: "refused" });
    backgroundTasks.mockRejectedValueOnce(new Error("Native response lost"));
    expect(await session.backgroundTask!(target)).toEqual({
      status: "unconfirmed",
      error: "Native response lost",
    });
    expect(backgroundTasks.mock.calls).toEqual([["tool-a"], ["tool-a"]]);
  });

  test("rejects an obsolete query without touching the replacement", async () => {
    const { session, internal, backgroundTasks } = await fixture();
    internal.query = { backgroundTasks: vi.fn() };
    expect(await session.backgroundTask!(target)).toEqual({ status: "stale" });
    expect(backgroundTasks.mock.calls).toEqual([]);
  });
});

test("targeted background control exposes observed task identity on the public tool timeline", async () => {
  const { internal } = await fixture("absent");
  const events = await internal.translateMessageToEvents(taskStarted());
  const projected = projectTimelineRows({
    mode: "projected",
    rows: events.flatMap((event, index) =>
      event.type === "timeline"
        ? [
            {
              seq: index + 1,
              timestamp: "2026-11-25T00:00:00Z",
              turnId: event.turnId,
              item: event.item,
            },
          ]
        : [],
    ),
  });
  const entry = AgentTimelineEntryPayloadSchema.parse({ ...projected[0], provider: "claude" });
  expect(entry).toMatchObject({
    turnId: "turn-a",
    item: {
      type: "tool_call",
      callId: "tool-a",
      metadata: {
        nativeTask: {
          nativeSessionId: "session-a",
          turnId: "turn-a",
          toolUseId: "tool-a",
          taskId: "native-task-a",
          state: "foreground",
        },
      },
    },
  });
});

test.each(["unknown", "absent"] as const)(
  "targeted background control refuses %s admission without native dispatch",
  async (registration) => {
    const { session, backgroundTasks } = await fixture(registration);
    expect(await session.backgroundTask!(target)).toEqual({ status: "refused" });
    expect(backgroundTasks.mock.calls).toEqual([]);
  },
);

test("targeted background control retains a completed tool's background task across a new turn", async () => {
  const { internal } = await fixture("absent");
  const original = await internal.translateMessageToEvents(taskStarted());
  const background = await internal.translateMessageToEvents({
    type: "system",
    subtype: "task_updated",
    task_id: "native-task-a",
    patch: { is_backgrounded: true },
    session_id: "session-a",
    uuid: "00000000-0000-4000-8000-000000000002",
  });
  const settled = await internal.translateMessageToEvents({
    type: "user",
    session_id: "session-a",
    parent_tool_use_id: null,
    message: {
      role: "user",
      content: [
        { type: "tool_result", tool_use_id: "tool-a", content: "Running in the background" },
      ],
    },
  });
  expect(settled).toEqual(
    expect.arrayContaining([
      expect.objectContaining({
        type: "timeline",
        item: expect.objectContaining({
          type: "tool_call",
          status: "completed",
          metadata: { nativeTask: expect.objectContaining({ state: "background" }) },
        }),
      }),
    ]),
  );
  expect(internal.toolUseCache.has("tool-a")).toBe(false);
  internal.activeForegroundTurnId = "turn-b";
  const terminal = await internal.translateMessageToEvents({
    type: "system",
    subtype: "task_updated",
    task_id: "native-task-a",
    patch: { status: "completed" },
    session_id: "session-a",
    uuid: "00000000-0000-4000-8000-000000000003",
  });
  const projected = projectTimelineRows({
    mode: "projected",
    rows: [
      ...original,
      ...background,
      ...settled,
      {
        type: "timeline" as const,
        provider: "claude",
        turnId: "turn-b",
        item: { type: "user_message" as const, text: "Another turn" },
      },
      ...terminal,
    ].flatMap((event, index) =>
      event.type === "timeline"
        ? [
            {
              seq: index + 1,
              timestamp: "2026-11-25T00:00:00Z",
              turnId: event.turnId ?? "turn-a",
              item: event.item,
            },
          ]
        : [],
    ),
  });
  expect(projected).toHaveLength(2);
  const originalTool = projected.find((entry) => entry.item.type === "tool_call");
  const entry = AgentTimelineEntryPayloadSchema.parse({ ...originalTool, provider: "claude" });
  expect(entry).toMatchObject({
    turnId: "turn-a",
    item: {
      status: "completed",
      metadata: {
        nativeTask: {
          nativeSessionId: "session-a",
          turnId: "turn-a",
          toolUseId: "tool-a",
          taskId: "native-task-a",
          state: "completed",
        },
      },
    },
  });
});

test.each(["Agent", "Task"])(
  "targeted %s conversion preserves the original parent card and later child output",
  async (name) => {
    const { session, internal, backgroundTasks } = await fixture("absent");
    const tool = internal.toolUseCache.get(target.toolUseId);
    if (!tool) throw new Error("Missing synthetic parent tool");
    internal.toolUseCache.set(target.toolUseId, {
      ...tool,
      name,
      server: name,
      input: {
        subagent_type: "general-purpose",
        description: "Synthetic task",
        prompt: "Return the original child result",
      },
    });
    const original = await internal.translateMessageToEvents(
      taskStarted({ task_type: "local_agent", subagent_type: "general-purpose" }),
    );
    expect(await session.backgroundTask!({ ...target, expectedTaskId: "another-child" })).toEqual({
      status: "stale",
    });
    expect(backgroundTasks.mock.calls).toEqual([]);
    expect(await session.backgroundTask!(target)).toEqual({ status: "requested" });
    expect(backgroundTasks.mock.calls).toEqual([[target.toolUseId]]);
    const background = await internal.translateMessageToEvents({
      type: "system",
      subtype: "task_updated",
      task_id: target.expectedTaskId,
      patch: { is_backgrounded: true },
      session_id: target.expectedNativeSessionId,
      uuid: "00000000-0000-4000-8000-000000000004",
    });
    const settled = await internal.translateMessageToEvents({
      type: "user",
      session_id: target.expectedNativeSessionId,
      parent_tool_use_id: null,
      message: {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: target.toolUseId,
            content: "Agent is running in the background",
          },
        ],
      },
    });
    internal.activeForegroundTurnId = "turn-b";
    const child = await internal.translateMessageToEvents({
      type: "assistant",
      parent_tool_use_id: target.toolUseId,
      session_id: target.expectedNativeSessionId,
      message: {
        role: "assistant",
        model: "claude-sonnet-4-6",
        content: [{ type: "text", text: "Original child result" }],
      },
    } as SDKMessage);
    expect(child).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "provider_subagent",
          event: expect.objectContaining({
            type: "timeline",
            id: target.toolUseId,
            item: expect.objectContaining({
              type: "assistant_message",
              text: "Original child result",
            }),
          }),
        }),
      ]),
    );
    expect(await session.backgroundTask!(target)).toEqual({ status: "stale" });
    expect(backgroundTasks.mock.calls).toEqual([[target.toolUseId]]);
    const terminal = await internal.translateMessageToEvents({
      type: "system",
      subtype: "task_notification",
      task_id: target.expectedTaskId,
      tool_use_id: target.toolUseId,
      status: "completed",
      output_file: "/synthetic/child.output",
      summary: "Original child result",
      session_id: target.expectedNativeSessionId,
      uuid: "00000000-0000-4000-8000-000000000005",
    });
    const projected = projectTimelineRows({
      mode: "projected",
      rows: [...original, ...background, ...settled, ...child, ...terminal].flatMap(
        (event, index) =>
          event.type === "timeline"
            ? [
                {
                  seq: index + 1,
                  timestamp: "2026-11-25T00:00:00Z",
                  turnId: event.turnId ?? target.expectedTurnId,
                  item: event.item,
                },
              ]
            : [],
      ),
    });
    const parents = projected.filter(
      (entry) => entry.item.type === "tool_call" && entry.item.callId === target.toolUseId,
    );
    expect(parents).toHaveLength(1);
    expect(parents[0]).toMatchObject({
      turnId: target.expectedTurnId,
      item: {
        status: "completed",
        detail: expect.objectContaining({ type: "sub_agent", description: "Synthetic task" }),
        metadata: {
          nativeTask: {
            nativeSessionId: target.expectedNativeSessionId,
            turnId: target.expectedTurnId,
            toolUseId: target.toolUseId,
            taskId: target.expectedTaskId,
            state: "completed",
          },
        },
      },
    });
  },
);
