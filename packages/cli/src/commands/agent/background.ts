import { Command } from "commander";
import { connectToDaemon } from "../../utils/client.js";
import type { CommandOptions, OutputSchema, SingleResult } from "../../output/index.js";
import {
  AgentBackgroundTaskTargetSchema,
  type AgentBackgroundTaskResult,
} from "@getpaseo/protocol/messages";

interface AgentBackgroundOptions extends CommandOptions {
  nativeSession?: string;
  turn?: string;
  tool?: string;
  task?: string;
}

interface AgentBackgroundCommandResult {
  agentId: string;
  result: AgentBackgroundTaskResult;
}

const backgroundSchema: OutputSchema<AgentBackgroundCommandResult> = {
  idField: "agentId",
  columns: [
    { header: "STATUS", field: ({ result }) => result.status },
    {
      header: "ERROR",
      field: ({ result }) => (result.status === "unconfirmed" ? result.error : ""),
    },
  ],
};

export function addBackgroundOptions(command: Command): Command {
  return command
    .description("Request background execution of one active native task without stopping it")
    .argument("<id>", "Exact agent ID")
    .requiredOption("--native-session <id>", "Expected native session ID")
    .requiredOption("--turn <id>", "Expected active turn ID")
    .requiredOption("--tool <id>", "Exact native tool-use ID")
    .requiredOption("--task <id>", "Observed native task ID");
}

export async function runBackgroundCommand(
  id: string,
  options: AgentBackgroundOptions,
  _command: Command,
): Promise<SingleResult<AgentBackgroundCommandResult>> {
  const target = AgentBackgroundTaskTargetSchema.parse({
    expectedTaskId: options.task,
    expectedNativeSessionId: options.nativeSession,
    expectedTurnId: options.turn,
    toolUseId: options.tool,
  });
  const client = await connectToDaemon({ target: options.daemonTarget });
  try {
    const result = await client.backgroundAgentTask(id, target);
    return { type: "single", data: { agentId: id, result }, schema: backgroundSchema };
  } finally {
    await client.close();
  }
}
