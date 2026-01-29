import { openai } from '@ai-sdk/openai-v5';
import { wrapLanguageModel, type LanguageModelMiddleware } from 'ai-v5';
import { Agent } from '@mastra/core/agent';
import { Mastra } from '@mastra/core/mastra';
import { createTool } from '@mastra/core/tools';
import { LibSQLStore } from '@mastra/libsql';
import { Memory } from '@mastra/memory';
import { z } from 'zod';

// Middleware to log messages sent to OpenAI
const logMiddleware: LanguageModelMiddleware = {
  wrapGenerate: async ({ doGenerate, params }: any) => {
    console.log('\n[OpenAI Request]');
    console.log(JSON.stringify(params.prompt, null, 2));
    const result = await doGenerate();
    console.log('\n[OpenAI Response]');
    console.log(JSON.stringify(result.content, null, 2));
    return result;
  },
};

// Server tool #1: Gets the current server time
const getServerTimeTool = createTool({
  id: 'getServerTime',
  description: 'Gets the current server time.',
  inputSchema: z.object({
    timezone: z.string().nullable().optional().describe('Optional timezone'),
  }),
  outputSchema: z.object({
    time: z.string(),
    timezone: z.string(),
  }),
  execute: async input => {
    const timezone = input.timezone || 'UTC';
    const time = new Date().toLocaleString('en-US', { timeZone: timezone });
    console.log(`[Server] getServerTime: ${time} (${timezone})`);
    return { time, timezone };
  },
});

// Server tool #2: Logs a greeting
const logGreetingTool = createTool({
  id: 'logGreeting',
  description: 'Logs a greeting message to the server.',
  inputSchema: z.object({
    userName: z.string().describe('The name of the user to greet'),
    greeting: z.string().describe('The greeting message'),
  }),
  outputSchema: z.object({
    logged: z.boolean(),
    logMessage: z.string(),
  }),
  execute: async input => {
    const logMessage = `${input.greeting}, ${input.userName}!`;
    console.log(`[Server] logGreeting: ${logMessage}`);
    return { logged: true, logMessage };
  },
});

const toolFlowAgent = new Agent({
  id: 'tool-flow-agent',
  name: 'Tool Flow Agent',
  instructions: `You are a helpful assistant that demonstrates the client tool calling flow.

When a user sends a message, you MUST follow this EXACT sequence:
1. FIRST, call the getServerTime tool to get the current time
2. SECOND, call the getUserConfirmation client tool to ask the user to confirm their name
3. THIRD, after receiving the user's name, call the logGreeting tool to log a greeting
4. FINALLY, respond with a friendly hello message that includes the time and their name

IMPORTANT: You must call the tools in this exact order: getServerTime -> getUserConfirmation -> logGreeting -> final response.
Do not skip any steps.`,
  model: wrapLanguageModel({ model: openai('gpt-4o'), middleware: logMiddleware }),
  tools: {
    getServerTime: getServerTimeTool,
    logGreeting: logGreetingTool,
  },
  memory: new Memory(),
});

export const mastra = new Mastra({
  agents: { toolFlowAgent },
  storage: new LibSQLStore({
    id: 'client-tool-calling-flow',
    url: ':memory:',
  }),
});
