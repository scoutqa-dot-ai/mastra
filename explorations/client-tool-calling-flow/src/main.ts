/**
 * Client Tool Calling Flow Example
 *
 * Demonstrates the client tool calling pattern in Mastra:
 * 1. User sends message
 * 2. LLM calls getServerTime (server tool - auto-executed)
 * 3. LLM calls getUserConfirmation (CLIENT tool - returns control here)
 * 4. We provide tool result and call agent.generate() again
 * 5. LLM calls logGreeting (server tool - auto-executed)
 * 6. LLM sends final greeting
 *
 * Run: pnpm start
 */

import { MessageListInput } from '@mastra/core/agent/message-list';
import { createTool } from '@mastra/core/tools';
import { z } from 'zod';

import { mastra } from './mastra';

const THREAD_ID = 'demo-thread';

// CLIENT tool - runs on client side, not server
// When LLM calls this, Mastra returns control to the caller
const getUserConfirmationClientTool = createTool({
  id: 'getUserConfirmation',
  description: 'Asks the user to confirm their name. Call this after getting the server time.',
  inputSchema: z.object({
    prompt: z.string().describe('The prompt to show the user'),
  }),
  outputSchema: z.object({
    confirmed: z.boolean(),
    userName: z.string(),
  }),
  execute: async () => {
    throw new Error('Client tool should not be executed on server');
  },
});

function agentGenerate(messages: MessageListInput) {
  const agent = mastra.getAgent('toolFlowAgent');
  const memoryConfig = { thread: THREAD_ID, resource: 'demo-user' };

  return agent.generate(messages, {
    memory: memoryConfig,
    clientTools: { getUserConfirmation: getUserConfirmationClientTool },
  });
}

async function dumpStorageMessages(threadId: string) {
  console.log('\n[Storage Dump]');
  try {
    const agent = mastra.getAgent('toolFlowAgent');
    const memory = await agent.getMemory();
    if (!memory) {
      console.log('  No memory available');
      return;
    }

    const { messages } = await memory.recall({ threadId, perPage: false });
    console.log(`  ${messages.length} message(s):`);
    for (const msg of messages) {
      console.log(`  [${msg.role.toUpperCase()}] ${JSON.stringify(msg.content.parts)}`);
    }
  } catch (error) {
    console.log(`  Error: ${error}`);
  }
}

async function main() {
  console.log('='.repeat(60));
  console.log('Client Tool Calling Flow Demo');
  console.log('='.repeat(60));

  // Step 1: Initial request
  // LLM will call getServerTime (server), then getUserConfirmation (client)
  console.log('\n[Step 1] Sending initial request...');

  const result1 = await agentGenerate('Hello! Please greet me properly using all the tools.');
  console.log(`\nFinish reason: ${result1.finishReason}`);
  await dumpStorageMessages(THREAD_ID);

  // Check if LLM called the client tool
  if (result1.finishReason !== 'tool-calls' || !result1.toolCalls) {
    console.log('Unexpected: LLM did not request client tool execution');
    return;
  }

  const clientToolCall = result1.toolCalls.find(tc => tc.payload?.toolName === 'getUserConfirmation');
  if (!clientToolCall) {
    console.log('Unexpected: getUserConfirmation not in tool calls');
    return;
  }

  const { toolCallId, toolName, args } = clientToolCall.payload;
  console.log(`\nClient tool called: ${toolName}`);
  console.log(`Arguments: ${JSON.stringify(args)}`);

  // Step 2: Provide client tool result and continue
  // Build tool result message and call agent.generate() again with same thread
  console.log('\n[Step 2] Providing client tool result (Alice)...');

  const toolResultMessage = {
    role: 'tool' as const,
    content: [
      {
        type: 'tool-result' as const,
        toolCallId,
        toolName,
        result: { confirmed: true, userName: 'Alice' },
      },
    ],
  };

  const result2 = await agentGenerate(toolResultMessage);
  console.log(`\nFinish reason: ${result2.finishReason}`);
  console.log(`Response: ${result2.text}`);
  await dumpStorageMessages(THREAD_ID);

  console.log('\n' + '='.repeat(60));
  console.log('Flow completed:');
  console.log('  1. getServerTime (server tool - auto-executed)');
  console.log('  2. getUserConfirmation (client tool - we provided "Alice")');
  console.log('  3. logGreeting (server tool - auto-executed)');
  console.log('  4. Final greeting from LLM');
  console.log('='.repeat(60));
}

main().catch(console.error);
