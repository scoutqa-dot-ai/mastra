import { describe, it, expect } from 'vitest';
import { MessageList } from '../index';
import type { MastraDBMessage } from '../state/types';

/**
 * Test for issue where tool call args are lost when a tool invocation is split
 * across two messages loaded from memory/storage:
 * - Message 1: tool-invocation with state='call' and args
 * - Message 2: tool-invocation with state='result' but args={} (empty)
 *
 * When loading these messages from storage and sending to LLM, the tool-result
 * should have the original args from the 'call' state, not the empty args from
 * the 'result' state.
 *
 * Real-world scenario: Agent generates a client tool call. Server cannot execute
 * client tools, so it saves the message with state='call'. Client executes the
 * tool and sends result back, this is saved as a separate message with empty args.
 */
describe('MessageList - Split tool call args across messages', () => {
  it('should recover tool call args when state=call and state=result are in different messages', () => {
    const messageList = new MessageList();

    // Message 1: tool call with state='call' and args
    const message1: MastraDBMessage = {
      id: 'msg-1',
      role: 'assistant',
      createdAt: new Date('2024-01-01T00:00:00Z'),
      content: {
        format: 2,
        parts: [
          {
            type: 'tool-invocation',
            toolInvocation: {
              state: 'call',
              toolCallId: 'call-1',
              toolName: 'test_tool',
              args: { foo: 'bar' },
            },
          },
        ],
      },
    };

    // Message 2: tool result with state='result' but empty args
    const message2: MastraDBMessage = {
      id: 'msg-2',
      role: 'assistant',
      createdAt: new Date('2024-01-01T00:00:01Z'),
      content: {
        format: 2,
        parts: [
          {
            type: 'tool-invocation',
            toolInvocation: {
              state: 'result',
              toolCallId: 'call-1',
              toolName: 'test_tool',
              result: { success: true },
              args: {}, // Empty - this is the problem!
            },
          },
        ],
      },
    };

    messageList.add(message1, 'memory');
    messageList.add(message2, 'memory');

    const modelMessages = messageList.get.all.aiV5.model();
    const toolResultMsg = modelMessages.find(
      msg => msg.role === 'tool' && Array.isArray(msg.content) && msg.content.length > 0,
    );

    const content = toolResultMsg?.content;
    const contentArray = Array.isArray(content) ? content : [];
    const toolResultPart = contentArray.find(part => part.type === 'tool-result');
    // @ts-expect-error - input field exists on StaticToolResult
    expect(toolResultPart?.input).toEqual({ foo: 'bar' });
  });
});
