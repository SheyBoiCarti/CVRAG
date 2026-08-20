import { describe, expect, it, vi } from "vitest";
import { parseSSEStream, sendChatMessage, streamChatMessage } from "./chat";
import { PipelineSSEEvent } from "./types";

describe("API chat client and SSE streaming", () => {
  it("parses single and multiple SSE frames correctly", async () => {
    const events: PipelineSSEEvent[] = [];
    const encoder = new TextEncoder();

    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(
          encoder.encode(
            'data: {"schema_version":"1","request_id":"r1","sequence":1,"event":"retrieval_started","data":{}}\n\n' +
              'data: {"schema_version":"1","request_id":"r1","sequence":2,"event":"answer_delta","data":{"delta":"Hello"}}\n\n'
          )
        );
        controller.close();
      },
    });

    await parseSSEStream(stream, (evt) => events.push(evt));

    expect(events.length).toBe(2);
    expect(events[0].event).toBe("retrieval_started");
    expect(events[1].event).toBe("answer_delta");
    if (events[1].event === "answer_delta") {
      expect(events[1].data.delta).toBe("Hello");
    }
  });

  it("handles split SSE frames across multiple chunk boundaries", async () => {
    const events: PipelineSSEEvent[] = [];
    const encoder = new TextEncoder();

    const chunk1 = 'data: {"schema_version":"1","request_id":"r1","sequence":1,"ev';
    const chunk2 = 'ent":"answer_delta","data":{"delta":"Split ';
    const chunk3 = 'Frame"}}\n\n';

    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(chunk1));
        controller.enqueue(encoder.encode(chunk2));
        controller.enqueue(encoder.encode(chunk3));
        controller.close();
      },
    });

    await parseSSEStream(stream, (evt) => events.push(evt));

    expect(events.length).toBe(1);
    expect(events[0].event).toBe("answer_delta");
    if (events[0].event === "answer_delta") {
      expect(events[0].data.delta).toBe("Split Frame");
    }
  });

  it("decodes multi-byte UTF-8 characters across chunk boundaries correctly", async () => {
    const events: PipelineSSEEvent[] = [];
    const encoder = new TextEncoder();

    // The emoji 🎓 is 4 bytes in UTF-8: 0xF0, 0x9F, 0x8E, 0x93
    // We split the byte array right in the middle of this multi-byte character!
    const jsonPrefix =
      'data: {"schema_version":"1","request_id":"r1","sequence":1,"event":"answer_delta","data":{"delta":"Degree: ';
    const jsonSuffix = '"}}\n\n';

    const fullBytes = encoder.encode(jsonPrefix + "🎓" + jsonSuffix);
    // Find the position where 🎓 starts in fullBytes
    const prefixBytes = encoder.encode(jsonPrefix);
    const splitIndex = prefixBytes.length + 2; // Split 2 bytes into the 4-byte emoji

    const chunk1 = fullBytes.slice(0, splitIndex);
    const chunk2 = fullBytes.slice(splitIndex);

    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(chunk1);
        controller.enqueue(chunk2);
        controller.close();
      },
    });

    await parseSSEStream(stream, (evt) => events.push(evt));

    expect(events.length).toBe(1);
    expect(events[0].event).toBe("answer_delta");
    if (events[0].event === "answer_delta") {
      expect(events[0].data.delta).toBe("Degree: 🎓");
    }
  });

  it("handles nonstreaming sendChatMessage successfully", async () => {
    const mockResult = {
      answer: "Sheharyar has 5+ years experience.",
      citations: [],
      claim_verifications: [],
      decision: "answered" as const,
      reason_codes: [],
      retrieval_query: "experience",
      pipeline_version: "dense-v1",
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResult,
    });

    const res = await sendChatMessage("What is your experience?", []);
    expect(res.answer).toBe("Sheharyar has 5+ years experience.");
    expect(global.fetch).toHaveBeenCalledWith(
      "/api/chat?stream=false",
      expect.objectContaining({
        method: "POST",
      })
    );
  });

  it("handles nonstreaming sendChatMessage error", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({
        error: {
          code: "pipeline_not_ready",
          message: "Pipeline initializing",
          retryable: true,
        },
      }),
    });

    await expect(sendChatMessage("Test")).rejects.toMatchObject({
      code: "pipeline_not_ready",
      status: 503,
      retryable: true,
    });
  });

  it("handles streamChatMessage abort signal gracefully", async () => {
    const onEvent = vi.fn();
    const onError = vi.fn();
    const controller = new AbortController();

    global.fetch = vi.fn().mockImplementation(() => {
      controller.abort();
      return Promise.reject(new DOMException("Aborted", "AbortError"));
    });

    await streamChatMessage({
      question: "Hello",
      onEvent,
      onError,
      signal: controller.signal,
    });

    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({
        code: "stream_interrupted",
      })
    );
  });
});
