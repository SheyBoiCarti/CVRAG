import {
  ChatRequestPayload,
  ChatResult,
  ChatTurn,
  PipelineSSEEvent,
} from "./types";

export class ApiError extends Error {
  readonly code: string;
  readonly retryable: boolean;
  readonly status: number;
  readonly details?: Record<string, unknown>;

  constructor(
    message: string,
    code: string = "internal_error",
    status: number = 500,
    retryable: boolean = false,
    details?: Record<string, unknown>
  ) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.retryable = retryable;
    this.details = details;
  }
}

/**
 * Synchronous non-streaming API helper for diagnostics and fallback.
 */
export async function sendChatMessage(
  question: string,
  history: ChatTurn[] = [],
  signal?: AbortSignal
): Promise<ChatResult> {
  const payload: ChatRequestPayload = {
    question,
    history,
  };

  let response: Response;
  try {
    response = await fetch("/api/chat?stream=false", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (err: unknown) {
    if (signal?.aborted) {
      throw new ApiError("Request cancelled", "stream_interrupted", 499, false);
    }
    throw new ApiError(
      "Failed to connect to chat service",
      "provider_unavailable",
      503,
      true
    );
  }

  if (!response.ok) {
    let errorCode = "internal_error";
    let errorMessage = "An error occurred while processing the request.";
    let retryable = false;
    let details: Record<string, unknown> | undefined;

    try {
      const errJson = await response.json();
      if (errJson?.error) {
        errorCode = errJson.error.code || errorCode;
        errorMessage = errJson.error.message || errorMessage;
        retryable = Boolean(errJson.error.retryable);
        details = errJson.error.details;
      }
    } catch {
      // Use fallback error info
    }

    throw new ApiError(
      errorMessage,
      errorCode,
      response.status,
      retryable,
      details
    );
  }

  return (await response.json()) as ChatResult;
}

export interface StreamChatOptions {
  question: string;
  history?: ChatTurn[];
  onEvent: (event: PipelineSSEEvent) => void;
  onError?: (error: ApiError) => void;
  signal?: AbortSignal;
}

/**
 * Parses Server-Sent Events from a ReadableStream using an SSE line buffer and UTF-8 stream decoder.
 * Handles split multi-byte UTF-8 sequences and multi-chunk SSE frames correctly.
 */
export async function parseSSEStream(
  stream: ReadableStream<Uint8Array>,
  onEvent: (event: PipelineSSEEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      if (signal?.aborted) {
        await reader.cancel();
        break;
      }

      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      // decode with stream: true to properly handle multi-byte characters across chunk boundaries
      buffer += decoder.decode(value, { stream: true });

      // Split into lines while maintaining buffer remainder
      const lines = buffer.split("\n");
      // The last element is either an empty string (if buffer ended with \n) or an incomplete line
      buffer = lines.pop() || "";

      let currentEventData = "";

      for (const line of lines) {
        const trimmed = line.trim();

        if (trimmed === "") {
          // Empty line indicates end of an SSE message block
          if (currentEventData) {
            try {
              const parsed = JSON.parse(currentEventData) as PipelineSSEEvent;
              onEvent(parsed);
            } catch (parseErr) {
              console.warn("Failed to parse SSE JSON payload:", currentEventData, parseErr);
            }
            currentEventData = "";
          }
          continue;
        }

        if (trimmed.startsWith("data:")) {
          const dataContent = trimmed.substring(5).trim();
          if (currentEventData) {
            currentEventData += "\n" + dataContent;
          } else {
            currentEventData = dataContent;
          }
        }
      }

      // If trailing data block was not terminated with empty line before EOF
      if (currentEventData) {
        try {
          const parsed = JSON.parse(currentEventData) as PipelineSSEEvent;
          onEvent(parsed);
        } catch {
          // Incomplete payload, let subsequent chunks append
        }
      }
    }

    // Flush any remaining decoder bytes
    const remaining = decoder.decode();
    if (remaining) {
      buffer += remaining;
    }

    if (buffer.trim().startsWith("data:")) {
      const dataContent = buffer.trim().substring(5).trim();
      try {
        const parsed = JSON.parse(dataContent) as PipelineSSEEvent;
        onEvent(parsed);
      } catch {
        // Ignored
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Server-Sent Events streaming chat client.
 * POSTs question and history to same-origin /api/chat proxy.
 */
export async function streamChatMessage({
  question,
  history = [],
  onEvent,
  onError,
  signal,
}: StreamChatOptions): Promise<void> {
  const payload: ChatRequestPayload = {
    question,
    history,
  };

  let response: Response;
  try {
    response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (err: unknown) {
    if (signal?.aborted) {
      const abortError = new ApiError(
        "Request cancelled",
        "stream_interrupted",
        499,
        false
      );
      onError?.(abortError);
      return;
    }
    const netError = new ApiError(
      "Failed to connect to chat service",
      "provider_unavailable",
      503,
      true
    );
    onError?.(netError);
    return;
  }

  if (!response.ok) {
    let errorCode = "internal_error";
    let errorMessage = "An error occurred while processing the request.";
    let retryable = false;
    let details: Record<string, unknown> | undefined;

    try {
      const errJson = await response.json();
      if (errJson?.error) {
        errorCode = errJson.error.code || errorCode;
        errorMessage = errJson.error.message || errorMessage;
        retryable = Boolean(errJson.error.retryable);
        details = errJson.error.details;
      }
    } catch {
      // Plain text or unparseable error
    }

    const apiError = new ApiError(
      errorMessage,
      errorCode,
      response.status,
      retryable,
      details
    );
    onError?.(apiError);
    return;
  }

  if (!response.body) {
    const bodyError = new ApiError(
      "No response body received from server",
      "internal_error",
      500,
      false
    );
    onError?.(bodyError);
    return;
  }

  try {
    await parseSSEStream(response.body, onEvent, signal);
  } catch (streamErr: unknown) {
    if (signal?.aborted) {
      const abortError = new ApiError(
        "Request cancelled",
        "stream_interrupted",
        499,
        false
      );
      onError?.(abortError);
      return;
    }

    const err = new ApiError(
      "Stream processing was interrupted",
      "stream_interrupted",
      500,
      true
    );
    onError?.(err);
  }
}
