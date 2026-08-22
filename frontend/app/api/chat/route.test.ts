import { describe, expect, it, vi } from "vitest";
import { POST } from "./route";
import { NextRequest } from "next/server";

describe("Next.js API route /api/chat", () => {
  it("proxies streaming request to FastAPI backend with correct headers", async () => {
    const mockBackendStream = new ReadableStream({
      start(controller) {
        controller.enqueue(
          new TextEncoder().encode(
            'data: {"schema_version":"1","request_id":"r1","sequence":1,"event":"generation_started","data":{}}\n\n'
          )
        );
        controller.close();
      },
    });

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers({
        "content-type": "text/event-stream",
      }),
      body: mockBackendStream,
    });

    const req = new NextRequest("http://localhost:3000/api/chat", {
      method: "POST",
      body: JSON.stringify({ question: "Hello" }),
      headers: {
        "content-type": "application/json",
      },
    });

    const response = await POST(req);

    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain("text/event-stream");
    expect(response.headers.get("x-accel-buffering")).toBe("no");
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/v1/chat/stream"),
      expect.objectContaining({
        method: "POST",
      })
    );
  });

  it("handles backend offline/unreachable failure with safe 503 response", async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error("Connection refused"));

    const req = new NextRequest("http://localhost:3000/api/chat", {
      method: "POST",
      body: JSON.stringify({ question: "Hello" }),
      headers: {
        "content-type": "application/json",
      },
    });

    const response = await POST(req);
    expect(response.status).toBe(503);

    const json = await response.json();
    expect(json.error.code).toBe("provider_unavailable");
    expect(json.error.retryable).toBe(true);
  });
});
