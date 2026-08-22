import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Same-origin API route proxying chat requests to the FastAPI backend.
 * Uses server-only runtime environment variable BACKEND_URL.
 */
export async function POST(req: NextRequest): Promise<Response> {
  const backendUrl =
    process.env.BACKEND_URL || "http://127.0.0.1:8000";

  const url = new URL(req.url);
  const isStream = url.searchParams.get("stream") !== "false";

  let bodyText: string;
  try {
    bodyText = await req.text();
  } catch {
    return NextResponse.json(
      {
        error: {
          code: "invalid_question",
          message: "Malformed request payload.",
          retryable: false,
        },
      },
      { status: 422 }
    );
  }

  const endpoint = isStream
    ? `${backendUrl}/api/v1/chat/stream`
    : `${backendUrl}/api/v1/chat`;

  const incomingRequestId =
    req.headers.get("x-request-id") ||
    `req-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;

  try {
    const backendResponse = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: isStream ? "text/event-stream" : "application/json",
        "x-request-id": incomingRequestId,
      },
      body: bodyText,
      // @ts-expect-error Node fetch duplex option for streaming
      duplex: isStream ? "half" : undefined,
    });

    if (!backendResponse.ok) {
      const responseContentType =
        backendResponse.headers.get("content-type") || "";
      if (responseContentType.includes("application/json")) {
        const errorJson = await backendResponse.json();
        return NextResponse.json(errorJson, {
          status: backendResponse.status,
        });
      }

      return NextResponse.json(
        {
          error: {
            code: "provider_unavailable",
            message: "The backend service returned an error.",
            retryable: backendResponse.status >= 500,
          },
        },
        { status: backendResponse.status }
      );
    }

    if (isStream) {
      return new Response(backendResponse.body, {
        status: 200,
        headers: {
          "Content-Type": "text/event-stream; charset=utf-8",
          "Cache-Control": "no-cache, no-transform",
          Connection: "keep-alive",
          "X-Accel-Buffering": "no",
          "x-request-id": incomingRequestId,
        },
      });
    }

    const data = await backendResponse.json();
    return NextResponse.json(data, { status: 200 });
  } catch (fetchError: unknown) {
    console.error("Backend proxy request failed:", fetchError);
    return NextResponse.json(
      {
        error: {
          code: "provider_unavailable",
          message:
            "Unable to connect to the backend service. Please ensure the backend is running.",
          retryable: true,
        },
      },
      { status: 503 }
    );
  }
}
