import { test, expect } from "@playwright/test";

test.describe("CV RAG Next.js Frontend Integration", () => {
  test.beforeEach(async ({ page }) => {
    // Mock the same-origin /api/chat streaming endpoint
    await page.route("/api/chat*", async (route) => {
      const request = route.request();
      if (request.method() === "POST") {
        const sseBody = [
          'data: {"schema_version":"1","request_id":"req-e2e","sequence":1,"event":"retrieval_started","data":{}}\n\n',
          'data: {"schema_version":"1","request_id":"req-e2e","sequence":2,"event":"generation_started","data":{}}\n\n',
          'data: {"schema_version":"1","request_id":"req-e2e","sequence":3,"event":"answer_delta","data":{"delta":"Sheharyar Nadeem is a Software Engineer."}}\n\n',
          'data: {"schema_version":"1","request_id":"req-e2e","sequence":4,"event":"verification_started","data":{"claim_count":1}}\n\n',
          'data: {"schema_version":"1","request_id":"req-e2e","sequence":5,"event":"answer_completed","data":{"result":{"answer":"Sheharyar Nadeem is a Software Engineer specializing in AI and backend systems.","citations":[{"evidence_id":"E1","chunk_id":"chk-1","source":"cv.md","page":1,"excerpt":"Software Engineer & AI Developer.","support_status":"supported"}],"claim_verifications":[{"claim_id":"C1","text":"Software Engineer","factual":true,"importance":"core","citation_ids":["E1"],"status":"supported","reason_code":""}],"decision":"answered","reason_codes":[],"retrieval_query":"Sheharyar experience","pipeline_version":"dense-v1"},"replace_streamed_answer":true}}\n\n',
        ].join("");

        await route.fulfill({
          status: 200,
          contentType: "text/event-stream; charset=utf-8",
          headers: {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
          },
          body: sseBody,
        });
      } else {
        await route.continue();
      }
    });
  });

  test("loads homepage, submits a suggestion, streams answer and verifies citations", async ({
    page,
  }) => {
    await page.goto("/");

    // Verify Header & Sidebar info
    await expect(page.getByRole("heading", { name: "Sheharyar Nadeem", level: 1 })).toBeVisible();
    await expect(page.getByText("Software Engineer & AI Developer")).toBeVisible();

    // Click suggestion
    const suggestionBtn = page.getByRole("button", { name: /Ask suggestion: Professional Summary/i });
    await expect(suggestionBtn).toBeVisible();
    await suggestionBtn.click();

    // Verify user message appeared in feed
    await expect(
      page.getByText("Give me a summary of Sheharyar's professional experience and key strengths.")
    ).toBeVisible();

    // Verify final answered response with verification replacement
    await expect(
      page.getByText("Sheharyar Nadeem is a Software Engineer specializing in AI and backend systems.")
    ).toBeVisible();

    // Verify citation section exists and expand it
    await expect(page.getByTestId("citation-list")).toBeVisible();
    await page.getByText(/Grounded RAG Context/i).click();
    await expect(page.getByText(/Citation \[E1\]/i)).toBeVisible();
    await expect(page.getByText("Supported")).toBeVisible();
  });

  test("clear conversation resets the feed back to suggestion grid", async ({ page }) => {
    await page.goto("/");

    // Submit a question
    const input = page.getByTestId("chat-input");
    await input.fill("What are his skills?");
    await page.getByTestId("send-button").click();

    await expect(page.getByText("What are his skills?")).toBeVisible();

    // Click clear conversation
    const clearBtn = page.getByTestId("clear-chat-feed-button");
    await expect(clearBtn).toBeVisible();
    await clearBtn.click();

    // Suggestion grid should be visible again
    await expect(
      page.getByText("Ask about my experience, skills & background")
    ).toBeVisible();
  });

  test("responsive mobile layout has no horizontal overflow at 375px width", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto("/");

    // Verify no horizontal overflow
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);
    expect(scrollWidth).toBeLessThanOrEqual(clientWidth + 2); // 1-2px rounding tolerance

    // Menu button is visible on mobile
    const menuBtn = page.getByRole("button", { name: /Open sidebar menu/i });
    await expect(menuBtn).toBeVisible();
  });
});
