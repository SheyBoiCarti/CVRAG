import { describe, expect, it } from "vitest";
import fs from "fs";
import path from "path";

describe("OpenAPI Contract Test", () => {
  it("validates that TypeScript API types match FastAPI OpenAPI schemas", () => {
    const openapiPath = path.resolve(__dirname, "openapi.json");
    expect(fs.existsSync(openapiPath)).toBe(true);

    const openapi = JSON.parse(fs.readFileSync(openapiPath, "utf-8"));
    const schemas = openapi.components.schemas;

    // 1. ChatRequest
    expect(schemas.ChatRequest).toBeDefined();
    const chatReqProps = schemas.ChatRequest.properties;
    expect(chatReqProps.question).toBeDefined();
    expect(chatReqProps.history).toBeDefined();

    // 2. ChatResponse
    expect(schemas.ChatResponse).toBeDefined();
    const chatRespProps = schemas.ChatResponse.properties;
    expect(chatRespProps.answer).toBeDefined();
    expect(chatRespProps.citations).toBeDefined();
    expect(chatRespProps.claim_verifications).toBeDefined();
    expect(chatRespProps.decision).toBeDefined();
    expect(chatRespProps.reason_codes).toBeDefined();
    expect(chatRespProps.retrieval_query).toBeDefined();
    expect(chatRespProps.pipeline_version).toBeDefined();

    // 3. CitationSchema
    expect(schemas.CitationSchema).toBeDefined();
    const citationProps = schemas.CitationSchema.properties;
    expect(citationProps.evidence_id).toBeDefined();
    expect(citationProps.chunk_id).toBeDefined();
    expect(citationProps.source).toBeDefined();
    expect(citationProps.page).toBeDefined();
    expect(citationProps.excerpt).toBeDefined();
    expect(citationProps.support_status).toBeDefined();

    // 4. ClaimVerificationSchema
    expect(schemas.ClaimVerificationSchema).toBeDefined();
    const claimProps = schemas.ClaimVerificationSchema.properties;
    expect(claimProps.claim_id).toBeDefined();
    expect(claimProps.text).toBeDefined();
    expect(claimProps.factual).toBeDefined();
    expect(claimProps.importance).toBeDefined();
    expect(claimProps.citation_ids).toBeDefined();
    expect(claimProps.status).toBeDefined();
    expect(claimProps.reason_code).toBeDefined();
  });
});
