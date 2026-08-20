import { describe, expect, it } from "vitest";
import {
  chatReducer,
  ConversationEntry,
  getCompletedHistory,
  getSafeErrorMessage,
  INITIAL_STATE,
} from "./state";
import {
  AnswerCompletedEvent,
  AnswerDeltaEvent,
  ErrorEvent,
  GenerationStartedEvent,
  RerankingStartedEvent,
  RetrievalStartedEvent,
  VerificationStartedEvent,
} from "../api/types";

describe("chatReducer and state management", () => {
  it("initializes with default state", () => {
    expect(INITIAL_STATE.entries).toEqual([]);
    expect(INITIAL_STATE.activeEntryId).toBeNull();
    expect(INITIAL_STATE.globalStatus).toBe("idle");
  });

  it("handles START_SUBMISSION correctly", () => {
    const state = chatReducer(INITIAL_STATE, {
      type: "START_SUBMISSION",
      payload: {
        entryId: "entry-1",
        userText: "What is Sheharyar's experience?",
      },
    });

    expect(state.entries.length).toBe(1);
    expect(state.entries[0].id).toBe("entry-1");
    expect(state.entries[0].userText).toBe("What is Sheharyar's experience?");
    expect(state.entries[0].status).toBe("submitting");
    expect(state.activeEntryId).toBe("entry-1");
    expect(state.globalStatus).toBe("submitting");
  });

  it("sets backend requestId via SET_REQUEST_ID", () => {
    let state = chatReducer(INITIAL_STATE, {
      type: "START_SUBMISSION",
      payload: { entryId: "entry-1", userText: "Hello" },
    });

    state = chatReducer(state, {
      type: "SET_REQUEST_ID",
      payload: { entryId: "entry-1", requestId: "req-123" },
    });

    expect(state.entries[0].requestId).toBe("req-123");
  });

  it("processes monotonic SSE events in order", () => {
    let state = chatReducer(INITIAL_STATE, {
      type: "START_SUBMISSION",
      payload: { entryId: "entry-1", userText: "Hello" },
    });

    // 1. retrieval_started
    const retEvent: RetrievalStartedEvent = {
      schema_version: "1",
      request_id: "req-1",
      sequence: 1,
      event: "retrieval_started",
      data: {},
    };
    state = chatReducer(state, {
      type: "RECEIVE_SSE_EVENT",
      payload: { entryId: "entry-1", requestId: "req-1", event: retEvent },
    });
    expect(state.entries[0].status).toBe("retrieving");
    expect(state.entries[0].lastSequence).toBe(1);

    // 2. reranking_started
    const rerankEvent: RerankingStartedEvent = {
      schema_version: "1",
      request_id: "req-1",
      sequence: 2,
      event: "reranking_started",
      data: { candidate_count: 10 },
    };
    state = chatReducer(state, {
      type: "RECEIVE_SSE_EVENT",
      payload: { entryId: "entry-1", requestId: "req-1", event: rerankEvent },
    });
    expect(state.entries[0].status).toBe("reranking");
    expect(state.entries[0].stageMetadata?.candidateCount).toBe(10);
    expect(state.entries[0].lastSequence).toBe(2);

    // 3. generation_started
    const genEvent: GenerationStartedEvent = {
      schema_version: "1",
      request_id: "req-1",
      sequence: 3,
      event: "generation_started",
      data: {},
    };
    state = chatReducer(state, {
      type: "RECEIVE_SSE_EVENT",
      payload: { entryId: "entry-1", requestId: "req-1", event: genEvent },
    });
    expect(state.entries[0].status).toBe("generating");

    // 4. answer_delta
    const deltaEvent: AnswerDeltaEvent = {
      schema_version: "1",
      request_id: "req-1",
      sequence: 4,
      event: "answer_delta",
      data: { delta: "Sheharyar is " },
    };
    state = chatReducer(state, {
      type: "RECEIVE_SSE_EVENT",
      payload: { entryId: "entry-1", requestId: "req-1", event: deltaEvent },
    });
    expect(state.entries[0].provisionalAnswer).toBe("Sheharyar is ");

    // 5. verification_started
    const verEvent: VerificationStartedEvent = {
      schema_version: "1",
      request_id: "req-1",
      sequence: 5,
      event: "verification_started",
      data: { claim_count: 3 },
    };
    state = chatReducer(state, {
      type: "RECEIVE_SSE_EVENT",
      payload: { entryId: "entry-1", requestId: "req-1", event: verEvent },
    });
    expect(state.entries[0].status).toBe("verifying");
    expect(state.entries[0].stageMetadata?.claimCount).toBe(3);

    // 6. answer_completed
    const completeEvent: AnswerCompletedEvent = {
      schema_version: "1",
      request_id: "req-1",
      sequence: 6,
      event: "answer_completed",
      data: {
        result: {
          answer: "Sheharyar is a Software Engineer.",
          citations: [
            {
              evidence_id: "E1",
              chunk_id: "chk-1",
              source: "cv.md",
              page: 1,
              excerpt: "Software Engineer",
              support_status: "supported",
            },
          ],
          claim_verifications: [],
          decision: "answered",
          reason_codes: [],
          retrieval_query: "Sheharyar experience",
          pipeline_version: "dense-v1",
        },
        replace_streamed_answer: false,
      },
    };
    state = chatReducer(state, {
      type: "RECEIVE_SSE_EVENT",
      payload: { entryId: "entry-1", requestId: "req-1", event: completeEvent },
    });
    expect(state.entries[0].status).toBe("complete");
    expect(state.entries[0].result?.decision).toBe("answered");
    expect(state.globalStatus).toBe("idle");
    expect(state.activeEntryId).toBeNull();
  });

  it("ignores duplicate, out-of-order, or late events", () => {
    let state = chatReducer(INITIAL_STATE, {
      type: "START_SUBMISSION",
      payload: { entryId: "entry-1", userText: "Hello" },
    });

    const event3: AnswerDeltaEvent = {
      schema_version: "1",
      request_id: "req-1",
      sequence: 3,
      event: "answer_delta",
      data: { delta: "Hello " },
    };

    state = chatReducer(state, {
      type: "RECEIVE_SSE_EVENT",
      payload: { entryId: "entry-1", requestId: "req-1", event: event3 },
    });
    expect(state.entries[0].provisionalAnswer).toBe("Hello ");
    expect(state.entries[0].lastSequence).toBe(3);

    // Out-of-order sequence 2 should be rejected
    const event2: AnswerDeltaEvent = {
      schema_version: "1",
      request_id: "req-1",
      sequence: 2,
      event: "answer_delta",
      data: { delta: "Ignored " },
    };
    state = chatReducer(state, {
      type: "RECEIVE_SSE_EVENT",
      payload: { entryId: "entry-1", requestId: "req-1", event: event2 },
    });
    expect(state.entries[0].provisionalAnswer).toBe("Hello ");

    // Duplicate sequence 3 should be rejected
    state = chatReducer(state, {
      type: "RECEIVE_SSE_EVENT",
      payload: { entryId: "entry-1", requestId: "req-1", event: event3 },
    });
    expect(state.entries[0].provisionalAnswer).toBe("Hello ");
  });

  it("replaces provisional answer when replace_streamed_answer is true", () => {
    let state = chatReducer(INITIAL_STATE, {
      type: "START_SUBMISSION",
      payload: { entryId: "entry-1", userText: "What is his degree?" },
    });

    const deltaEvent: AnswerDeltaEvent = {
      schema_version: "1",
      request_id: "req-1",
      sequence: 1,
      event: "answer_delta",
      data: { delta: "He holds a degree in Biology." }, // Incorrect stream
    };
    state = chatReducer(state, {
      type: "RECEIVE_SSE_EVENT",
      payload: { entryId: "entry-1", requestId: "req-1", event: deltaEvent },
    });
    expect(state.entries[0].provisionalAnswer).toBe("He holds a degree in Biology.");

    // Verification abstains and replaces answer
    const abstentionCompleteEvent: AnswerCompletedEvent = {
      schema_version: "1",
      request_id: "req-1",
      sequence: 2,
      event: "answer_completed",
      data: {
        result: {
          answer: "I do not have sufficient verified evidence to answer this question.",
          citations: [],
          claim_verifications: [],
          decision: "abstained",
          reason_codes: ["insufficient_evidence"],
          retrieval_query: "degree",
          pipeline_version: "dense-v1",
        },
        replace_streamed_answer: true,
      },
    };

    state = chatReducer(state, {
      type: "RECEIVE_SSE_EVENT",
      payload: {
        entryId: "entry-1",
        requestId: "req-1",
        event: abstentionCompleteEvent,
      },
    });

    expect(state.entries[0].provisionalAnswer).toBe(
      "I do not have sufficient verified evidence to answer this question."
    );
    expect(state.entries[0].result?.answer).toBe(
      "I do not have sufficient verified evidence to answer this question."
    );
    expect(state.entries[0].result?.decision).toBe("abstained");
  });

  it("handles terminal error event and preserves partial answer", () => {
    let state = chatReducer(INITIAL_STATE, {
      type: "START_SUBMISSION",
      payload: { entryId: "entry-1", userText: "Test question" },
    });

    const deltaEvent: AnswerDeltaEvent = {
      schema_version: "1",
      request_id: "req-1",
      sequence: 1,
      event: "answer_delta",
      data: { delta: "Partial text generated so far..." },
    };
    state = chatReducer(state, {
      type: "RECEIVE_SSE_EVENT",
      payload: { entryId: "entry-1", requestId: "req-1", event: deltaEvent },
    });

    const errorEvent: ErrorEvent = {
      schema_version: "1",
      request_id: "req-1",
      sequence: 2,
      event: "error",
      data: {
        code: "stream_interrupted",
        message: "The stream was interrupted.",
        retryable: true,
        partial_answer: "Partial text generated so far...",
      },
    };

    state = chatReducer(state, {
      type: "RECEIVE_SSE_EVENT",
      payload: { entryId: "entry-1", requestId: "req-1", event: errorEvent },
    });

    expect(state.entries[0].status).toBe("error");
    expect(state.entries[0].error?.code).toBe("stream_interrupted");
    expect(state.entries[0].error?.retryable).toBe(true);
    expect(state.entries[0].provisionalAnswer).toBe(
      "Partial text generated so far..."
    );
    expect(state.globalStatus).toBe("idle");
    expect(state.activeEntryId).toBeNull();
  });

  it("clears conversation and resets state on CLEAR_CONVERSATION", () => {
    let state = chatReducer(INITIAL_STATE, {
      type: "START_SUBMISSION",
      payload: { entryId: "entry-1", userText: "Question 1" },
    });

    state = chatReducer(state, { type: "CLEAR_CONVERSATION" });
    expect(state.entries).toEqual([]);
    expect(state.activeEntryId).toBeNull();
    expect(state.globalStatus).toBe("idle");
  });

  it("getCompletedHistory extracts only completed turns and bounds to limit", () => {
    const dummyEntries: ConversationEntry[] = [
      {
        id: "1",
        requestId: "r1",
        lastSequence: 1,
        userText: "Turn 1",
        provisionalAnswer: "Ans 1",
        result: {
          answer: "Ans 1",
          citations: [],
          claim_verifications: [],
          decision: "answered",
          reason_codes: [],
          retrieval_query: "q1",
          pipeline_version: "v1",
        },
        status: "complete",
        error: null,
        createdAt: 1,
      },
      {
        id: "2",
        requestId: "r2",
        lastSequence: 1,
        userText: "Turn 2 In Progress",
        provisionalAnswer: "Partial...",
        result: null,
        status: "generating",
        error: null,
        createdAt: 2,
      },
      {
        id: "3",
        requestId: "r3",
        lastSequence: 1,
        userText: "Turn 3 Failed",
        provisionalAnswer: "",
        result: null,
        status: "error",
        error: { code: "internal_error", message: "fail", retryable: false },
        createdAt: 3,
      },
      {
        id: "4",
        requestId: "r4",
        lastSequence: 1,
        userText: "Turn 4",
        provisionalAnswer: "Ans 4",
        result: {
          answer: "Ans 4",
          citations: [],
          claim_verifications: [],
          decision: "answered",
          reason_codes: [],
          retrieval_query: "q4",
          pipeline_version: "v1",
        },
        status: "complete",
        error: null,
        createdAt: 4,
      },
    ];

    const history = getCompletedHistory(dummyEntries);
    expect(history).toEqual([
      { user: "Turn 1", assistant: "Ans 1" },
      { user: "Turn 4", assistant: "Ans 4" },
    ]);
  });

  it("maps error codes to safe user messages according to Spec 06 §9", () => {
    expect(getSafeErrorMessage("invalid_question")).toContain("valid question");
    expect(getSafeErrorMessage("provider_unavailable")).toContain(
      "temporarily unavailable"
    );
    expect(getSafeErrorMessage("pipeline_not_ready")).toContain(
      "initializing or temporarily offline"
    );
    expect(getSafeErrorMessage("stream_interrupted")).toContain("interrupted");
    expect(getSafeErrorMessage("unknown_secret_code")).toContain(
      "unexpected error occurred"
    );
  });
});
