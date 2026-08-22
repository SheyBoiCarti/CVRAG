/**
 * API types mirroring the backend schema exactly according to Spec 06 & Spec 00.
 */

export type PipelineDecision = "answered" | "abstained" | "degraded";

export type EvidenceSupportStatus =
  | "supported"
  | "unsupported"
  | "not_checked";

export type ClaimVerificationStatus =
  | EvidenceSupportStatus
  | "uncited"
  | "invalid_citation";

export type ClaimImportance = "core" | "optional";

export interface ChatTurn {
  user: string;
  assistant: string;
}

export interface Citation {
  evidence_id: string;
  chunk_id: string;
  source: string;
  page: number;
  excerpt: string;
  support_status: EvidenceSupportStatus;
}

export interface ClaimVerification {
  claim_id: string;
  text: string;
  factual: boolean;
  importance: ClaimImportance;
  citation_ids: string[];
  status: ClaimVerificationStatus;
  reason_code: string;
}

export interface ChatResult {
  answer: string;
  citations: Citation[];
  claim_verifications: ClaimVerification[];
  decision: PipelineDecision;
  reason_codes: string[];
  retrieval_query: string;
  pipeline_version: string;
}

export interface ChatRequestPayload {
  question: string;
  history?: ChatTurn[];
}

export interface ErrorDetail {
  code: string;
  message: string;
  retryable: boolean;
  details?: Record<string, unknown>;
}

export interface ApiErrorResponse {
  error: ErrorDetail;
}

// SSE Event types
export interface BaseSSEEvent<E extends string, D> {
  schema_version: string;
  request_id: string;
  sequence: number;
  event: E;
  data: D;
}

export type RetrievalStartedEvent = BaseSSEEvent<
  "retrieval_started",
  Record<string, unknown>
>;

export type RetrievalCompletedEvent = BaseSSEEvent<
  "retrieval_completed",
  {
    evidence_count: number;
    degraded_reason_codes?: string[];
  }
>;

export type RerankingStartedEvent = BaseSSEEvent<
  "reranking_started",
  {
    candidate_count: number;
  }
>;

export type RerankingCompletedEvent = BaseSSEEvent<
  "reranking_completed",
  {
    evidence_count: number;
    degraded_reason_codes?: string[];
  }
>;

export type GenerationStartedEvent = BaseSSEEvent<
  "generation_started",
  Record<string, unknown>
>;

export type AnswerDeltaEvent = BaseSSEEvent<
  "answer_delta",
  {
    delta: string;
  }
>;

export type VerificationStartedEvent = BaseSSEEvent<
  "verification_started",
  {
    claim_count: number | null;
  }
>;

export type AnswerCompletedEvent = BaseSSEEvent<
  "answer_completed",
  {
    result: ChatResult;
    replace_streamed_answer: boolean;
  }
>;

export type ErrorEvent = BaseSSEEvent<
  "error",
  {
    code: string;
    message: string;
    retryable: boolean;
    partial_answer: string | null;
  }
>;

export type PipelineSSEEvent =
  | RetrievalStartedEvent
  | RetrievalCompletedEvent
  | RerankingStartedEvent
  | RerankingCompletedEvent
  | GenerationStartedEvent
  | AnswerDeltaEvent
  | VerificationStartedEvent
  | AnswerCompletedEvent
  | ErrorEvent;
