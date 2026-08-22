"""Optional NVIDIA LLM Judge adapter with temperature zero and disk caching."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from evaluation.judges.base import JudgeCacheManager, JudgeResult
from evaluation.schema import GoldenCase
from rag.settings import Settings, get_settings

LOGGER = logging.getLogger(__name__)

JUDGE_PROMPT_VERSION = "2026-08-18-v1"

JUDGE_SYSTEM_PROMPT = """You are an impartial, strict evaluation judge measuring the quality of answers produced by a RAG system.
Given the User Question, Reference Answer, Grounding Evidence, and Candidate Answer, evaluate the Candidate Answer on three dimensions:
1. correctness (0.0 to 1.0): Does the candidate answer accurately convey the key facts in the reference answer?
2. faithfulness (0.0 to 1.0): Are all claims in the candidate answer supported by the grounding evidence, without hallucinating unsupported facts?
3. citation_support (0.0 to 1.0): Are explicit evidence citations [S#] valid and supporting their claims?

If the candidate answer is an abstention ("I do not have enough information"), and the question is unanswerable from evidence, correctness and faithfulness are 1.0. If the question was answerable but it abstained, correctness is 0.0.

Output ONLY valid JSON with this exact schema:
{
  "correctness": <float 0.0-1.0>,
  "faithfulness": <float 0.0-1.0>,
  "citation_support": <float 0.0-1.0>,
  "rationale": "<1-2 sentence explanation>"
}"""


class NvidiaJudge:
    """Judge adapter powered by NVIDIA AI Endpoints (meta/llama-3.3-70b-instruct)."""

    def __init__(
        self,
        settings: Settings | None = None,
        cache_manager: JudgeCacheManager | None = None,
        model_name: str = "meta/llama-3.3-70b-instruct",
        prompt_version: str = JUDGE_PROMPT_VERSION,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache_manager = cache_manager or JudgeCacheManager()
        self.model_name = model_name
        self.prompt_version = prompt_version
        self._llm = None

    def _get_llm(self) -> Any:
        if self._llm is None:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA

            if not self.settings.NVIDIA_API_KEY:
                raise ValueError("NVIDIA_API_KEY is required to run the live NvidiaJudge.")
            self._llm = ChatNVIDIA(
                model=self.model_name,
                api_key=self.settings.NVIDIA_API_KEY,
                temperature=0.0,
                max_tokens=512,
            )
        return self._llm

    @staticmethod
    def _parse_judge_response(raw: str) -> dict[str, Any]:
        """Extracts JSON object from markdown or raw text response."""
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return {
                    "correctness": float(data.get("correctness", 0.0)),
                    "faithfulness": float(data.get("faithfulness", 0.0)),
                    "citation_support": float(data.get("citation_support", 0.0)),
                    "rationale": str(data.get("rationale", "")),
                }
            except Exception as e:
                return {"error": f"JSON parse error: {e}", "correctness": None, "faithfulness": None, "citation_support": None, "rationale": raw}
        return {"error": "No JSON block found in response", "correctness": None, "faithfulness": None, "citation_support": None, "rationale": raw}

    def judge(
        self,
        case: GoldenCase,
        answer: str,
        evidence_texts: Sequence[str],
    ) -> JudgeResult:
        # 1. Check cache first
        cached = self.cache_manager.get(
            case_id=case.id,
            answer=answer,
            evidence_texts=evidence_texts,
            model=self.model_name,
            prompt_version=self.prompt_version,
        )
        if cached is not None:
            return cached

        # 2. Invoke LLM judge
        llm = self._get_llm()
        evidence_block = "\n---\n".join(evidence_texts) if evidence_texts else "No retrieved evidence."
        user_prompt = (
            f"Question: {case.question}\n\n"
            f"Reference Answer: {case.reference_answer}\n\n"
            f"Grounding Evidence:\n{evidence_block}\n\n"
            f"Candidate Answer:\n{answer}"
        )

        messages = [
            ("system", JUDGE_SYSTEM_PROMPT),
            ("human", user_prompt),
        ]

        try:
            resp = llm.invoke(messages)
            raw_text = str(resp.content)
            parsed = self._parse_judge_response(raw_text)

            result = JudgeResult(
                correctness=parsed.get("correctness"),
                faithfulness=parsed.get("faithfulness"),
                citation_support=parsed.get("citation_support"),
                rationale=parsed.get("rationale", ""),
                model=self.model_name,
                prompt_version=self.prompt_version,
                provider_revision="",
                timestamp=datetime.now(timezone.utc).isoformat(),
                raw_response=raw_text,
                error=parsed.get("error"),
            )

            # Only cache successful parsed results
            if not result.error:
                self.cache_manager.put(
                    case_id=case.id,
                    answer=answer,
                    evidence_texts=evidence_texts,
                    model=self.model_name,
                    prompt_version=self.prompt_version,
                    result=result,
                )
            return result

        except Exception as exc:
            LOGGER.error("NvidiaJudge request failed: %s", exc)
            return JudgeResult(
                correctness=None,
                faithfulness=None,
                citation_support=None,
                rationale="",
                model=self.model_name,
                prompt_version=self.prompt_version,
                timestamp=datetime.now(timezone.utc).isoformat(),
                error=f"{type(exc).__name__}: {exc}",
            )


__all__ = [
    "JUDGE_PROMPT_VERSION",
    "NvidiaJudge",
]
