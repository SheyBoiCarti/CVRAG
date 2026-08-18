# RAG Evaluation Framework — System Design Document

**Date:** 2026-08-18  
**Topic:** RAG Evaluation Framework (Specification 03)  
**Status:** Approved  

---

## 1. Overview & Scope

The RAG Evaluation Framework provides repository-native evaluation for the CVRAG system. It measures retrieval performance, answer correctness, fact coverage, citation fidelity, abstention behavior, latency, and cost signals without relying on heavy external evaluation platforms.

The design strictly maintains test isolation: all unit tests run offline in sub-second execution using deterministic hand-worked fixtures and fakes. Live provider execution (including optional LLM-as-a-judge) is supported via explicit CLI commands and cached replayable artifacts.

---

## 2. Architecture & File Structure

```
evaluation/
├── __init__.py
├── schema.py              # GoldenCase, EvaluationTrace, RunArtifact, MetricResults
├── calibration.py         # Stratified 5-fold CV, bootstrap confidence intervals, cost optimization
├── runner.py              # CLI evaluation runner with pluggable pipeline and offline mode
├── compare.py             # CLI comparison tool calculating category & metric deltas
├── datasets/
│   └── cv_golden.jsonl    # Curated golden evaluation dataset over Custom_CV.md
├── metrics/
│   ├── __init__.py
│   ├── retrieval.py       # Hit@K, Recall@K, AllEvidence@K, MRR, nDCG@K, evidence matching
│   ├── answers.py         # Required fact coverage, forbidden facts, exact match, abstentions
│   └── citations.py       # Citation validity, coverage, locator accuracy, claim support
├── judges/
│   ├── __init__.py
│   ├── base.py            # AnswerJudge Protocol, JudgeResult dataclass, CacheManager
│   ├── fake.py            # Deterministic offline FakeJudge
│   └── nvidia.py          # Optional NVIDIA LLM judge adapter (disabled by default)
└── baselines/
    └── .gitkeep           # Check-in directory for baseline JSON artifacts
```

Supporting modifications:
- `rag/retrieval/base.py`: Add canonical `.to_dict()` serialization to `RankedChunk`.
- `rag/pipeline.py`: Expose `answer_with_trace(...)` returning `(AnswerResult, EvaluationTrace)`.
- `requirements-dev.txt`: Include development/testing dependencies if needed.
- `tests/evaluation/`: Comprehensive test suite verifying all metrics, schemas, calibration, runner, and comparison.

---

## 3. Data Contracts & Schemas (`evaluation/schema.py`)

### 3.1 Golden Case Schema
```python
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

CaseCategory = Literal[
    "straightforward",
    "exact_term",
    "multi_hop",
    "conversational_follow_up",
    "ambiguous",
    "unanswerable",
]

FactMatcher = Literal["exact_phrase", "token_phrase", "alias_set"]

@dataclass(frozen=True)
class FactSpec:
    matcher: FactMatcher
    value: str
    aliases: tuple[str, ...] = ()

@dataclass(frozen=True)
class EvidenceLocator:
    source: str
    page: int
    anchor: str

@dataclass(frozen=True)
class GoldenCase:
    id: str
    category: CaseCategory
    question: str
    history: tuple[tuple[str, str], ...] = ()
    answerable: bool = True
    reference_answer: str = ""
    required_facts: tuple[FactSpec, ...] = ()
    forbidden_facts: tuple[FactSpec, ...] = ()
    evidence_groups: tuple[tuple[EvidenceLocator, ...], ...] = ()
    ambiguity_note: str | None = None
    tags: tuple[str, ...] = ()
```

### 3.2 Evaluation Trace
```python
@dataclass(frozen=True)
class EvaluationTrace:
    case_id: str
    retrieval_query: str
    dense_results: tuple[RankedChunk, ...]
    sparse_results: tuple[RankedChunk, ...]
    fused_results: tuple[RankedChunk, ...]
    reranked_results: tuple[RankedChunk, ...]
    answer_result: AnswerResult
    timings_ms: Mapping[str, float]
    token_usage: Mapping[str, int]
    configuration: Mapping[str, Any]
    corpus_fingerprint: str
    index_fingerprint: str
```

### 3.3 Artifact Structure & Content Fingerprinting
The output artifact JSON strictly separates stable deterministic `content` from volatile `run_metadata`:
- **`content`**: `{"dataset_hash": ..., "corpus_fingerprint": ..., "index_fingerprint": ..., "configuration": ..., "cases": [...], "category_aggregates": {...}, "macro_aggregates": {...}}`
- **`content_fingerprint`**: `sha256(canonical_json(content))`
- **`run_metadata`**: `{"run_id": ..., "timestamp": ..., "git_revision": ..., "git_dirty": ..., "duration_ms": ..., "host": ...}`

---

## 4. Evaluation Metrics Specifications

### 4.1 Retrieval Metrics (`evaluation/metrics/retrieval.py`)
- **Evidence Locator Match**: A chunk matches an `EvidenceLocator` iff:
  1. `chunk.source == locator.source`
  2. `chunk.page == locator.page`
  3. `normalize_text(locator.anchor) in normalize_text(chunk.content)`
- **Multi-Hop Group Rule**: A group is satisfied if at least one locator in that group matches a retrieved chunk in top K.
- **Metrics**:
  - `Hit@K`: 1.0 if at least one required evidence group is matched in top K, else 0.0.
  - `Recall@K`: `len(matched_groups) / len(required_groups)` (omitted/excluded for `unanswerable` cases).
  - `AllEvidence@K`: 1.0 if all required evidence groups are matched in top K, else 0.0.
  - `MRR`: `1.0 / first_matched_rank` (or 0.0 if no match).
  - `nDCG@K`: Discounted cumulative gain for cases with graded relevance labels.

### 4.2 Answer & Fact Metrics (`evaluation/metrics/answers.py`)
- **Text Normalization**: Unicode NFKC, `casefold()`, whitespace collapsing, and word token boundary boundaries `\b`.
- **Fact Matchers**:
  - `exact_phrase`: normalized substring match.
  - `token_phrase`: token sequence match respecting token boundaries.
  - `alias_set`: primary value or any reviewed alias matched.
- **Metrics**:
  - `required_fact_coverage`: matched required facts / total required facts.
  - `forbidden_fact_violation`: fraction of forbidden facts present.
  - `exact_match`: full normalized answer string equality.
  - `correct_abstention`: 1.0 if unanswerable and pipeline abstained (`decision == "abstained"` or empty retrieval).
  - `incorrect_abstention`: 1.0 if answerable but pipeline abstained.

### 4.3 Citation Metrics (`evaluation/metrics/citations.py`)
- `citation_id_validity`: Valid chunk/evidence ID reference ratio.
- `claim_citation_coverage`: Ratio of claims having valid citation IDs.
- `locator_accuracy`: Citations whose source and page match the underlying evidence.

---

## 5. Calibration & Bootstrap Protocol (`evaluation/calibration.py`)

1. **Stratified 5-Fold Cross-Validation**: Folds preserve category and answerability balance.
2. **Cost Function**: $\text{Cost} = 5 \times \text{false\_answer} + 1 \times \text{false\_abstention}$.
3. **Locked Test Set**: Held-out partition (e.g. 20 cases with at least 10 answerable and 10 unanswerable) untouched during threshold selection.
4. **Bootstrap 95% Confidence Intervals**: 1,000 resamples using a fixed seed (e.g. `seed=42`) to compute deterministic confidence intervals for false-answer and false-abstention rates.

---

## 6. Judges & Caching (`evaluation/judges/`)

- Protocol `AnswerJudge` defines `judge(case: GoldenCase, trace: EvaluationTrace) -> JudgeResult`.
- `JudgeResult`: `correctness: float`, `faithfulness: float`, `citation_support: float`, `rationale: str`, `model: str`, `prompt_version: str`, `timestamp: str`.
- Caching: SHA-256 key built from `(case.id, trace.answer_result.answer, evidence_texts, model, prompt_version)`.
- `NvidiaJudge`: Configurable endpoint with temperature 0.0, disabled by default.
- `FakeJudge`: Provides deterministic test fixtures with zero network calls.

---

## 7. Testing & Verification Strategy

- Schema validation tests (`tests/evaluation/test_schema.py`).
- Hand-worked retrieval metric calculation tests (`tests/evaluation/test_retrieval_metrics.py`).
- Hand-worked answer/citation metric tests (`tests/evaluation/test_answer_metrics.py`, `tests/evaluation/test_citation_metrics.py`).
- Runner & artifact serialization tests verifying content fingerprint stability (`tests/evaluation/test_runner.py`).
- Comparison tests verifying regression and improvement detection (`tests/evaluation/test_compare.py`).
- Calibration stratification and bootstrap reproducibility tests (`tests/evaluation/test_calibration.py`).
- Judge caching and offline fake tests (`tests/evaluation/test_judges.py`).
- Integration test executing offline evaluation on the full golden dataset (`tests/evaluation/test_evaluation_integration.py`).
