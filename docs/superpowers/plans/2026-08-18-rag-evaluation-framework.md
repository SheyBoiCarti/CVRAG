# RAG Evaluation Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repository-native, zero-external-platform evaluation framework that measures retrieval, answer, citation, abstention, latency, and cost signals with deterministic metrics, offline fast testing, versioned golden datasets, optional cached LLM judges, and comparison reporting.

**Architecture:** A decoupled `evaluation/` package containing schema definitions, curated golden JSONL dataset, deterministic metric calculations (`retrieval.py`, `answers.py`, `citations.py`), artifact serialization separating stable content hashes from volatile execution metadata, stratified cross-validation and bootstrap calibration, and CLI runner/comparison tools.

**Tech Stack:** Python 3.10+, standard library `dataclasses`, `pydantic`, `numpy`, `pytest`, `langchain-core` / `langchain-nvidia-ai-endpoints` (optional judge only).

## Global Constraints

- Normal test suite MUST run 100% offline with zero network calls and sub-second unit tests.
- Evidence ground truth MUST use `(source, page, anchor)` and never depend on volatile chunk IDs.
- Artifact serialization MUST separate `content` (reproducible JSON hash) from `run_metadata` (timestamps, run IDs, durations).
- Macro averages in metrics MUST only include eligible cases for that metric; failures must never be silently dropped.
- All pytest commands on Windows MUST use `--basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run"`.

---

### Task 1: Schema Definitions & Serialization Models

**Files:**
- Create: `evaluation/__init__.py`
- Create: `evaluation/schema.py`
- Modify: `rag/retrieval/base.py:40-54`
- Test: `tests/evaluation/test_schema.py`

**Interfaces:**
- Produces:
  - `FactMatcher`: Literal["exact_phrase", "token_phrase", "alias_set"]
  - `FactSpec(matcher: FactMatcher, value: str, aliases: tuple[str, ...] = ())`
  - `EvidenceLocator(source: str, page: int, anchor: str)`
  - `GoldenCase(id: str, category: CaseCategory, question: str, history: tuple[tuple[str, str], ...], answerable: bool, reference_answer: str, required_facts: tuple[FactSpec, ...], forbidden_facts: tuple[FactSpec, ...], evidence_groups: tuple[tuple[EvidenceLocator, ...], ...], ambiguity_note: str | None, tags: tuple[str, ...])`
  - `EvaluationTrace(...)` dataclass and `.to_dict()`
  - `RunArtifact(...)` dataclass with `compute_content_fingerprint()`
  - `RankedChunk.to_dict()`

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_schema.py
import json
import pytest
from evaluation.schema import (
    GoldenCase,
    FactSpec,
    EvidenceLocator,
    EvaluationTrace,
    RunArtifact,
    canonical_json_dumps,
    compute_content_hash,
)
from rag.models import AnswerResult, Citation
from rag.retrieval.base import ChunkRecord, RankedChunk

def test_ranked_chunk_to_dict():
    chunk = ChunkRecord(chunk_id="c1", content="Text", source="Custom_CV.md", page=1, chunk_index=0)
    ranked = RankedChunk(chunk=chunk, rank=1, score=0.85, methods=("dense",), stage_scores={"dense": 0.85})
    data = ranked.to_dict()
    assert data["chunk"]["chunk_id"] == "c1"
    assert data["rank"] == 1
    assert data["methods"] == ["dense"]

def test_golden_case_validation():
    case = GoldenCase.from_dict({
        "id": "work-001",
        "category": "straightforward",
        "question": "What role at Wirk?",
        "history": [],
        "answerable": True,
        "reference_answer": "Software Engineer & Game Developer",
        "required_facts": [{"matcher": "token_phrase", "value": "Software Engineer", "aliases": []}],
        "forbidden_facts": [],
        "evidence_groups": [[{"source": "Custom_CV.md", "page": 1, "anchor": "Wirk"}]],
        "ambiguity_note": None,
        "tags": ["employment"],
    })
    assert case.id == "work-001"
    assert case.category == "straightforward"
    assert len(case.evidence_groups) == 1

def test_unanswerable_case_rejects_required_evidence():
    with pytest.raises(ValueError, match="Unanswerable case cannot have required evidence"):
        GoldenCase.from_dict({
            "id": "bad-001",
            "category": "unanswerable",
            "question": "Invalid",
            "history": [],
            "answerable": False,
            "reference_answer": "I do not have information.",
            "required_facts": [],
            "forbidden_facts": [],
            "evidence_groups": [[{"source": "Custom_CV.md", "page": 1, "anchor": "Wirk"}]],
        })

def test_content_fingerprint_deterministic_and_excludes_metadata():
    content1 = {"dataset_hash": "abc", "config": {"k": 5}}
    content2 = {"config": {"k": 5}, "dataset_hash": "abc"}
    assert compute_content_hash(content1) == compute_content_hash(content2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" tests/evaluation/test_schema.py -v`  
Expected: FAIL with "ModuleNotFoundError: No module named 'evaluation'"

- [ ] **Step 3: Implement minimal schema and RankedChunk serialization**

Update `rag/retrieval/base.py` to add `to_dict()` on `RankedChunk`.
Create `evaluation/__init__.py` and `evaluation/schema.py` implementing `FactSpec`, `EvidenceLocator`, `GoldenCase`, `EvaluationTrace`, `RunArtifact`, `compute_content_hash`, and `canonical_json_dumps`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" tests/evaluation/test_schema.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add rag/retrieval/base.py evaluation/__init__.py evaluation/schema.py tests/evaluation/test_schema.py
git commit -m "feat(evaluation): define schema contracts, trace models, and content fingerprinting"
```

---

### Task 2: Curated Golden Dataset Authoring & Validation

**Files:**
- Create: `evaluation/datasets/cv_golden.jsonl`
- Create: `tests/evaluation/test_dataset.py`

**Interfaces:**
- Consumes: `GoldenCase`, `evaluation/schema.py`
- Produces: Valid JSONL dataset with at least 8 reviewed records per category across all 6 categories:
  1. `straightforward` (at least 8)
  2. `exact_term` (at least 8)
  3. `multi_hop` (at least 8)
  4. `conversational_follow_up` (at least 8)
  5. `ambiguous` (at least 8)
  6. `unanswerable` (at least 8, with total dataset having 30+ answerable and 30+ unanswerable for calibration eligibility)

- [ ] **Step 1: Write dataset validation test**

```python
# tests/evaluation/test_dataset.py
from pathlib import Path
import json
from evaluation.schema import GoldenCase

DATASET_PATH = Path(__file__).parents[2] / "evaluation" / "datasets" / "cv_golden.jsonl"

def test_cv_golden_dataset_parses_and_validates():
    assert DATASET_PATH.exists()
    cases = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            case = GoldenCase.from_dict(data)
            cases.append(case)
    
    assert len(cases) >= 60
    
    # Check category distribution
    categories = {}
    answerable_count = 0
    unanswerable_count = 0
    ids = set()
    for case in cases:
        assert case.id not in ids, f"Duplicate ID {case.id}"
        ids.add(case.id)
        categories[case.category] = categories.get(case.category, 0) + 1
        if case.answerable:
            answerable_count += 1
        else:
            unanswerable_count += 1
            
    for req_cat in ["straightforward", "exact_term", "multi_hop", "conversational_follow_up", "ambiguous", "unanswerable"]:
        assert categories.get(req_cat, 0) >= 8, f"Category {req_cat} has fewer than 8 cases ({categories.get(req_cat, 0)})"
        
    assert answerable_count >= 30, f"Need at least 30 answerable cases, got {answerable_count}"
    assert unanswerable_count >= 30, f"Need at least 30 unanswerable cases, got {unanswerable_count}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" tests/evaluation/test_dataset.py -v`  
Expected: FAIL with FileNotFoundError or missing dataset

- [ ] **Step 3: Author comprehensive `cv_golden.jsonl` matching `documents/Custom_CV.md`**

Author 60+ meticulously reviewed records (30+ answerable across straightforward, exact_term, multi_hop, conversational_follow_up, ambiguous, and 30+ unanswerable with realistic queries out of scope for Sheharyar's CV).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" tests/evaluation/test_dataset.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/datasets/cv_golden.jsonl tests/evaluation/test_dataset.py
git commit -m "feat(evaluation): add curated cv_golden dataset with balanced categories"
```

---

### Task 3: Deterministic Retrieval Metrics

**Files:**
- Create: `evaluation/metrics/__init__.py`
- Create: `evaluation/metrics/retrieval.py`
- Test: `tests/evaluation/test_retrieval_metrics.py`

**Interfaces:**
- Consumes: `GoldenCase`, `RankedChunk`, `EvidenceLocator`
- Produces:
  - `matches_locator(chunk: RankedChunk, locator: EvidenceLocator) -> bool`
  - `evaluate_retrieval(case: GoldenCase, retrieved_chunks: Sequence[RankedChunk], ks: Sequence[int] = (1, 3, 5, 10)) -> RetrievalCaseMetrics`
  - `aggregate_retrieval_metrics(case_metrics: Sequence[RetrievalCaseMetrics]) -> dict[str, float]`

- [ ] **Step 1: Write hand-calculated retrieval metrics tests**

```python
# tests/evaluation/test_retrieval_metrics.py
import pytest
from evaluation.schema import GoldenCase, EvidenceLocator
from evaluation.metrics.retrieval import evaluate_retrieval, matches_locator
from rag.retrieval.base import ChunkRecord, RankedChunk

def test_locator_matching_case_and_whitespace_normalization():
    chunk = RankedChunk(
        chunk=ChunkRecord("c1", "Built high-throughput GraphQL backends at Wirk (Pty) Ltd.", "Custom_CV.md", 1, 0),
        rank=1, score=0.9, methods=("dense",),
    )
    assert matches_locator(chunk, EvidenceLocator("Custom_CV.md", 1, "Wirk (Pty) Ltd"))
    assert matches_locator(chunk, EvidenceLocator("custom_cv.md", 1, "wirk (pty) ltd"))
    assert not matches_locator(chunk, EvidenceLocator("Other.md", 1, "Wirk"))
    assert not matches_locator(chunk, EvidenceLocator("Custom_CV.md", 2, "Wirk"))
    assert not matches_locator(chunk, EvidenceLocator("Custom_CV.md", 1, "NonExistent"))

def test_multihop_group_matching_hand_worked():
    case = GoldenCase.from_dict({
        "id": "hop-001", "category": "multi_hop", "question": "q", "history": [], "answerable": True,
        "reference_answer": "ans", "required_facts": [], "forbidden_facts": [],
        "evidence_groups": [
            [{"source": "Custom_CV.md", "page": 1, "anchor": "Wirk"}],
            [{"source": "Custom_CV.md", "page": 1, "anchor": "PRIMED"}],
        ],
    })
    c_wirk = RankedChunk(ChunkRecord("c1", "Worked at Wirk", "Custom_CV.md", 1, 0), 1, 0.9, ("dense",))
    c_irrelevant = RankedChunk(ChunkRecord("c2", "Random info", "Custom_CV.md", 1, 1), 2, 0.8, ("dense",))
    c_primed = RankedChunk(ChunkRecord("c3", "Created PRIMED platform", "Custom_CV.md", 1, 2), 3, 0.7, ("dense",))

    # At K=1: matches group 1 only -> Hit@1=1.0, Recall@1=0.5, AllEvidence@1=0.0, MRR=1.0
    res_k1 = evaluate_retrieval(case, [c_wirk], ks=[1, 3])
    assert res_k1.hit_at_k[1] == 1.0
    assert res_k1.recall_at_k[1] == 0.5
    assert res_k1.all_evidence_at_k[1] == 0.0
    assert res_k1.mrr == 1.0

    # At K=3: matches both groups -> Hit@3=1.0, Recall@3=1.0, AllEvidence@3=1.0
    res_k3 = evaluate_retrieval(case, [c_wirk, c_irrelevant, c_primed], ks=[1, 3])
    assert res_k3.hit_at_k[3] == 1.0
    assert res_k3.recall_at_k[3] == 1.0
    assert res_k3.all_evidence_at_k[3] == 1.0
    assert res_k3.mrr == 1.0

def test_unanswerable_excluded_from_retrieval_recall():
    case = GoldenCase.from_dict({
        "id": "unans-001", "category": "unanswerable", "question": "q", "history": [], "answerable": False,
        "reference_answer": "ans", "required_facts": [], "forbidden_facts": [], "evidence_groups": [],
    })
    res = evaluate_retrieval(case, [])
    assert res.is_applicable is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" tests/evaluation/test_retrieval_metrics.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement retrieval metrics module**

Implement `evaluation/metrics/__init__.py` and `evaluation/metrics/retrieval.py` with normalization, locator checks, group coverage, Hit@K, Recall@K, AllEvidence@K, MRR, nDCG@K, and macro aggregations.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" tests/evaluation/test_retrieval_metrics.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics/__init__.py evaluation/metrics/retrieval.py tests/evaluation/test_retrieval_metrics.py
git commit -m "feat(evaluation): implement deterministic retrieval metrics and evidence locator matching"
```

---

### Task 4: Deterministic Answer & Citation Metrics

**Files:**
- Create: `evaluation/metrics/answers.py`
- Create: `evaluation/metrics/citations.py`
- Test: `tests/evaluation/test_answer_metrics.py`
- Test: `tests/evaluation/test_citation_metrics.py`

**Interfaces:**
- Consumes: `GoldenCase`, `AnswerResult`, `EvaluationTrace`, `FactSpec`
- Produces:
  - `evaluate_answer(case: GoldenCase, answer_result: AnswerResult) -> AnswerCaseMetrics`
  - `evaluate_citations(case: GoldenCase, answer_result: AnswerResult, retrieved_evidence: Sequence[RankedChunk]) -> CitationCaseMetrics`

- [ ] **Step 1: Write hand-calculated answer and citation metrics tests**

```python
# tests/evaluation/test_answer_metrics.py
import pytest
from evaluation.schema import GoldenCase, FactSpec
from evaluation.metrics.answers import evaluate_answer, match_fact
from rag.models import AnswerResult

def test_fact_matching_token_phrase_and_aliases():
    fact_token = FactSpec("token_phrase", "Software Engineer")
    assert match_fact("He was a software engineer at Wirk.", fact_token)
    assert not match_fact("He engineered software.", fact_token)

    fact_alias = FactSpec("alias_set", "Unreal Engine 5", aliases=("UE5", "Unreal 5"))
    assert match_fact("Built games with UE5.", fact_alias)
    assert match_fact("Used Unreal Engine 5 extensively.", fact_alias)
    assert not match_fact("Used Unity.", fact_alias)

def test_answer_metrics_abstentions_and_coverage():
    case_ans = GoldenCase.from_dict({
        "id": "c1", "category": "straightforward", "question": "q", "history": [], "answerable": True,
        "reference_answer": "Software Engineer",
        "required_facts": [{"matcher": "token_phrase", "value": "Software Engineer", "aliases": []}],
        "forbidden_facts": [{"matcher": "token_phrase", "value": "Doctor", "aliases": []}],
        "evidence_groups": [],
    })
    
    # Correct answer
    res1 = evaluate_answer(case_ans, AnswerResult("He is a Software Engineer.", decision="answered"))
    assert res1.required_fact_coverage == 1.0
    assert res1.forbidden_fact_violation == 0.0
    assert res1.incorrect_abstention == 0.0

    # Incorrect abstention
    res2 = evaluate_answer(case_ans, AnswerResult("I cannot find information.", decision="abstained"))
    assert res2.incorrect_abstention == 1.0
    assert res2.required_fact_coverage == 0.0

    # Unanswerable case
    case_unans = GoldenCase.from_dict({
        "id": "u1", "category": "unanswerable", "question": "q", "history": [], "answerable": False,
        "reference_answer": "I do not know", "required_facts": [], "forbidden_facts": [], "evidence_groups": [],
    })
    res3 = evaluate_answer(case_unans, AnswerResult("I do not have enough info.", decision="abstained"))
    assert res3.correct_abstention == 1.0
```

```python
# tests/evaluation/test_citation_metrics.py
import pytest
from evaluation.schema import GoldenCase
from evaluation.metrics.citations import evaluate_citations
from rag.models import AnswerResult, Citation, ClaimVerification
from rag.retrieval.base import ChunkRecord, RankedChunk

def test_citation_metrics_validity_and_coverage():
    case = GoldenCase.from_dict({
        "id": "c1", "category": "straightforward", "question": "q", "history": [], "answerable": True,
        "reference_answer": "ans", "required_facts": [], "forbidden_facts": [], "evidence_groups": [],
    })
    retrieved = [RankedChunk(ChunkRecord("c1", "Content", "Custom_CV.md", 1, 0), 1, 0.9, ("dense",))]
    
    citations = (Citation(evidence_id="c1", chunk_id="c1", source="Custom_CV.md", page=1, excerpt="Content"),)
    claims = (ClaimVerification(claim_id="cl1", text="Worked at Wirk", citation_ids=("c1",)),)
    ans = AnswerResult("Answer [S1]", citations=citations, claim_verifications=claims)

    res = evaluate_citations(case, ans, retrieved)
    assert res.citation_id_validity == 1.0
    assert res.claim_citation_coverage == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" tests/evaluation/test_answer_metrics.py tests/evaluation/test_citation_metrics.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement answer and citation metrics**

Implement `evaluation/metrics/answers.py` and `evaluation/metrics/citations.py` with Unicode normalization, fact matchers, abstention tracking, and citation validity.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" tests/evaluation/test_answer_metrics.py tests/evaluation/test_citation_metrics.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics/answers.py evaluation/metrics/citations.py tests/evaluation/test_answer_metrics.py tests/evaluation/test_citation_metrics.py
git commit -m "feat(evaluation): implement deterministic answer and citation metrics"
```

---

### Task 5: Pipeline Evaluation Trace Capture

**Files:**
- Modify: `rag/pipeline.py:100-300`
- Test: `tests/evaluation/test_pipeline_trace.py`

**Interfaces:**
- Modifies: `RAGPipeline`
- Adds: `answer_with_trace(self, question: str, history: Sequence[ChatTurn] | None = None, case_id: str = "") -> tuple[AnswerResult, EvaluationTrace]`

- [ ] **Step 1: Write failing test for pipeline trace capture**

```python
# tests/evaluation/test_pipeline_trace.py
from rag.models import ChatTurn
from rag.pipeline import RAGPipeline
from rag.settings import Settings
from tests.fakes import FakeChatModel, FakeEmbeddings, FakeProviderFactories

def test_pipeline_answer_with_trace_captures_all_stages():
    factories = FakeProviderFactories(
        embeddings=FakeEmbeddings(),
        chat_model=FakeChatModel(default_response="Sheharyar worked at Wirk [S1]."),
    )
    settings = Settings(
        HYBRID_RETRIEVAL_ENABLED=True,
        RETRIEVAL_MODE="hybrid",
        RERANK_ENABLED=False,
    )
    pipeline = RAGPipeline(settings=settings, provider_factories=factories)
    
    ans, trace = pipeline.answer_with_trace("What role at Wirk?", case_id="test-001")
    assert ans.answer == "Sheharyar worked at Wirk [S1]."
    assert trace.case_id == "test-001"
    assert trace.retrieval_query == "What role at Wirk?"
    assert isinstance(trace.dense_results, tuple)
    assert isinstance(trace.fused_results, tuple)
    assert trace.corpus_fingerprint != ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" tests/evaluation/test_pipeline_trace.py -v`  
Expected: FAIL with AttributeError: 'RAGPipeline' object has no attribute 'answer_with_trace'

- [ ] **Step 3: Implement `answer_with_trace` in `rag/pipeline.py`**

Instrument internal query reformulation, dense, sparse, fusion, and reranking timings and candidate outputs, returning `(AnswerResult, EvaluationTrace)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" tests/evaluation/test_pipeline_trace.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add rag/pipeline.py tests/evaluation/test_pipeline_trace.py
git commit -m "feat(pipeline): expose evaluation trace diagnostics on RAGPipeline"
```

---

### Task 6: Evaluation Runner & Artifact Serialization

**Files:**
- Create: `evaluation/runner.py`
- Test: `tests/evaluation/test_runner.py`

**Interfaces:**
- Consumes: `GoldenCase`, `RAGPipeline`, `EvaluationTrace`, `evaluate_retrieval`, `evaluate_answer`, `evaluate_citations`
- Produces: CLI `python -m evaluation.runner` generating canonical JSON artifacts with deterministic content hash.

- [ ] **Step 1: Write runner unit and CLI test**

```python
# tests/evaluation/test_runner.py
import json
from pathlib import Path
from evaluation.runner import EvaluationRunner, run_evaluation
from evaluation.schema import GoldenCase
from tests.fakes import FakeChatModel, FakeEmbeddings, FakeProviderFactories
from rag.pipeline import RAGPipeline
from rag.settings import Settings

def test_runner_executes_cases_and_builds_artifact(tmp_path):
    cases = [
        GoldenCase.from_dict({
            "id": "c1", "category": "straightforward", "question": "Role at Wirk?", "history": [],
            "answerable": True, "reference_answer": "Software Engineer",
            "required_facts": [{"matcher": "token_phrase", "value": "Software Engineer", "aliases": []}],
            "forbidden_facts": [], "evidence_groups": [[{"source": "Custom_CV.md", "page": 1, "anchor": "Wirk"}]],
        })
    ]
    factories = FakeProviderFactories(
        embeddings=FakeEmbeddings(),
        chat_model=FakeChatModel(default_response="He is a Software Engineer at Wirk [S1]."),
    )
    pipeline = RAGPipeline(settings=Settings(), provider_factories=factories)
    runner = EvaluationRunner(pipeline=pipeline)
    
    artifact = runner.run(cases, run_id="fixed-run-id", timestamp="2026-08-18T12:00:00Z")
    out_file = tmp_path / "run.json"
    artifact.save(out_file)
    
    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert "content" in data
    assert "content_fingerprint" in data
    assert "run_metadata" in data
    assert data["content_fingerprint"] == artifact.content_fingerprint
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" tests/evaluation/test_runner.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement `evaluation/runner.py`**

Implement `EvaluationRunner`, argparsing CLI for `--dataset`, `--config` (`dense-baseline`, `hybrid-no-rerank`, `hybrid-rerank`), `--output`, and optional judge flags.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" tests/evaluation/test_runner.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/runner.py tests/evaluation/test_runner.py
git commit -m "feat(evaluation): implement runner CLI and deterministic artifact export"
```

---

### Task 7: Comparison Tool & Category Deltas

**Files:**
- Create: `evaluation/compare.py`
- Test: `tests/evaluation/test_compare.py`

**Interfaces:**
- Consumes: Run artifact JSON files
- Produces: CLI `python -m evaluation.compare base.json candidate.json` printing category deltas, macro changes, and regression alerts.

- [ ] **Step 1: Write comparison test**

```python
# tests/evaluation/test_compare.py
from evaluation.compare import compare_runs, ComparisonReport
from evaluation.schema import RunArtifact

def test_compare_detects_improvements_and_regressions():
    base_dict = {
        "content_fingerprint": "hash1",
        "content": {
            "category_aggregates": {
                "straightforward": {"hit_at_5": 0.8, "required_fact_coverage": 0.8},
                "unanswerable": {"correct_abstention": 1.0},
            },
            "macro_aggregates": {"hit_at_5": 0.8, "required_fact_coverage": 0.8},
            "cases": [{"case_id": "c1", "metrics": {"hit_at_5": 0.0}}],
        },
        "run_metadata": {"run_id": "r1"},
    }
    cand_dict = {
        "content_fingerprint": "hash2",
        "content": {
            "category_aggregates": {
                "straightforward": {"hit_at_5": 1.0, "required_fact_coverage": 1.0},
                "unanswerable": {"correct_abstention": 0.9},
            },
            "macro_aggregates": {"hit_at_5": 1.0, "required_fact_coverage": 1.0},
            "cases": [{"case_id": "c1", "metrics": {"hit_at_5": 1.0}}],
        },
        "run_metadata": {"run_id": "r2"},
    }
    report = compare_runs(base_dict, cand_dict)
    assert report.macro_deltas["hit_at_5"] == pytest.approx(0.2)
    assert report.category_deltas["unanswerable"]["correct_abstention"] == pytest.approx(-0.1)
    assert "c1" in report.improved_case_ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" tests/evaluation/test_compare.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement `evaluation/compare.py`**

Implement comparison logic, category delta tables, formatting, regression/improvement breakdowns, and CLI entry point.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" tests/evaluation/test_compare.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/compare.py tests/evaluation/test_compare.py
git commit -m "feat(evaluation): implement baseline comparison tool with category deltas"
```

---

### Task 8: Optional Judge Protocol, Cache & Adapters

**Files:**
- Create: `evaluation/judges/__init__.py`
- Create: `evaluation/judges/base.py`
- Create: `evaluation/judges/fake.py`
- Create: `evaluation/judges/nvidia.py`
- Test: `tests/evaluation/test_judges.py`

**Interfaces:**
- Produces:
  - `JudgeResult(correctness: float, faithfulness: float, citation_support: float, rationale: str, model: str, prompt_version: str, timestamp: str)`
  - `AnswerJudge(Protocol)`
  - `JudgeCacheManager`
  - `FakeJudge(AnswerJudge)`
  - `NvidiaJudge(AnswerJudge)`

- [ ] **Step 1: Write judge protocol, cache, and fake tests**

```python
# tests/evaluation/test_judges.py
import pytest
from evaluation.schema import GoldenCase
from evaluation.judges.base import JudgeResult, JudgeCacheManager
from evaluation.judges.fake import FakeJudge
from rag.models import AnswerResult

def test_fake_judge_returns_deterministic_scores_offline():
    judge = FakeJudge(correctness=1.0, faithfulness=0.9, citation_support=1.0)
    case = GoldenCase.from_dict({
        "id": "c1", "category": "straightforward", "question": "q", "history": [],
        "answerable": True, "reference_answer": "ans", "required_facts": [], "forbidden_facts": [], "evidence_groups": [],
    })
    res = judge.judge(case, answer="ans", evidence_texts=["chunk text"])
    assert res.correctness == 1.0
    assert res.faithfulness == 0.9

def test_judge_cache_avoids_recomputation(tmp_path):
    cache = JudgeCacheManager(cache_dir=tmp_path)
    result = JudgeResult(1.0, 1.0, 1.0, "Good", "fake-model", "v1", "2026-08-18")
    cache.put("c1", "ans", ["evidence"], "fake-model", "v1", result)
    
    cached = cache.get("c1", "ans", ["evidence"], "fake-model", "v1")
    assert cached is not None
    assert cached.rationale == "Good"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" tests/evaluation/test_judges.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement judge modules and cache manager**

Implement `evaluation/judges/base.py`, `fake.py`, `nvidia.py`, and `__init__.py`. Ensure `nvidia.py` only initializes network clients when explicitly invoked, temperature is 0.0, and failures are parsed into structured failure records without crashing the runner.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" tests/evaluation/test_judges.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/judges/ tests/evaluation/test_judges.py
git commit -m "feat(evaluation): implement optional judge protocol, disk cache, and adapters"
```

---

### Task 9: Stratified Calibration & Bootstrap Confidence Intervals

**Files:**
- Create: `evaluation/calibration.py`
- Test: `tests/evaluation/test_calibration.py`

**Interfaces:**
- Consumes: Golden dataset and evaluation outputs
- Produces:
  - `stratified_k_fold_split(cases: Sequence[GoldenCase], n_splits: int = 5, seed: int = 42) -> list[tuple[list[GoldenCase], list[GoldenCase]]]`
  - `calibrate_answerability_threshold(cases: Sequence[GoldenCase], predictions: Sequence[dict], cost_weights: tuple[float, float] = (5.0, 1.0)) -> CalibrationOutcome`
  - `compute_bootstrap_ci(scores: Sequence[float], n_resamples: int = 1000, confidence: float = 0.95, seed: int = 42) -> tuple[float, float]`

- [ ] **Step 1: Write calibration and bootstrap tests**

```python
# tests/evaluation/test_calibration.py
import pytest
from evaluation.calibration import stratified_k_fold_split, compute_bootstrap_ci, calibrate_threshold
from evaluation.schema import GoldenCase

def test_stratified_split_preserves_category_distribution():
    cases = []
    for i in range(25):
        cases.append(GoldenCase.from_dict({
            "id": f"ans-{i}", "category": "straightforward", "question": "q", "history": [], "answerable": True,
            "reference_answer": "a", "required_facts": [], "forbidden_facts": [], "evidence_groups": [],
        }))
        cases.append(GoldenCase.from_dict({
            "id": f"unans-{i}", "category": "unanswerable", "question": "q", "history": [], "answerable": False,
            "reference_answer": "a", "required_facts": [], "forbidden_facts": [], "evidence_groups": [],
        }))
    folds = stratified_k_fold_split(cases, n_splits=5, seed=42)
    assert len(folds) == 5
    for train, val in folds:
        assert len(train) == 40
        assert len(val) == 10
        val_unans = sum(1 for c in val if not c.answerable)
        assert val_unans == 5

def test_bootstrap_confidence_interval_deterministic():
    scores = [1.0] * 80 + [0.0] * 20
    low, high = compute_bootstrap_ci(scores, n_resamples=500, confidence=0.95, seed=42)
    assert 0.70 <= low <= 0.80
    assert 0.80 <= high <= 0.90
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" tests/evaluation/test_calibration.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement `evaluation/calibration.py`**

Implement stratified cross-validation, cost-optimal threshold selection ($5 \times \text{false\_answer} + 1 \times \text{false\_abstention}$), locked test set evaluation, and bootstrap confidence intervals.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" tests/evaluation/test_calibration.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/calibration.py tests/evaluation/test_calibration.py
git commit -m "feat(evaluation): add stratified calibration and bootstrap confidence intervals"
```

---

### Task 10: Baseline Generation, Documentation & Integration Tests

**Files:**
- Create: `evaluation/baselines/.gitkeep`
- Create: `evaluation/baselines/deterministic_fixture_baseline.json`
- Create: `tests/evaluation/test_evaluation_integration.py`
- Modify: `requirements-dev.txt`
- Modify: `README.md`

**Interfaces:**
- Produces:
  - Checked-in deterministic fixture baseline.
  - End-to-end integration test running the entire evaluation pipeline on the full dataset offline.
  - Clear user documentation for running evaluation, comparing baselines, and authoring new cases.

- [ ] **Step 1: Write full offline integration test**

```python
# tests/evaluation/test_evaluation_integration.py
from pathlib import Path
from evaluation.runner import EvaluationRunner
from evaluation.schema import GoldenCase
from tests.fakes import FakeChatModel, FakeEmbeddings, FakeProviderFactories
from rag.pipeline import RAGPipeline
from rag.settings import Settings

DATASET_PATH = Path(__file__).parents[2] / "evaluation" / "datasets" / "cv_golden.jsonl"

def test_full_offline_evaluation_run_on_golden_dataset():
    cases = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(GoldenCase.from_dict(json.loads(line)))
                
    factories = FakeProviderFactories(
        embeddings=FakeEmbeddings(),
        chat_model=FakeChatModel(default_response="He was a Software Engineer at Wirk [S1]."),
    )
    pipeline = RAGPipeline(settings=Settings(HYBRID_RETRIEVAL_ENABLED=True), provider_factories=factories)
    runner = EvaluationRunner(pipeline=pipeline)
    
    artifact = runner.run(cases, run_id="integration-test-run")
    assert len(artifact.content["cases"]) == len(cases)
    assert "macro_aggregates" in artifact.content
    assert artifact.content_fingerprint != ""
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" tests/evaluation/test_evaluation_integration.py -v`  
Expected: PASS

- [ ] **Step 3: Update documentation and checked-in fixture baseline**

Update `README.md` with usage instructions (`python -m evaluation.runner`, `python -m evaluation.compare`), add `evaluation/baselines/deterministic_fixture_baseline.json`, and verify `requirements-dev.txt`.

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest --basetemp="C:/Users/sheha/OneDrive/Desktop/CVRAG/.pytest_tmp_run" -v`  
Expected: 100% tests PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/baselines/ README.md requirements-dev.txt tests/evaluation/
git commit -m "feat(evaluation): add baseline fixtures, integration tests, and documentation"
```
