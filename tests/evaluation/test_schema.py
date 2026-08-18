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
    assert data["score"] == 0.85
    assert data["stage_scores"] == {"dense": 0.85}


def test_golden_case_validation_valid():
    case = GoldenCase.from_dict({
        "id": "work-001",
        "category": "straightforward",
        "question": "What role did Sheharyar hold at Wirk?",
        "history": [],
        "answerable": True,
        "reference_answer": "Software Engineer & Game Developer.",
        "required_facts": [{"matcher": "token_phrase", "value": "Software Engineer", "aliases": []}],
        "forbidden_facts": [],
        "evidence_groups": [[{"source": "Custom_CV.md", "page": 1, "anchor": "Wirk (Pty) Ltd"}]],
        "ambiguity_note": None,
        "tags": ["employment", "exact-title"],
    })
    assert case.id == "work-001"
    assert case.category == "straightforward"
    assert case.answerable is True
    assert len(case.evidence_groups) == 1
    assert case.evidence_groups[0][0].anchor == "Wirk (Pty) Ltd"
    assert case.to_dict()["id"] == "work-001"


def test_golden_case_invalid_category_raises():
    with pytest.raises(ValueError, match="Invalid category"):
        GoldenCase.from_dict({
            "id": "bad-cat",
            "category": "invalid_category",
            "question": "Question?",
            "history": [],
            "answerable": True,
            "reference_answer": "Ans",
            "required_facts": [],
            "forbidden_facts": [],
            "evidence_groups": [],
        })


def test_unanswerable_case_rejects_required_evidence():
    with pytest.raises(ValueError, match="Unanswerable case .* cannot have required evidence"):
        GoldenCase.from_dict({
            "id": "bad-unans",
            "category": "unanswerable",
            "question": "Invalid",
            "history": [],
            "answerable": False,
            "reference_answer": "I do not have enough information.",
            "required_facts": [],
            "forbidden_facts": [],
            "evidence_groups": [[{"source": "Custom_CV.md", "page": 1, "anchor": "Wirk"}]],
        })


def test_answerable_case_requires_non_empty_anchor():
    with pytest.raises(ValueError, match="Anchor cannot be empty"):
        GoldenCase.from_dict({
            "id": "bad-anchor",
            "category": "straightforward",
            "question": "Question?",
            "history": [],
            "answerable": True,
            "reference_answer": "Ans",
            "required_facts": [],
            "forbidden_facts": [],
            "evidence_groups": [[{"source": "Custom_CV.md", "page": 1, "anchor": "   "}]],
        })


def test_content_fingerprint_deterministic_and_excludes_metadata():
    content1 = {"dataset_hash": "abc", "config": {"k": 5}}
    content2 = {"config": {"k": 5}, "dataset_hash": "abc"}
    assert compute_content_hash(content1) == compute_content_hash(content2)


def test_run_artifact_serialization():
    content = {
        "dataset_hash": "hash-123",
        "corpus_fingerprint": "corp-1",
        "index_fingerprint": "idx-1",
        "configuration": {"RETRIEVAL_MODE": "hybrid"},
        "cases": [],
        "category_aggregates": {},
        "macro_aggregates": {},
    }
    artifact = RunArtifact.create(
        content=content,
        run_id="run-001",
        timestamp="2026-08-18T12:00:00Z",
        git_revision="abcdef",
        git_dirty=False,
    )
    assert artifact.content_fingerprint == compute_content_hash(content)
    d = artifact.to_dict()
    assert d["run_metadata"]["run_id"] == "run-001"
    assert d["content_fingerprint"] == artifact.content_fingerprint
    assert d["content"]["dataset_hash"] == "hash-123"
