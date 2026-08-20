import pytest

from evaluation.metrics.retrieval import (
    aggregate_retrieval_metrics,
    evaluate_retrieval,
    matches_locator,
    normalize_text,
)
from evaluation.schema import EvidenceLocator, GoldenCase
from rag.retrieval.base import ChunkRecord, RankedChunk


def test_normalize_text_whitespace_and_case():
    assert normalize_text("  Wirk (Pty)  Ltd \n") == "wirk (pty) ltd"
    assert normalize_text("NVIDIA\u00A0AI") == "nvidia ai"


def test_locator_matching_case_and_whitespace_normalization():
    chunk = RankedChunk(
        chunk=ChunkRecord("c1", "Built high-throughput GraphQL backends at Wirk (Pty) Ltd.", "Custom_CV.md", 1, 0),
        rank=1,
        score=0.9,
        methods=("dense",),
    )
    assert matches_locator(chunk, EvidenceLocator("Custom_CV.md", 1, "Wirk (Pty) Ltd"))
    assert matches_locator(chunk, EvidenceLocator("custom_cv.md", 1, "wirk (pty) ltd"))
    assert not matches_locator(chunk, EvidenceLocator("Other.md", 1, "Wirk"))
    assert not matches_locator(chunk, EvidenceLocator("Custom_CV.md", 2, "Wirk"))
    assert not matches_locator(chunk, EvidenceLocator("Custom_CV.md", 1, "NonExistentAnchor"))


def test_multihop_group_matching_hand_worked():
    case = GoldenCase.from_dict({
        "id": "hop-001",
        "category": "multi_hop",
        "question": "Compare Wirk and PRIMED",
        "history": [],
        "answerable": True,
        "reference_answer": "ans",
        "required_facts": [],
        "forbidden_facts": [],
        "evidence_groups": [
            [{"source": "Custom_CV.md", "page": 1, "anchor": "Wirk"}],
            [{"source": "Custom_CV.md", "page": 1, "anchor": "PRIMED"}],
        ],
    })
    c_wirk = RankedChunk(ChunkRecord("c1", "Worked at Wirk (Pty) Ltd", "Custom_CV.md", 1, 0), 1, 0.9, ("dense",))
    c_irrelevant = RankedChunk(ChunkRecord("c2", "Random info about CellZone", "Custom_CV.md", 1, 1), 2, 0.8, ("dense",))
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


def test_no_matches_mrr_and_hit_are_zero():
    case = GoldenCase.from_dict({
        "id": "c1",
        "category": "straightforward",
        "question": "q",
        "history": [],
        "answerable": True,
        "reference_answer": "ans",
        "required_facts": [],
        "forbidden_facts": [],
        "evidence_groups": [[{"source": "Custom_CV.md", "page": 1, "anchor": "Target"}]],
    })
    chunk = RankedChunk(ChunkRecord("c1", "Unrelated text", "Custom_CV.md", 1, 0), 1, 0.5, ("dense",))
    res = evaluate_retrieval(case, [chunk], ks=[1, 5])
    assert res.hit_at_k[1] == 0.0
    assert res.hit_at_k[5] == 0.0
    assert res.recall_at_k[1] == 0.0
    assert res.all_evidence_at_k[1] == 0.0
    assert res.mrr == 0.0


def test_unanswerable_case_excluded_from_retrieval():
    case = GoldenCase.from_dict({
        "id": "unans-001",
        "category": "unanswerable",
        "question": "q",
        "history": [],
        "answerable": False,
        "reference_answer": "ans",
        "required_facts": [],
        "forbidden_facts": [],
        "evidence_groups": [],
    })
    res = evaluate_retrieval(case, [])
    assert res.is_applicable is False


def test_aggregate_retrieval_metrics():
    case1 = GoldenCase.from_dict({
        "id": "c1", "category": "straightforward", "question": "q", "history": [], "answerable": True,
        "reference_answer": "ans", "required_facts": [], "forbidden_facts": [],
        "evidence_groups": [[{"source": "Custom_CV.md", "page": 1, "anchor": "A"}]],
    })
    case2 = GoldenCase.from_dict({
        "id": "c2", "category": "straightforward", "question": "q", "history": [], "answerable": True,
        "reference_answer": "ans", "required_facts": [], "forbidden_facts": [],
        "evidence_groups": [[{"source": "Custom_CV.md", "page": 1, "anchor": "B"}]],
    })
    unans = GoldenCase.from_dict({
        "id": "u1", "category": "unanswerable", "question": "q", "history": [], "answerable": False,
        "reference_answer": "ans", "required_facts": [], "forbidden_facts": [], "evidence_groups": [],
    })
    chunk_a = RankedChunk(ChunkRecord("c1", "Text A", "Custom_CV.md", 1, 0), 1, 0.9, ("dense",))
    chunk_b = RankedChunk(ChunkRecord("c2", "Text B", "Custom_CV.md", 1, 1), 2, 0.8, ("dense",))

    m1 = evaluate_retrieval(case1, [chunk_a], ks=[1, 5])  # Hit@1=1, Recall=1, MRR=1
    m2 = evaluate_retrieval(case2, [chunk_a, chunk_b], ks=[1, 5])  # Hit@1=0, Hit@5=1, Recall@1=0, Recall@5=1, MRR=0.5
    m3 = evaluate_retrieval(unans, [chunk_a], ks=[1, 5])  # is_applicable = False

    agg = aggregate_retrieval_metrics([m1, m2, m3])
    assert agg["hit_at_1"] == pytest.approx(0.5)
    assert agg["hit_at_5"] == pytest.approx(1.0)
    assert agg["recall_at_1"] == pytest.approx(0.5)
    assert agg["recall_at_5"] == pytest.approx(1.0)
    assert agg["mrr"] == pytest.approx(0.75)
    assert agg["applicable_cases"] == 2
