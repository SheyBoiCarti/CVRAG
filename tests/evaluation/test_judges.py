from evaluation.judges.base import JudgeCacheManager, JudgeResult
from evaluation.judges.fake import FakeJudge
from evaluation.judges.nvidia import NvidiaJudge
from evaluation.schema import GoldenCase


def test_fake_judge_returns_deterministic_scores_offline():
    judge = FakeJudge(
        correctness=1.0,
        faithfulness=0.9,
        citation_support=1.0,
        rationale="Clear evidence support.",
    )
    case = GoldenCase.from_dict({
        "id": "sf-001",
        "category": "straightforward",
        "question": "What is Sheharyar's role?",
        "history": [],
        "answerable": True,
        "reference_answer": "Software Engineer & Game Developer",
        "required_facts": [],
        "forbidden_facts": [],
        "evidence_groups": [],
    })

    result = judge.judge(
        case=case,
        answer="He is a Software Engineer & Game Developer.",
        evidence_texts=["Worked as Software Engineer & Game Developer at Wirk."],
    )

    assert isinstance(result, JudgeResult)
    assert result.correctness == 1.0
    assert result.faithfulness == 0.9
    assert result.citation_support == 1.0
    assert result.rationale == "Clear evidence support."
    assert result.model == "fake-judge-model"
    assert len(judge.calls) == 1


def test_judge_cache_avoids_recomputation(tmp_path):
    cache = JudgeCacheManager(cache_dir=tmp_path)
    res1 = JudgeResult(
        correctness=1.0,
        faithfulness=0.95,
        citation_support=1.0,
        rationale="Cached rationale",
        model="fake-model",
        prompt_version="v1",
        provider_revision="rev-0",
        timestamp="2026-08-18T12:00:00Z",
    )

    cache.put(
        case_id="case-1",
        answer="An answer",
        evidence_texts=["Evidence 1"],
        model="fake-model",
        prompt_version="v1",
        result=res1,
    )

    cached = cache.get(
        case_id="case-1",
        answer="An answer",
        evidence_texts=["Evidence 1"],
        model="fake-model",
        prompt_version="v1",
    )

    assert cached is not None
    assert cached.rationale == "Cached rationale"
    assert cached.correctness == 1.0

    # Cache miss on different answer
    miss = cache.get(
        case_id="case-1",
        answer="Different answer",
        evidence_texts=["Evidence 1"],
        model="fake-model",
        prompt_version="v1",
    )
    assert miss is None


def test_nvidia_judge_parser_handles_structured_json():
    # Test JSON parser extraction
    raw_output = '```json\n{"correctness": 1.0, "faithfulness": 0.85, "citation_support": 1.0, "rationale": "Accurate."}\n```'
    parsed = NvidiaJudge._parse_judge_response(raw_output)
    assert parsed["correctness"] == 1.0
    assert parsed["faithfulness"] == 0.85
    assert parsed["rationale"] == "Accurate."


def test_nvidia_judge_parser_handles_malformed_json():
    raw_output = "I think it is 100% correct."
    parsed = NvidiaJudge._parse_judge_response(raw_output)
    assert "error" in parsed
    assert parsed["correctness"] is None
