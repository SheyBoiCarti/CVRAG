import pytest
from evaluation.schema import GoldenCase, FactSpec
from evaluation.metrics.answers import (
    evaluate_answer,
    match_fact,
    aggregate_answer_metrics,
    AnswerCaseMetrics,
)
from rag.models import AnswerResult


def test_fact_matching_exact_phrase():
    fact = FactSpec("exact_phrase", "Cum Laude")
    assert match_fact("He graduated with Cum Laude honours.", fact)
    assert match_fact("cum laude distinction", fact)
    assert not match_fact("CumLaude", fact)


def test_fact_matching_token_phrase_and_boundaries():
    fact = FactSpec("token_phrase", "Software Engineer")
    assert match_fact("He was a Software Engineer at Wirk.", fact)
    assert match_fact("Software Engineer & Game Developer", fact)
    assert not match_fact("He engineered software.", fact)


def test_fact_matching_alias_set():
    fact = FactSpec("alias_set", "Unreal Engine 5", aliases=("UE5", "Unreal 5"))
    assert match_fact("Built games with UE5.", fact)
    assert match_fact("Used Unreal Engine 5 extensively.", fact)
    assert match_fact("Mastered Unreal 5 mechanics.", fact)
    assert not match_fact("Used Unity.", fact)


def test_answer_metrics_abstentions_and_coverage():
    case_ans = GoldenCase.from_dict({
        "id": "c1",
        "category": "straightforward",
        "question": "q",
        "history": [],
        "answerable": True,
        "reference_answer": "Software Engineer",
        "required_facts": [{"matcher": "token_phrase", "value": "Software Engineer", "aliases": []}],
        "forbidden_facts": [{"matcher": "token_phrase", "value": "Doctor", "aliases": []}],
        "evidence_groups": [],
    })

    # Case 1: Correct answer
    res1 = evaluate_answer(case_ans, AnswerResult("He is a Software Engineer at Wirk.", decision="answered"))
    assert res1.required_fact_coverage == 1.0
    assert res1.forbidden_fact_violation == 0.0
    assert res1.incorrect_abstention == 0.0
    assert res1.correct_abstention == 0.0

    # Case 2: Incorrect abstention
    res2 = evaluate_answer(case_ans, AnswerResult("I cannot find information.", decision="abstained"))
    assert res2.incorrect_abstention == 1.0
    assert res2.required_fact_coverage == 0.0
    assert res2.correct_abstention == 0.0

    # Case 3: Forbidden fact violation
    res3 = evaluate_answer(case_ans, AnswerResult("He is a Software Engineer and Doctor.", decision="answered"))
    assert res3.required_fact_coverage == 1.0
    assert res3.forbidden_fact_violation == 1.0

    # Case 4: Unanswerable case correctly abstained
    case_unans = GoldenCase.from_dict({
        "id": "u1",
        "category": "unanswerable",
        "question": "q",
        "history": [],
        "answerable": False,
        "reference_answer": "I do not have information.",
        "required_facts": [],
        "forbidden_facts": [],
        "evidence_groups": [],
    })
    res4 = evaluate_answer(case_unans, AnswerResult("I do not have enough information.", decision="abstained"))
    assert res4.correct_abstention == 1.0
    assert res4.incorrect_abstention == 0.0

    # Case 5: Unanswerable case falsely answered (hallucination)
    res5 = evaluate_answer(case_unans, AnswerResult("His phone number is 12345.", decision="answered"))
    assert res5.correct_abstention == 0.0
    assert res5.incorrect_abstention == 0.0


def test_exact_match_for_exact_term_category():
    case_exact = GoldenCase.from_dict({
        "id": "et-01",
        "category": "exact_term",
        "question": "What honor?",
        "history": [],
        "answerable": True,
        "reference_answer": "Cum Laude",
        "required_facts": [],
        "forbidden_facts": [],
        "evidence_groups": [],
    })
    res_match = evaluate_answer(case_exact, AnswerResult("Cum Laude", decision="answered"))
    assert res_match.exact_match == 1.0

    res_no_match = evaluate_answer(case_exact, AnswerResult("He graduated Cum Laude with honors.", decision="answered"))
    assert res_no_match.exact_match == 0.0


def test_aggregate_answer_metrics():
    case_ans = GoldenCase.from_dict({
        "id": "c1", "category": "straightforward", "question": "q", "history": [], "answerable": True,
        "reference_answer": "ans", "required_facts": [{"matcher": "token_phrase", "value": "Fact", "aliases": []}],
        "forbidden_facts": [], "evidence_groups": [],
    })
    case_unans = GoldenCase.from_dict({
        "id": "u1", "category": "unanswerable", "question": "q", "history": [], "answerable": False,
        "reference_answer": "ans", "required_facts": [], "forbidden_facts": [], "evidence_groups": [],
    })
    m1 = evaluate_answer(case_ans, AnswerResult("Contains Fact", decision="answered"))
    m2 = evaluate_answer(case_unans, AnswerResult("I do not know", decision="abstained"))

    agg = aggregate_answer_metrics([m1, m2])
    assert agg["answerable_cases"] == 1
    assert agg["unanswerable_cases"] == 1
    assert agg["required_fact_coverage"] == 1.0
    assert agg["correct_abstention_rate"] == 1.0
    assert agg["incorrect_abstention_rate"] == 0.0
