from __future__ import annotations

from rag.verification.citation_parser import parse_claims


def test_single_sentence_single_citation():
    text = "Sheharyar worked as a Software Engineer at Wirk [S1]."
    claims = parse_claims(text)
    assert len(claims) == 1
    claim = claims[0]
    assert claim.claim_id == "C1"
    assert "Sheharyar worked as a Software Engineer at Wirk." in claim.text
    assert claim.citation_ids == ("S1",)
    assert claim.factual is True
    assert claim.importance == "core"


def test_multiple_citations_bracket_format():
    text = "He architected 3 desktop game launchers [S1][S2]."
    claims = parse_claims(text)
    assert len(claims) == 1
    assert claims[0].citation_ids == ("S1", "S2")
    assert claims[0].factual is True


def test_multiple_citations_comma_format():
    text = "He used NVIDIA embeddings and Llama-3.3 [S1, S2]."
    claims = parse_claims(text)
    assert len(claims) == 1
    assert claims[0].citation_ids == ("S1", "S2")


def test_repeated_citations_deduplicated():
    text = "He graduated Cum Laude [S1][S1]."
    claims = parse_claims(text)
    assert len(claims) == 1
    assert claims[0].citation_ids == ("S1",)


def test_punctuation_after_citation():
    text = "He tutored over 20 students [S1], and he served as Tech Lead [S2]."
    claims = parse_claims(text)
    assert len(claims) == 2
    assert claims[0].citation_ids == ("S1",)
    assert claims[1].citation_ids == ("S2",)
    assert claims[0].factual is True
    assert claims[1].factual is True
    assert claims[0].text == "He tutored over 20 students."
    assert claims[1].text == "He served as Tech Lead."


def test_compound_sentence_semicolon_split():
    text = "Sheharyar worked at Wirk [S1]; he also worked at CellZone [S2]."
    claims = parse_claims(text)
    assert len(claims) == 2
    assert claims[0].citation_ids == ("S1",)
    assert claims[1].citation_ids == ("S2",)
    assert "Wirk" in claims[0].text
    assert "CellZone" in claims[1].text


def test_compound_sentence_conjunction_with_subject():
    text = "He built the PRIMED platform [S1], and he integrated PostgreSQL [S2]."
    claims = parse_claims(text)
    assert len(claims) == 2
    assert claims[0].citation_ids == ("S1",)
    assert claims[1].citation_ids == ("S2",)


def test_sentence_without_explicit_second_subject_remains_single_claim():
    text = "He designed and implemented the architecture using Python and Chroma [S1]."
    claims = parse_claims(text)
    assert len(claims) == 1
    assert claims[0].citation_ids == ("S1",)


def test_markdown_list_items():
    text = """
- Built 3 custom desktop game launchers [S1]
- Graduated with Honours in Computer Science Cum Laude [S2]
- Tutored 20 undergraduate students in Data Structures [S3]
"""
    claims = parse_claims(text)
    assert len(claims) == 3
    assert claims[0].citation_ids == ("S1",)
    assert claims[1].citation_ids == ("S2",)
    assert claims[2].citation_ids == ("S3",)
    for c in claims:
        assert c.factual is True
        assert c.importance == "core"


def test_uncited_factual_claim():
    text = "He holds a PhD in Quantum Computing."
    claims = parse_claims(text)
    assert len(claims) == 1
    assert claims[0].citation_ids == ()
    assert claims[0].factual is True
    assert claims[0].importance == "core"


def test_nonfactual_headings_and_questions():
    text = """
### Professional Summary
What roles has Sheharyar held?
He worked as a Tech Lead at NMU [S1].
"""
    claims = parse_claims(text)
    # 3 items: heading (nonfactual), question (nonfactual), affirmative statement (factual)
    assert len(claims) == 3
    assert claims[0].factual is False
    assert claims[1].factual is False
    assert claims[2].factual is True
    assert claims[2].citation_ids == ("S1",)


def test_nonfactual_connective_fragments_and_abstention():
    text = "In summary, I don't have enough supported information in the CV to answer that confidently."
    claims = parse_claims(text)
    for c in claims:
        assert c.factual is False


def test_malformed_citation_tags_do_not_crash():
    text = "He knows Python [S] and C++ [1] and Java [Sabc] [S01] [S2]."
    claims = parse_claims(text)
    assert len(claims) >= 1
    # Only valid S<number> (where number > 0) is parsed, e.g., S1, S2
    all_cits = [cit for c in claims for cit in c.citation_ids]
    assert "S2" in all_cits
    assert "S1" in all_cits or "S01" not in all_cits  # S01 normalizes to S1


def test_html_and_markdown_inertness():
    text = "<script>alert('xss')</script> He worked at Wirk [S1]."
    claims = parse_claims(text)
    assert len(claims) == 1
    assert claims[0].citation_ids == ("S1",)
    assert "<script>" in claims[0].text  # Treated purely as text
