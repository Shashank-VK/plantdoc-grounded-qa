"""Tests for ungrounded negative-control baseline."""
import pytest
from plantdoc_qa.ungrounded import answer_ungrounded, UngroundedAnswer


def test_ungrounded_returns_ungrounded_answer_type():
    """answer_ungrounded returns an UngroundedAnswer with negative control status."""
    res = answer_ungrounded("What is the pre-start lube oil pressure for BFP-402?")
    assert isinstance(res, UngroundedAnswer)
    assert res.status == "ungrounded_baseline"
    assert len(res.answer.strip()) > 20


def test_ungrounded_does_not_use_retrieval_or_plant_doc_citations():
    """Ungrounded answer should not contain valid plant doc citation IDs."""
    res = answer_ungrounded("What is the glutaraldehyde dosing schedule CT-3 cooling tower?")
    assert "sop_ct03_cooling_tower_water_treatment" not in res.citations
    assert not any("sop_" in c or "spec_" in c or "maint_" in c or "log_" in c for c in res.citations)


def test_ungrounded_provides_plausible_generic_response():
    """Ungrounded answers provide plausible engineering responses across varied topics."""
    queries = [
        "What is the maximum bearing temperature for compressor CC-101?",
        "What is the required pull cord switch spacing on conveyor CV-88?",
        "What are the transformer TR-04 DGA limits?",
    ]
    for q in queries:
        ans = answer_ungrounded(q)
        assert ans.question == q
        assert ans.answer.strip() != ""
