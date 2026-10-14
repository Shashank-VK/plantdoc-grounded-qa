"""Tests for grounded extractive Q&A agent."""
import pytest
from pathlib import Path
from plantdoc_qa.loader import load_and_chunk_documents
from plantdoc_qa.retriever import BM25Retriever
from plantdoc_qa.qa import GroundedQAAgent, GroundedAnswer

DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"


@pytest.fixture(scope="module")
def qa_agent():
    chunks = load_and_chunk_documents(DOCS_DIR)
    retriever = BM25Retriever()
    retriever.index(chunks)
    return GroundedQAAgent(retriever=retriever)


def test_grounded_qa_bfp402_pressure(qa_agent):
    """Grounded QA should extract exact lube oil pressure and cite the right chunk."""
    res = qa_agent.answer("What is the pre-start lube oil pressure for BFP-402?", k=3)
    assert isinstance(res, GroundedAnswer)
    assert res.status == "grounded"
    assert "1.8 bar" in res.answer
    assert any("sop_bfp402" in c for c in res.citations)
    assert "[sop_bfp402_boiler_feed_pump:sec1]" in res.answer or any(c in res.answer for c in res.citations)


def test_grounded_qa_refusal_on_irrelevant_query(qa_agent):
    """Grounded QA must refuse/warn explicitly when question is completely outside plant docs."""
    res = qa_agent.answer("What is the optimal flight altitude for a Boeing 787 aircraft?", k=3)
    assert res.status == "refused_insufficient_evidence"
    assert len(res.citations) == 0
    assert "INSUFFICIENT_EVIDENCE" in res.answer or "Warning:" in res.answer or "insufficient" in res.answer.lower()


def test_grounded_qa_cc101_oil_capacity(qa_agent):
    """Grounded QA extracts CC-101 oil capacity with citation."""
    res = qa_agent.answer("What is the total reservoir capacity and oil grade for CC-101 compressor?", k=3)
    assert res.status == "grounded"
    assert "450" in res.answer or "VG 46" in res.answer
    assert any("cc101" in c for c in res.citations)


def test_grounded_qa_etp_ph_limits(qa_agent):
    """Grounded QA extracts ETP-1 final discharge statutory pH limits."""
    res = qa_agent.answer("What are the final discharge pH limits for ETP-1?", k=3)
    assert res.status == "grounded"
    assert "6.5" in res.answer and "8.5" in res.answer
    assert any("etp01" in c for c in res.citations)


def test_grounded_qa_citations_are_strictly_valid(qa_agent):
    """All citations returned by grounded QA must correspond to indexed chunks."""
    res = qa_agent.answer("What is the pull cord switch spacing on conveyor CV-88?", k=3)
    assert res.status == "grounded"
    assert len(res.citations) > 0
    all_citation_ids = {c.citation_id for c in qa_agent.retriever.chunks}
    for c in res.citations:
        assert c in all_citation_ids
