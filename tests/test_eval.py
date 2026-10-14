"""Tests for evaluation harness and deterministic checkers."""
import json
import pytest
from pathlib import Path
from plantdoc_qa.loader import load_and_chunk_documents
from plantdoc_qa.retriever import BM25Retriever
from plantdoc_qa.qa import GroundedQAAgent
from plantdoc_qa.ungrounded import answer_ungrounded
from plantdoc_qa.eval import (
    evaluate_pipeline,
    check_citation_precision,
    check_hallucination_rate,
    load_eval_set,
    EvaluationResult,
)

EVAL_FILE = Path(__file__).parent.parent / "eval" / "eval_set.json"
DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"


def test_eval_set_exists_and_has_25_pairs():
    """eval/eval_set.json must exist and contain exactly 25 Q/A pairs."""
    eval_set = load_eval_set(EVAL_FILE)
    assert len(eval_set) == 25, f"Expected 25 pairs, got {len(eval_set)}"
    for item in eval_set:
        assert "id" in item
        assert "question" in item
        assert "expected_key_claims" in item
        assert isinstance(item["expected_key_claims"], list)


def test_check_citation_precision_grounded():
    """A claim citing its source chunk with matching content should have 1.0 precision."""
    chunks = load_and_chunk_documents(DOCS_DIR)
    chunk_map = {c.citation_id: c.text for c in chunks}
    cid = chunks[0].citation_id
    sample_text = chunks[0].text.splitlines()[-1]

    answer = f"{sample_text} [{cid}]"
    precision, verified, total = check_citation_precision(answer, chunk_map)
    assert precision == 1.0
    assert verified == 1
    assert total == 1


def test_check_citation_precision_fake_or_ungrounded():
    """An answer with fake or invalid citations must have 0.0 precision."""
    chunks = load_and_chunk_documents(DOCS_DIR)
    chunk_map = {c.citation_id: c.text for c in chunks}

    answer = "Compressor bearing temperature must stay under 90C. [fake_doc:chunk_99]"
    precision, verified, total = check_citation_precision(answer, chunk_map)
    assert precision == 0.0
    assert total == 1


def test_check_hallucination_rate_grounded_vs_ungrounded():
    """Extract from chunk should have 0.0 hallucination rate; generic invented claim should be high."""
    chunks = load_and_chunk_documents(DOCS_DIR)
    retrieved_texts = [c.text for c in chunks if "sop_bfp402" in c.doc_id]

    grounded_ans = (
        "Radial bearing metal temperatures (TE-402-1 through TE-402-4) must remain below 80°C. "
        "[sop_bfp402_boiler_feed_pump:sec3]"
    )
    hr_grounded, unsupp_g, tot_g = check_hallucination_rate(grounded_ans, retrieved_texts)
    assert hr_grounded == 0.0

    ungrounded_ans = (
        "Standard cryogenic rockets require liquid hydrogen cooling at minus 250 degrees celsius. "
        "The hypergolic propellant burns at high chamber pressures."
    )
    hr_ungrounded, unsupp_u, tot_u = check_hallucination_rate(ungrounded_ans, retrieved_texts)
    assert hr_ungrounded > 0.5


def test_run_eval_comparison(tmp_path):
    """Running evaluation across pairs yields real measurable comparison."""
    chunks = load_and_chunk_documents(DOCS_DIR)
    retriever = BM25Retriever()
    retriever.index(chunks)
    grounded_agent = GroundedQAAgent(retriever=retriever)

    # Run on a mini slice of 3 items
    eval_set = load_eval_set(EVAL_FILE)[:3]
    result = evaluate_pipeline(eval_set, grounded_agent, answer_ungrounded, chunks)

    assert isinstance(result, EvaluationResult)
    assert result.num_pairs == 3
    assert result.grounded_citation_precision >= 0.8
    assert result.ungrounded_citation_precision == 0.0
    assert result.grounded_hallucination_rate < result.ungrounded_hallucination_rate
