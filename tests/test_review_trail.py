"""Tests for review trail persistence and logging."""
import json
import pytest
from pathlib import Path
from plantdoc_qa.loader import load_and_chunk_documents
from plantdoc_qa.retriever import BM25Retriever
from plantdoc_qa.qa import GroundedQAAgent, GroundedAnswer, read_review_trail

DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"


def test_review_trail_persisted_to_jsonl(tmp_path):
    """Review trail should write an entry per answered query into a JSONL log."""
    log_file = tmp_path / "test_review_log.jsonl"
    chunks = load_and_chunk_documents(DOCS_DIR)
    retriever = BM25Retriever()
    retriever.index(chunks)
    agent = GroundedQAAgent(retriever=retriever, review_log_path=log_file)

    # Ask two questions
    ans1 = agent.answer("What is the trip vibration limit on BFP-402?")
    ans2 = agent.answer("What are the DGA gas limits for transformer TR-04?")

    assert log_file.exists()
    records = read_review_trail(log_file)
    assert len(records) == 2

    r1 = records[0]
    assert r1["query"] == "What is the trip vibration limit on BFP-402?"
    assert r1["answer"] == ans1.answer
    assert "citations" in r1
    assert "retrieved_chunks" in r1
    assert len(r1["retrieved_chunks"]) > 0
    assert r1["reviewer_status"] == "pending"
    assert "timestamp" in r1

    r2 = records[1]
    assert r2["query"] == "What are the DGA gas limits for transformer TR-04?"
    assert len(r2["citations"]) > 0


def test_review_trail_handles_refusal_logging(tmp_path):
    """Refused/unanswerable queries are also logged for auditing with status pending."""
    log_file = tmp_path / "test_review_log.jsonl"
    chunks = load_and_chunk_documents(DOCS_DIR)
    retriever = BM25Retriever()
    retriever.index(chunks)
    agent = GroundedQAAgent(retriever=retriever, review_log_path=log_file)

    ans = agent.answer("How do I bake sourdough bread in a blast furnace?")
    assert ans.status == "refused_insufficient_evidence"

    records = read_review_trail(log_file)
    assert len(records) == 1
    assert records[0]["reviewer_status"] == "pending"
    assert records[0]["citations"] == []
