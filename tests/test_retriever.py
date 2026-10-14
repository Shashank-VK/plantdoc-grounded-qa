"""Tests for pure-Python BM25 / TF-IDF local retriever."""
import pytest
from pathlib import Path
from plantdoc_qa.loader import DocumentChunk, load_and_chunk_documents
from plantdoc_qa.retriever import BM25Retriever, tokenize, ScoredChunk

DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"


@pytest.fixture(scope="module")
def corpus():
    return load_and_chunk_documents(DOCS_DIR)


@pytest.fixture(scope="module")
def retriever(corpus):
    r = BM25Retriever()
    r.index(corpus)
    return r


def test_tokenize():
    """Verify tokenizer lowercases, removes punctuation, and filters tokens."""
    tokens = tokenize("Boiler Feed Pump BFP-402, lube oil: 1.8 bar!")
    assert "boiler" in tokens
    assert "feed" in tokens
    assert "pump" in tokens
    assert "bfp-402" in tokens or "bfp402" in tokens or ("bfp" in tokens and "402" in tokens)
    assert "bar" in tokens


def test_retriever_initialization(retriever, corpus):
    """Retriever must index all provided chunks."""
    assert retriever.total_docs == len(corpus)
    assert retriever.avg_doc_len > 0


def test_retriever_known_phrase_bfp402(retriever):
    """Known-phrase query surfaces BFP-402 chunk at rank 1."""
    results = retriever.retrieve("What is the pre-start lube oil pressure for BFP-402?", k=3)
    assert len(results) > 0
    top = results[0]
    assert isinstance(top, ScoredChunk)
    assert "bfp402" in top.chunk.doc_id.lower() or "bfp-402" in top.chunk.text.lower()
    assert top.score > 0
    assert "1.8 bar" in top.chunk.text


def test_retriever_known_phrase_cooling_tower(retriever):
    """Known-phrase query surfaces CT-3 biocide glutaraldehyde schedule."""
    results = retriever.retrieve("glutaraldehyde dosing schedule CT-3 cooling tower", k=3)
    assert len(results) > 0
    top = results[0]
    assert "ct03" in top.chunk.doc_id.lower() or "cooling_tower" in top.chunk.doc_id.lower()
    assert "glutaraldehyde" in top.chunk.text.lower()


def test_retriever_ranking_descending(retriever):
    """Retrieved results must be strictly ordered in descending score."""
    results = retriever.retrieve("steam turbine ST-200 differential expansion", k=5)
    assert len(results) >= 2
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retriever_unrelated_query(retriever):
    """Unrelated query returns empty or very low scoring chunks."""
    results = retriever.retrieve("quantum cryptographic entanglement in superconductors", k=3)
    # Either empty results or score below minimal threshold
    if results:
        assert results[0].score < 1.0


def test_retriever_empty_query(retriever):
    """Empty or whitespace query should return empty list."""
    assert retriever.retrieve("", k=3) == []
    assert retriever.retrieve("   ", k=3) == []


def test_retriever_k_limit(retriever):
    """Retriever respects k parameter."""
    res_1 = retriever.retrieve("pressure temperature", k=1)
    res_3 = retriever.retrieve("pressure temperature", k=3)
    assert len(res_1) <= 1
    assert len(res_3) <= 3
