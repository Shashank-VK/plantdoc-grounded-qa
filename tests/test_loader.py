"""Tests for plantdoc document loader and chunker."""
import os
import pytest
from pathlib import Path
from plantdoc_qa.loader import DocumentChunk, load_and_chunk_documents, parse_synthetic_header

DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"


def test_synthetic_docs_exist_and_count():
    """Ensure data/docs contains between 8 and 12 synthetic documents."""
    docs = list(DOCS_DIR.glob("*.txt"))
    assert 8 <= len(docs) <= 12, f"Expected 8-12 docs, found {len(docs)}"


def test_synthetic_header_in_all_docs():
    """All synthetic plant documents must have a SYNTHETIC DATA header note."""
    docs = list(DOCS_DIR.glob("*.txt"))
    assert len(docs) > 0
    for doc_path in docs:
        content = doc_path.read_text(encoding="utf-8")
        assert content.startswith("# SYNTHETIC DATA"), (
            f"Doc {doc_path.name} missing # SYNTHETIC DATA header"
        )
        has_header, cleaned_text = parse_synthetic_header(content)
        assert has_header is True
        assert len(cleaned_text.strip()) > 50


def test_load_and_chunk_documents():
    """Verify document chunker produces structured DocumentChunk objects."""
    chunks = load_and_chunk_documents(DOCS_DIR)
    assert len(chunks) > 10, f"Expected at least 10 chunks, got {len(chunks)}"

    # Check structure of each chunk
    for chunk in chunks:
        assert isinstance(chunk, DocumentChunk)
        assert chunk.doc_id.strip() != ""
        assert chunk.chunk_id.strip() != ""
        assert chunk.citation_id == f"{chunk.doc_id}:{chunk.chunk_id}"
        assert chunk.title.strip() != ""
        assert len(chunk.text.strip()) > 30


def test_chunker_retains_numbers_and_equipment_ids():
    """Check that specific equipment numbers and IDs are preserved in chunks."""
    chunks = load_and_chunk_documents(DOCS_DIR)
    all_text = " ".join(c.text for c in chunks)
    assert "BFP-402" in all_text
    assert "CC-101" in all_text
    assert "CT-3" in all_text
    assert "ST-200" in all_text
    assert "1.8 bar" in all_text
    assert "glutaraldehyde" in all_text


def test_load_nonexistent_directory():
    """Loader should return empty list or raise FileNotFoundError for invalid path."""
    with pytest.raises(FileNotFoundError):
        load_and_chunk_documents(Path("/nonexistent/directory/path/here"))
