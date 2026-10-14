"""Document loader and chunker for synthetic plant documents."""

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DocumentChunk:
    """Represents a discrete semantic chunk of a plant document."""
    doc_id: str
    chunk_id: str
    citation_id: str
    title: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


def parse_synthetic_header(raw_text: str) -> Tuple[bool, str]:
    """Check for synthetic data banner and strip it from the body."""
    lines = raw_text.splitlines()
    has_header = False
    clean_lines = []

    for line in lines:
        if line.startswith("# SYNTHETIC DATA"):
            has_header = True
            continue
        clean_lines.append(line)

    return has_header, "\n".join(clean_lines).strip()


def extract_title_and_metadata(text: str) -> Tuple[str, Dict[str, str], str]:
    """Extract document title and metadata lines from top of document."""
    lines = text.splitlines()
    title = ""
    metadata = {}
    content_start_idx = 0

    for i, line in enumerate(lines):
        line_str = line.strip()
        if not line_str:
            continue
        if not title and (line_str.startswith("STANDARD OPERATING PROCEDURE:") or
                          line_str.startswith("TECHNICAL SPECIFICATION") or
                          line_str.startswith("MAINTENANCE MANUAL") or
                          line_str.startswith("OPERATIONS SHIFT HANDOVER LOG:")):
            title = line_str
            continue
        if ":" in line_str and any(k in line_str for k in ["EQUIPMENT:", "REVISION:", "UNIT:", "LOCATION:"]):
            parts = [p.strip() for p in line_str.split("|")]
            for p in parts:
                if ":" in p:
                    k, v = p.split(":", 1)
                    metadata[k.strip().lower()] = v.strip()
            continue
        # Section header starts with number e.g. "1. "
        if re.match(r"^\d+\.\s+", line_str):
            content_start_idx = i
            break

    if not title:
        title = lines[0].strip() if lines else "Plant Document"

    remaining_text = "\n".join(lines[content_start_idx:]).strip()
    return title, metadata, remaining_text


def chunk_document_text(doc_id: str, title: str, metadata: Dict[str, str], text: str) -> List[DocumentChunk]:
    """Split document into coherent section chunks."""
    # Pattern to match numbered sections like "1. PRE-STARTUP..." or "2. "
    section_pattern = re.compile(r"(?=^\d+\.\s+[A-Z0-9\s\-_,()]+$)", re.MULTILINE)
    raw_sections = section_pattern.split(text)

    chunks = []
    chunk_index = 0
    for section in raw_sections:
        clean_sec = section.strip()
        if not clean_sec:
            continue

        chunk_id = f"chunk_{chunk_index}"
        first_line = clean_sec.splitlines()[0].strip()
        # If it looks like "1. PRE-STARTUP...", create a descriptive section id
        sec_num_match = re.match(r"^(\d+)\.", first_line)
        if sec_num_match:
            chunk_id = f"sec{sec_num_match.group(1)}"

        citation_id = f"{doc_id}:{chunk_id}"
        chunk = DocumentChunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            citation_id=citation_id,
            title=title,
            text=clean_sec,
            metadata=dict(metadata)
        )
        chunks.append(chunk)
        chunk_index += 1

    # Fallback if no numbered sections found
    if not chunks and text.strip():
        chunk = DocumentChunk(
            doc_id=doc_id,
            chunk_id="chunk_0",
            citation_id=f"{doc_id}:chunk_0",
            title=title,
            text=text.strip(),
            metadata=dict(metadata)
        )
        chunks.append(chunk)

    return chunks


def load_and_chunk_documents(docs_dir: Path | str) -> List[DocumentChunk]:
    """Load all .txt files from docs_dir, chunk them, and return a list of DocumentChunk."""
    path = Path(docs_dir)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {docs_dir}")

    all_chunks: List[DocumentChunk] = []
    file_paths = sorted(path.glob("*.txt"))

    for file_path in file_paths:
        raw_content = file_path.read_text(encoding="utf-8")
        has_header, clean_text = parse_synthetic_header(raw_content)
        doc_id = file_path.stem
        title, metadata, body = extract_title_and_metadata(clean_text)
        metadata["synthetic"] = str(has_header).lower()
        metadata["filename"] = file_path.name

        chunks = chunk_document_text(doc_id, title, metadata, body)
        all_chunks.extend(chunks)

    return all_chunks
