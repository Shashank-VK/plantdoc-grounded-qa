"""Grounded extractive Q&A agent with citation tracking and review trail logging."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

from plantdoc_qa.loader import DocumentChunk
from plantdoc_qa.retriever import BM25Retriever, ScoredChunk, tokenize, STOPWORDS


@dataclass
class GroundedAnswer:
    """Structured response from the grounded Q&A agent."""
    question: str
    answer: str
    citations: List[str]
    retrieved_chunks: List[ScoredChunk]
    status: str  # "grounded" | "refused_insufficient_evidence"
    review_record: Optional[Dict] = None


def read_review_trail(log_path: Path | str) -> List[Dict]:
    """Read all entries from a JSONL review log file."""
    p = Path(log_path)
    if not p.exists():
        return []
    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def append_review_record(log_path: Path | str, record: Dict) -> None:
    """Append a single record to the JSONL review log."""
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_sentences(text: str) -> List[str]:
    """Split chunk text into clean candidate sentences/assertions."""
    lines = text.strip().splitlines()
    candidates: List[str] = []

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        # Strip section numbering like "1. PRE-STARTUP..."
        if re.match(r"^\d+\.\s+[A-Z0-9\s\-_,()]+$", line_clean):
            continue
        # Split on sentence boundaries within line
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line_clean)
        for p in parts:
            p_str = p.strip()
            if len(p_str) > 12:
                candidates.append(p_str)

    return candidates


def score_sentence(sentence: str, query_tokens: List[str], idf_map: Dict[str, float]) -> float:
    """Score a candidate sentence using query token weights and informative density."""
    sent_tokens = set(tokenize(sentence))
    if not sent_tokens:
        return 0.0

    overlap = sent_tokens.intersection(set(query_tokens))
    # IDF weighted score
    score = sum(idf_map.get(t, 1.0) for t in overlap)

    # Boost for numerical values and engineering units
    if re.search(r"\d+(?:\.\d+)?\s*(?:bar|barg|ppm|rpm|cst|l/min|m3/h|mm/s|mbar|°c|kv|pa|kg|m/s|hours|months)", sentence, re.IGNORECASE):
        score += 2.0

    return score


class GroundedQAAgent:
    """Agent that answers plant engineering queries extractively with verifiable citations."""

    def __init__(
        self,
        retriever: BM25Retriever,
        review_log_path: Optional[Path | str] = None,
        min_score: float = 6.0,
        min_coverage: float = 0.35,
    ):
        self.retriever = retriever
        self.review_log_path = Path(review_log_path) if review_log_path else None
        self.min_score = min_score
        self.min_coverage = min_coverage

    def answer(self, question: str, k: int = 3) -> GroundedAnswer:
        """Retrieve top-k chunks, compose an extractive cited answer or refuse."""
        query_clean = question.strip()
        query_tokens = [t for t in tokenize(query_clean) if t not in STOPWORDS]
        now_iso = datetime.now(timezone.utc).isoformat()

        if not query_tokens:
            return self._refuse(query_clean, now_iso, "empty_query")

        scored_chunks = self.retriever.retrieve(query_clean, k=k, min_score=0.1)

        # Confidence gating
        is_confident = False
        if scored_chunks:
            top = scored_chunks[0]
            top_chunk_tokens = set(tokenize(f"{top.chunk.title} {top.chunk.doc_id} {top.chunk.text}"))
            matched_tokens = [t for t in query_tokens if t in top_chunk_tokens]
            coverage = len(matched_tokens) / len(query_tokens) if query_tokens else 0.0

            if top.score >= self.min_score and coverage >= self.min_coverage:
                is_confident = True

        if not is_confident or not scored_chunks:
            return self._refuse(query_clean, now_iso, "insufficient_evidence", scored_chunks)

        # Relevant chunks found -> extract best sentences
        idf_map = self.retriever.idf
        extracted_citations = set()
        ranked_sentences: List[Tuple[str, str, float]] = []

        for rank, sc in enumerate(scored_chunks):
            # Prioritize top-ranked chunks
            rank_weight = max(1.0, 5.0 - rank * 1.5)
            sentences = split_sentences(sc.chunk.text)
            for sent in sentences:
                s_score = score_sentence(sent, query_tokens, idf_map)
                if s_score > 0.0:
                    total_s_score = s_score * 2.0 + (sc.score * 0.3) + rank_weight
                    ranked_sentences.append((sent, sc.chunk.citation_id, total_s_score))

        # Sort sentences descending by score
        ranked_sentences.sort(key=lambda x: x[2], reverse=True)

        chosen_claims: List[Tuple[str, str]] = []
        seen_stems = set()

        for sent, citation_id, _ in ranked_sentences:
            stem = sent[:35].lower()
            if stem in seen_stems:
                continue
            seen_stems.add(stem)
            chosen_claims.append((sent, citation_id))
            extracted_citations.add(citation_id)
            if len(chosen_claims) >= 3:
                break

        # Fallback to first sentence of top chunk if no individual sentence matched
        if not chosen_claims:
            top_chunk = scored_chunks[0].chunk
            sentences = split_sentences(top_chunk.text)
            fallback_sent = sentences[0] if sentences else top_chunk.text.strip().splitlines()[0]
            chosen_claims.append((fallback_sent, top_chunk.citation_id))
            extracted_citations.add(top_chunk.citation_id)

        # Format extractive cited answer
        claim_lines = [f"{sent} [{cid}]" for sent, cid in chosen_claims]
        answer_text = " ".join(claim_lines)
        citations_list = sorted(list(extracted_citations))
        status = "grounded"

        retrieved_summary = [
            {
                "citation_id": sc.chunk.citation_id,
                "doc_id": sc.chunk.doc_id,
                "chunk_id": sc.chunk.chunk_id,
                "title": sc.chunk.title,
                "score": sc.score,
                "text_preview": sc.chunk.text[:120] + "...",
            }
            for sc in scored_chunks
        ]

        record = {
            "timestamp": now_iso,
            "query": query_clean,
            "answer": answer_text,
            "citations": citations_list,
            "retrieved_chunks": retrieved_summary,
            "reviewer_status": "pending",
            "reviewer_feedback": None,
        }

        if self.review_log_path:
            append_review_record(self.review_log_path, record)

        return GroundedAnswer(
            question=query_clean,
            answer=answer_text,
            citations=citations_list,
            retrieved_chunks=scored_chunks,
            status=status,
            review_record=record,
        )

    def _refuse(
        self,
        query: str,
        now_iso: str,
        reason: str,
        retrieved_chunks: Optional[List[ScoredChunk]] = None
    ) -> GroundedAnswer:
        """Construct refusal response when grounded evidence is insufficient."""
        answer_text = (
            f"INSUFFICIENT_EVIDENCE: No sufficiently relevant plant documentation was "
            f"retrieved to answer the query '{query}'. Please consult plant engineering records "
            f"or site safety officers directly."
        )
        citations: List[str] = []
        status = "refused_insufficient_evidence"

        retrieved_summary = []
        if retrieved_chunks:
            retrieved_summary = [
                {
                    "citation_id": sc.chunk.citation_id,
                    "doc_id": sc.chunk.doc_id,
                    "chunk_id": sc.chunk.chunk_id,
                    "title": sc.chunk.title,
                    "score": sc.score,
                    "text_preview": sc.chunk.text[:120] + "...",
                }
                for sc in retrieved_chunks
            ]

        record = {
            "timestamp": now_iso,
            "query": query,
            "answer": answer_text,
            "citations": citations,
            "retrieved_chunks": retrieved_summary,
            "reviewer_status": "pending",
            "reviewer_feedback": None,
        }

        if self.review_log_path:
            append_review_record(self.review_log_path, record)

        return GroundedAnswer(
            question=query,
            answer=answer_text,
            citations=citations,
            retrieved_chunks=retrieved_chunks or [],
            status=status,
            review_record=record,
        )
