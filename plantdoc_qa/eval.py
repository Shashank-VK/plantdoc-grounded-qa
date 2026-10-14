"""Evaluation harness measuring citation precision and hallucination rate."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Callable, Dict, List, Optional, Tuple

from plantdoc_qa.loader import DocumentChunk
from plantdoc_qa.retriever import STOPWORDS, tokenize


@dataclass
class EvaluationResult:
    """Aggregated evaluation metrics comparing grounded vs ungrounded pipelines."""
    num_pairs: int
    grounded_citation_precision: float
    grounded_hallucination_rate: float
    grounded_refusal_rate: float
    ungrounded_citation_precision: float
    ungrounded_hallucination_rate: float
    ungrounded_refusal_rate: float
    pair_results: List[Dict]
    timestamp: str
    summary_table: str


def load_eval_set(eval_path: Path | str) -> List[Dict]:
    """Load evaluation pairs from JSON file."""
    p = Path(eval_path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of evaluation pairs in {eval_path}")
    return data


def extract_claims_and_citations(answer: str) -> List[Tuple[str, Optional[str]]]:
    """Extract individual sentences and their associated [doc:chunk] citation if present."""
    # Split on sentence ends
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    results: List[Tuple[str, Optional[str]]] = []

    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        cite_match = re.search(r"\[([a-zA-Z0-9_\-]+:[a-zA-Z0-9_\-]+)\]", s_clean)
        if cite_match:
            citation_id = cite_match.group(1)
            text_without_cite = re.sub(r"\[[a-zA-Z0-9_\-]+:[a-zA-Z0-9_\-]+\]", "", s_clean).strip()
            results.append((text_without_cite, citation_id))
        else:
            results.append((s_clean, None))

    return results


def check_citation_precision(answer: str, chunk_map: Dict[str, str]) -> Tuple[float, int, int]:
    """
    Compute citation precision = (supported citations) / (total citations claimed).
    A citation is verified if:
      1) The cited chunk exists in chunk_map.
      2) The citing claim shares substantive tokens (word overlap) with the cited chunk text.
    """
    if "INSUFFICIENT_EVIDENCE" in answer or "refused_insufficient_evidence" in answer:
        # Refusal correctly made no false citations
        return 1.0, 0, 0

    all_citations = re.findall(r"\[([a-zA-Z0-9_\-]+:[a-zA-Z0-9_\-]+)\]", answer)
    if not all_citations:
        # If the answer made factual claims without any citation, precision is 0.0
        return 0.0, 0, 1

    claims_with_cites = extract_claims_and_citations(answer)
    verified = 0
    total = len(all_citations)

    for claim_text, cite_id in claims_with_cites:
        if not cite_id:
            continue

        if cite_id not in chunk_map:
            # Fake/hallucinated citation id
            continue

        chunk_text = chunk_map[cite_id]
        claim_tokens = [
            t for t in tokenize(claim_text)
            if t not in STOPWORDS and len(t) >= 2
        ]

        if not claim_tokens:
            verified += 1
            continue

        chunk_tokens_set = set(tokenize(chunk_text))
        overlap = [t for t in claim_tokens if t in chunk_tokens_set]
        overlap_ratio = len(overlap) / len(claim_tokens)

        # Verified if at least 35% of content tokens or >= 3 distinctive tokens match
        if overlap_ratio >= 0.35 or len(overlap) >= 3:
            verified += 1

    precision = round(verified / total, 4) if total > 0 else 0.0
    return precision, verified, total


def check_hallucination_rate(
    answer: str,
    retrieved_chunk_texts: List[str]
) -> Tuple[float, int, int]:
    """
    Compute hallucination rate = (unsupported claims) / (total claims).
    A claim is unsupported if its substantive content is absent from all retrieved chunks.
    Refusals/insufficient evidence responses have 0.0 hallucination rate.
    """
    if "INSUFFICIENT_EVIDENCE" in answer or "refused_insufficient_evidence" in answer:
        return 0.0, 0, 0

    claims_with_cites = extract_claims_and_citations(answer)
    if not claims_with_cites:
        return 0.0, 0, 0

    # Build reference token set from retrieved chunks
    combined_ref_text = " ".join(retrieved_chunk_texts)
    ref_tokens = set(tokenize(combined_ref_text))

    unsupported_claims = 0
    total_claims = 0

    for claim_text, _ in claims_with_cites:
        clean_claim = re.sub(r"\[.*?\]", "", claim_text).strip()
        tokens = [t for t in tokenize(clean_claim) if t not in STOPWORDS and len(t) >= 2]
        if not tokens:
            continue

        total_claims += 1

        if not ref_tokens:
            # No reference retrieved at all -> entire claim is ungrounded / hallucinated
            unsupported_claims += 1
            continue

        matching = [t for t in tokens if t in ref_tokens]
        ratio = len(matching) / len(tokens)

        # If less than 40% of substantive tokens exist in retrieved docs, it is an unsupported claim
        if ratio < 0.40:
            unsupported_claims += 1

    rate = round(unsupported_claims / total_claims, 4) if total_claims > 0 else 0.0
    return rate, unsupported_claims, total_claims


def build_markdown_summary(
    num_pairs: int,
    g_prec: float,
    g_hall: float,
    g_refusal: float,
    u_prec: float,
    u_hall: float,
    u_refusal: float
) -> str:
    """Format evaluation comparison table as markdown."""
    return f"""| Metric | Grounded QA (Ours) | Ungrounded Baseline | Grounding Delta |
| :--- | :--- | :--- | :--- |
| **Citation Precision** | {g_prec * 100:.1f}% | {u_prec * 100:.1f}% | +{(g_prec - u_prec) * 100:.1f}% |
| **Hallucination Rate** | {g_hall * 100:.1f}% | {u_hall * 100:.1f}% | -{(u_hall - g_hall) * 100:.1f}% |
| **Refusal on Out-of-Scope** | {g_refusal * 100:.1f}% | {u_refusal * 100:.1f}% | +{(g_refusal - u_refusal) * 100:.1f}% |
| **Evaluated Pairs** | {num_pairs} | {num_pairs} | — |
"""


def evaluate_pipeline(
    eval_set: List[Dict],
    grounded_agent,
    ungrounded_fn: Callable[[str], object],
    corpus_chunks: List[DocumentChunk]
) -> EvaluationResult:
    """Run evaluation across all pairs for both pipelines and compute verified metrics."""
    chunk_map = {c.citation_id: c.text for c in corpus_chunks}

    g_precisions: List[float] = []
    g_hall_rates: List[float] = []
    g_refusals = 0

    u_precisions: List[float] = []
    u_hall_rates: List[float] = []
    u_refusals = 0

    pair_results: List[Dict] = []

    for item in eval_set:
        qid = item["id"]
        question = item["question"]
        in_domain = item.get("in_domain", True)

        # 1. Grounded Run
        g_ans = grounded_agent.answer(question)
        g_retrieved_texts = [sc.chunk.text for sc in g_ans.retrieved_chunks]
        g_prec, _, _ = check_citation_precision(g_ans.answer, chunk_map)
        g_hall, _, _ = check_hallucination_rate(g_ans.answer, g_retrieved_texts)
        if g_ans.status == "refused_insufficient_evidence":
            if not in_domain:
                g_refusals += 1
        g_precisions.append(g_prec)
        g_hall_rates.append(g_hall)

        # 2. Ungrounded Run
        u_ans = ungrounded_fn(question)
        # Ungrounded has NO retrieved chunks (negative control)
        u_prec, _, _ = check_citation_precision(u_ans.answer, chunk_map)
        u_hall, _, _ = check_hallucination_rate(u_ans.answer, [])
        u_precisions.append(u_prec)
        u_hall_rates.append(u_hall)

        pair_results.append({
            "id": qid,
            "question": question,
            "in_domain": in_domain,
            "grounded": {
                "answer": g_ans.answer,
                "citations": g_ans.citations,
                "status": g_ans.status,
                "citation_precision": g_prec,
                "hallucination_rate": g_hall,
            },
            "ungrounded": {
                "answer": u_ans.answer,
                "citations": u_ans.citations,
                "status": u_ans.status,
                "citation_precision": u_prec,
                "hallucination_rate": u_hall,
            }
        })

    num_pairs = len(eval_set)
    out_of_domain_count = sum(1 for item in eval_set if not item.get("in_domain", True))
    denom_ood = out_of_domain_count if out_of_domain_count > 0 else 1

    avg_g_prec = round(sum(g_precisions) / num_pairs, 4)
    avg_g_hall = round(sum(g_hall_rates) / num_pairs, 4)
    g_refusal_rate = round(g_refusals / denom_ood, 4)

    avg_u_prec = round(sum(u_precisions) / num_pairs, 4)
    avg_u_hall = round(sum(u_hall_rates) / num_pairs, 4)
    u_refusal_rate = round(u_refusals / denom_ood, 4)

    summary_table = build_markdown_summary(
        num_pairs=num_pairs,
        g_prec=avg_g_prec,
        g_hall=avg_g_hall,
        g_refusal=g_refusal_rate,
        u_prec=avg_u_prec,
        u_hall=avg_u_hall,
        u_refusal=u_refusal_rate,
    )

    now_iso = datetime.now(timezone.utc).isoformat()

    return EvaluationResult(
        num_pairs=num_pairs,
        grounded_citation_precision=avg_g_prec,
        grounded_hallucination_rate=avg_g_hall,
        grounded_refusal_rate=g_refusal_rate,
        ungrounded_citation_precision=avg_u_prec,
        ungrounded_hallucination_rate=avg_u_hall,
        ungrounded_refusal_rate=u_refusal_rate,
        pair_results=pair_results,
        timestamp=now_iso,
        summary_table=summary_table,
    )


def save_results(result: EvaluationResult, output_path: Path | str) -> None:
    """Save evaluation metrics and pair records to JSON file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)
