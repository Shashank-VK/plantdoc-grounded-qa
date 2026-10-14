"""Command-line interface for plantdoc-grounded-qa (ask and eval commands)."""

import argparse
from pathlib import Path
import sys
from typing import List, Optional

from plantdoc_qa.eval import evaluate_pipeline, load_eval_set, save_results
from plantdoc_qa.loader import load_and_chunk_documents
from plantdoc_qa.qa import GroundedQAAgent
from plantdoc_qa.retriever import BM25Retriever
from plantdoc_qa.ungrounded import answer_ungrounded

DEFAULT_DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"
DEFAULT_EVAL_SET = Path(__file__).parent.parent / "eval" / "eval_set.json"
DEFAULT_RESULTS = Path(__file__).parent.parent / "eval" / "results.json"
DEFAULT_REVIEW_LOG = Path(__file__).parent.parent / "data" / "review_log.jsonl"


def handle_ask(args: argparse.Namespace) -> int:
    """Handle the 'ask' CLI command."""
    docs_dir = Path(args.docs_dir)
    review_log = Path(args.review_log)

    chunks = load_and_chunk_documents(docs_dir)
    retriever = BM25Retriever()
    retriever.index(chunks)

    agent = GroundedQAAgent(retriever=retriever, review_log_path=review_log)
    ans = agent.answer(args.question, k=args.k)

    print("=" * 80)
    print("PLANTDOC GROUNDED Q&A")
    print("=" * 80)
    print(f"Question : {ans.question}")
    print(f"Status   : {ans.status.upper()}")
    print("-" * 80)
    print("ANSWER:")
    print(ans.answer)
    print("-" * 80)

    if ans.citations:
        print("CITATIONS:")
        chunk_map = {c.citation_id: c for c in chunks}
        for cite_id in ans.citations:
            chunk = chunk_map.get(cite_id)
            title = chunk.title if chunk else "Indexed Plant Chunk"
            print(f"  • [{cite_id}] — {title}")
    else:
        print("CITATIONS: None (Refusal or no verified source)")

    print("-" * 80)
    print("REVIEW TRAIL:")
    print(f"  Timestamp       : {ans.review_record['timestamp'] if ans.review_record else 'N/A'}")
    print(f"  Reviewer Status : {ans.review_record['reviewer_status'] if ans.review_record else 'N/A'}")
    print(f"  Review Trail Logged to: {review_log}")
    print("=" * 80)

    return 0


def handle_eval(args: argparse.Namespace) -> int:
    """Handle the 'eval' CLI command."""
    docs_dir = Path(args.docs_dir)
    eval_set_path = Path(args.eval_set)
    output_path = Path(args.output)

    print(f"Loading plant documents from: {docs_dir}")
    chunks = load_and_chunk_documents(docs_dir)
    retriever = BM25Retriever()
    retriever.index(chunks)
    grounded_agent = GroundedQAAgent(retriever=retriever)

    print(f"Loading evaluation set from: {eval_set_path}")
    eval_set = load_eval_set(eval_set_path)

    print(f"Running deterministic evaluation on {len(eval_set)} pairs (Grounded vs Ungrounded)...")
    result = evaluate_pipeline(
        eval_set=eval_set,
        grounded_agent=grounded_agent,
        ungrounded_fn=answer_ungrounded,
        corpus_chunks=chunks,
    )

    save_results(result, output_path)
    print(f"Results written to: {output_path}\n")

    print("=" * 80)
    print("EVALUATION COMPARISON SUMMARY")
    print("=" * 80)
    print(result.summary_table)
    print("=" * 80)

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for CLI commands."""
    parser = argparse.ArgumentParser(
        prog="plantdoc_qa",
        description="plantdoc-grounded-qa: Grounded plant-doc Q&A with extractive citations and review trails."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: ask
    ask_parser = subparsers.add_parser("ask", help="Ask a question against indexed plant docs.")
    ask_parser.add_argument("question", type=str, help="The operational/engineering question to ask.")
    ask_parser.add_argument("--docs-dir", type=str, default=str(DEFAULT_DOCS_DIR), help="Directory of plant docs.")
    ask_parser.add_argument("--review-log", type=str, default=str(DEFAULT_REVIEW_LOG), help="Path to JSONL review log.")
    ask_parser.add_argument("-k", type=int, default=3, help="Number of chunks to retrieve.")

    # Subcommand: eval
    eval_parser = subparsers.add_parser("eval", help="Run benchmark evaluation measuring citation precision & hallucination rate.")
    eval_parser.add_argument("--eval-set", type=str, default=str(DEFAULT_EVAL_SET), help="Path to eval_set.json.")
    eval_parser.add_argument("--docs-dir", type=str, default=str(DEFAULT_DOCS_DIR), help="Directory of plant docs.")
    eval_parser.add_argument("--output", type=str, default=str(DEFAULT_RESULTS), help="Path to write results.json.")
    eval_parser.add_argument("-k", type=int, default=3, help="Number of chunks to retrieve.")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "ask":
        return handle_ask(args)
    elif args.subcommand == "eval":
        return handle_eval(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
