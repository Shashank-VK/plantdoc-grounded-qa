"""Pure-Python BM25 / TF-IDF local retriever with zero external dependencies."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Dict, List, Set, Tuple

from plantdoc_qa.loader import DocumentChunk

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its",
    "itself", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "shouldn't", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "were", "weren't",
    "what", "when", "where", "which", "while", "who", "whom", "why", "with",
    "won't", "would", "wouldn't", "you", "your", "yours", "yourself", "yourselves"
}


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens and preserve compound identifiers."""
    if not text:
        return []

    raw_tokens = re.findall(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*", text.lower())
    tokens = []

    for t in raw_tokens:
        clean = t.strip("-_")
        if not clean or len(clean) < 1:
            continue
        if clean not in STOPWORDS:
            tokens.append(clean)
        # For hyphenated identifiers like bfp-402, index both the compound and components
        if "-" in clean or "_" in clean:
            subparts = re.split(r"[-_]", clean)
            for part in subparts:
                if part and part not in STOPWORDS and part != clean:
                    tokens.append(part)

    return tokens


@dataclass
class ScoredChunk:
    """A retrieved document chunk annotated with its relevance score."""
    chunk: DocumentChunk
    score: float


class BM25Retriever:
    """Local, pure-Python BM25 Okapi retriever."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[DocumentChunk] = []
        self.chunk_tokens: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.total_docs: int = 0
        self.inverted_index: Dict[str, List[int]] = defaultdict(list)
        self.term_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}

    def index(self, chunks: List[DocumentChunk]) -> None:
        """Build the inverted index and BM25 statistics over the chunks."""
        self.chunks = list(chunks)
        self.total_docs = len(chunks)
        self.chunk_tokens = []
        self.doc_lens = []
        self.term_freqs = []
        self.inverted_index = defaultdict(list)
        self.idf = {}

        if self.total_docs == 0:
            self.avg_doc_len = 0.0
            return

        doc_frequency: Dict[str, int] = Counter()

        for idx, chunk in enumerate(self.chunks):
            # Include title, doc_id and body text in searchable representation
            combined_text = f"{chunk.title} {chunk.doc_id} {chunk.text}"
            tokens = tokenize(combined_text)
            self.chunk_tokens.append(tokens)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)

            tf = Counter(tokens)
            self.term_freqs.append(tf)

            for term in tf.keys():
                self.inverted_index[term].append(idx)
                doc_frequency[term] += 1

        self.avg_doc_len = sum(self.doc_lens) / self.total_docs

        # Calculate Robertson-Spärck Jones IDF
        for term, df in doc_frequency.items():
            # Standard BM25 IDF formulation with smoothing
            self.idf[term] = math.log(1.0 + (self.total_docs - df + 0.5) / (df + 0.5))

    def retrieve(self, query: str, k: int = 3, min_score: float = 0.0) -> List[ScoredChunk]:
        """Retrieve top-k chunks matching the query, ordered by BM25 score descending."""
        query_tokens = tokenize(query)
        if not query_tokens or self.total_docs == 0:
            return []

        doc_scores: Dict[int, float] = defaultdict(float)

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            term_idf = self.idf.get(term, 0.0)
            if term_idf <= 0:
                continue

            posting_list = self.inverted_index[term]
            for doc_idx in posting_list:
                tf = self.term_freqs[doc_idx][term]
                doc_len = self.doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score = term_idf * (numerator / denominator)
                doc_scores[doc_idx] += score

        scored_results: List[ScoredChunk] = []
        for doc_idx, score in doc_scores.items():
            if score >= min_score:
                scored_results.append(ScoredChunk(chunk=self.chunks[doc_idx], score=round(score, 4)))

        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:k]
