# PLAN.md — plantdoc-grounded-qa (Raven proof-of-work artifact)

## Working-backwards (finished state as Raven's hiring manager sees it)
- Public repo: https://github.com/Shashank-VK/plantdoc-grounded-qa
- README: what it does, how to run, test count, measured eval numbers
- Demo flow: ask an ops question → get answer with [doc:chunk] citations + a review trail (retrieved chunks, scores, reviewer status) → show grounded vs ungrounded eval comparison
- Pytest suite fully green, one-command repro (`pytest`)

## Scope gate
- IN: doc adapter ingesting 8–12 synthetic plant docs (SOPs, maintenance records,
  equipment datasheets, shift logs) into a REAL local retriever; cited Q&A with
  per-answer review trail; 25-pair Q/A eval set measuring citation precision and
  hallucination rate with-grounding vs without-grounding.
- OUT: real plant data, frontend, new RAG framework, LLM-embedding dependency.
- Hard constraints: NO fake RAG (retrieval must genuinely happen — no hardcoded
  answers); pure-Python/BM25-TF-IDF-style local retriever (no model downloads, no
  API keys); stdlib-only runtime so `pytest` just works; every metric from a real
  run, never estimated; no secrets; no invented claims.

## Milestones (uphill first)
1. **Retriever (riskiest):** doc loader + chunker (doc id, chunk id, text) +
   pure-Python TF-IDF/BM25 scorer; `retrieve(query, k)` returns scored chunks.
   Proof: tests asserting that a known-phrase query surfaces the right doc/chunk
   and that an unrelated query does not.
2. **Grounded Q&A:** `answer(question)` → retrieve top-k → extractive answer
   composed from retrieved chunks with `[doc:chunk]` citations; refuse/warn when
   nothing relevant is retrieved. Review trail record per answer
   (query, chunks+scores, answer, citations, timestamp, reviewer status,
   feedback). Persisted to JSONL.
3. **Ungrounded baseline:** `answer_ungrounded(question)` — deliberately
   generates a plausible answer WITHOUT retrieval (documented as the negative
   control for the eval; uses only generic templates, never doc content).
4. **Eval harness:** 25 Q/A pairs (question, expected key claims). Automated
   checkers: citation precision (cited chunk supports the claim it cites) and
   hallucination rate (claims unsupported by any retrieved chunk). Run
   grounded vs ungrounded, write `eval/results.json` with real measured numbers.
   Optional LLM judge via free Pollinations endpoint ONLY if it stays keyless
   and deterministic-friendly; default = deterministic string/overlap checkers.
5. **CLI + packaging:** `python -m plantdoc_qa ask "..."`, `python -m plantdoc_qa
   eval`, README, requirements (stdlib-only), tests green, push to GitHub.

## TDD
- Tests written FIRST for each milestone (red), then code (green), then refactor.
- Target: meaningful suite (20+ tests); final count recorded from the real run.

## Review + verify
- FCAR reviewer (fresh context) on repo URL + this spec: binary PASS/FAIL on
  (a) tests green (b) README accurate (c) no invented claims/metrics
  (d) repo live with real code.
- Verify gates: pytest green + count, pushed + `gh repo view` live, README
  documents what/how-to-run/test count, one-command repro, no secrets,
  no invented metrics.

## Anti-invention reminders
- Synthetic docs are labeled synthetic (data/docs/*.txt with header note).
- Eval numbers come only from `python -m plantdoc_qa eval` real output.
- Resume/cite only what the code shows.
