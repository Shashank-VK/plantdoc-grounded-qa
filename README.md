# plantdoc-grounded-qa

Grounded operational plant-doc Q&A agent with verifiable extractive citations, audit review trails, and automated hallucination benchmarking against an ungrounded negative control.

---

## Highlights

- **Standard Library Runtime**: Built entirely with Python stdlib (no PyTorch, no LangChain, no external vector databases, no paid API keys). Only `pytest` is used for dev/testing.
- **Genuine Local BM25 Retriever**: Pure-Python Okapi BM25 ranking with term-frequency saturation ($k_1=1.5$), document length normalization ($b=0.75$), and Robertson-Spärck Jones IDF over document chunks.
- **Verifiable Extractive Citations**: Answers are composed strictly from retrieved chunks, tagged with `[doc_id:chunk_id]` inline citations.
- **Grounded Refusal & Safety Gating**: Refuses queries when retrieved evidence falls below confidence thresholds or lacks token coverage.
- **JSONL Audit Review Trail**: Every query, retrieved chunk, score, citation list, and status is logged to `data/review_log.jsonl` with `reviewer_status: "pending"` for human-in-the-loop signoff.
- **Empirical Grounding Benchmark**: Evaluates grounded vs. ungrounded baseline across 25 evaluation pairs using deterministic citation precision and hallucination checkers.

---

## Measured Benchmark Results

All metrics below come from real execution runs against the 25-pair evaluation benchmark (`eval/eval_set.json`):

| Metric | Grounded QA (Ours) | Ungrounded Baseline | Grounding Delta |
| :--- | :--- | :--- | :--- |
| **Citation Precision** | **92.0%** (0.9200) | **0.0%** (0.0000) | **+92.0%** |
| **Hallucination Rate** | **0.0%** (0.0000) | **100.0%** (1.0000) | **-100.0%** |
| **Out-of-Scope Refusal Rate** | **80.0%** (0.8000) | **0.0%** (0.0000) | **+80.0%** |
| **Evaluated Pairs** | 25 | 25 | — |
| **Test Suite** | **32 / 32 tests passing** (100% green) | — | — |

*Results saved to `eval/results.json` from `python3 -m plantdoc_qa eval`.*

### Metric Definitions
1. **Citation Precision**: Fraction of answer citations whose cited document chunk actually exists in the corpus and directly supports the citing claim via content word overlap.
2. **Hallucination Rate**: Fraction of answer claims unsupported by any retrieved chunk. Answers that explicitly refuse due to insufficient evidence incur 0.0% hallucination.
3. **Out-of-Scope Refusal Rate**: Fraction of out-of-domain or unanswerable queries correctly rejected with an explicit `INSUFFICIENT_EVIDENCE` notice.

---

## Synthetic Plant Document Corpus

The benchmark corpus in `data/docs/` contains 10 synthetic documents covering typical industrial plant equipment, operating envelopes, and logs (each file begins with `# SYNTHETIC DATA` header):

1. `sop_bfp402_boiler_feed_pump.txt`: Multi-stage Boiler Feed Pump BFP-402 startup permissives, vibration limits (4.5 mm/s alarm, 7.1 mm/s trip), seal flush rates.
2. `spec_cc101_centrifugal_compressor.txt`: Centrifugal Compressor CC-101 lubrication specs (ISO VG 46, 450 L reservoir), nitrogen dry gas seal differential (+0.35 bar).
3. `sop_ct03_cooling_tower_water_treatment.txt`: Cooling Tower CT-3 biocide schedule (glutaraldehyde 45 ppm slug dosing), blowdown conductivity setpoint (2400 uS/cm).
4. `sop_st200_steam_turbine_startup.txt`: Steam Turbine ST-200 barring gear run time (4.0 hours at 45 RPM), governor ramp rate (300 RPM/min), differential expansion limits.
5. `sop_cv88_conveyor_emergency_and_tensioning.txt`: Bulk Conveyor CV-88 pull-cord switch spacing (30.0 m max), counterweight ballast mass (4,200 kg).
6. `sop_dc50_distillation_column_limits.txt`: Distillation Column DC-50 flooding differential pressure (85 mbar alarm), reboiler steam pressure (4.2 barg max), relief valve setting (6.0 barg).
7. `sop_etp01_effluent_treatment_neutralization.txt`: Effluent Treatment Plant ETP-1 retention time (8.0 hours), statutory discharge pH limits (6.5 to 8.5), COD limit (250 mg/L).
8. `maint_dc12_baghouse_dust_collector.txt`: Baghouse Dust Collector DC-12 pulse-jet pressure (5.5 - 6.0 bar), pulse duration (120 ms), bag replacement schedule (18 months).
9. `spec_tr04_electrical_transformer.txt`: 25 MVA Transformer TR-04 oil breakdown voltage (50.0 kV), DGA acetylene limit (2.0 ppm), winding temperature trip (105°C).
10. `log_shift_handover_line3_20260314.txt`: Production Line 3 Night Shift handover log, CC-101 cooler fan breaker trip incident, BFP-402 seal filter high-dp swap.

---

## Installation & Quickstart

### Prerequisites
- Python 3.10+ (Tested on Python 3.12)
- Zero external runtime dependencies.

```bash
# Optional: install pytest for testing
pip install -r requirements.txt
```

### 1. Interactive Ask Command
Ask an operational question against the plant document corpus:

```bash
python3 -m plantdoc_qa ask "What is the pre-start lube oil pressure for BFP-402?"
```

Sample output:
```text
================================================================================
PLANTDOC GROUNDED Q&A
================================================================================
Question : What is the pre-start lube oil pressure for BFP-402?
Status   : GROUNDED
--------------------------------------------------------------------------------
ANSWER:
If lube oil pressure fails to reach 1.8 bar within 45 seconds of pump start, the permissive interlock will reject the startup sequence. [sop_bfp402_boiler_feed_pump:sec1] Before initiating the electric motor drive for BFP-402, the shift technician must verify mechanical seal integrity and auxiliary lube oil supply. [sop_bfp402_boiler_feed_pump:sec1] Verify lube oil delivery pressure is at least 1.8 bar (normal operating range 2.2 to 2.5 bar). [sop_bfp402_boiler_feed_pump:sec1]
--------------------------------------------------------------------------------
CITATIONS:
  • [sop_bfp402_boiler_feed_pump:sec1] — STANDARD OPERATING PROCEDURE: SOP-MEC-402
--------------------------------------------------------------------------------
REVIEW TRAIL:
  Timestamp       : 2026-10-14T10:29:54.131063+00:00
  Reviewer Status : pending
  Review Trail Logged to: .../data/review_log.jsonl
================================================================================
```

### 2. Run the Benchmark Evaluation
Run the 25-pair evaluation comparing Grounded QA against the Ungrounded negative control:

```bash
python3 -m plantdoc_qa eval
```

This writes full per-pair records and summary metrics to `eval/results.json`.

---

## One-Command Reproduction Steps

To verify all tests and run the evaluation benchmark in a single command:

```bash
pytest -v && python3 -m plantdoc_qa eval
```

---

## Architecture

```
plantdoc-grounded-qa/
├── data/
│   ├── docs/                      # 10 synthetic plant documents (*.txt)
│   └── review_log.jsonl           # Append-only audit trail for human review
├── eval/
│   ├── eval_set.json              # 25 Q/A evaluation pairs (in-domain & out-of-domain)
│   └── results.json               # Real benchmark evaluation results
├── plantdoc_qa/
│   ├── __init__.py
│   ├── __main__.py                # python -m plantdoc_qa entry point
│   ├── cli.py                     # CLI handlers for ask and eval
│   ├── loader.py                  # Document chunker & metadata parser
│   ├── retriever.py               # Pure-Python BM25 Okapi retriever
│   ├── qa.py                      # Grounded extractive Q&A agent & review trail
│   ├── ungrounded.py              # Negative-control ungrounded baseline
│   └── eval.py                    # Evaluation runner & deterministic checkers
├── tests/
│   ├── test_loader.py             # Doc loading and chunking tests
│   ├── test_retriever.py          # BM25 tokenization, scoring, and ranking tests
│   ├── test_grounded_qa.py        # Extractive answer generation & citation tests
│   ├── test_review_trail.py       # JSONL audit log tests
│   ├── test_ungrounded_qa.py      # Negative control behavior tests
│   ├── test_eval.py               # Citation precision & hallucination checker tests
│   └── test_cli.py                # CLI commands and subprocess tests
├── pytest.ini
├── requirements.txt
└── README.md
```
