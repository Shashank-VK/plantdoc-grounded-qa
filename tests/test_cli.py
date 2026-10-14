"""Tests for the CLI interface (ask and eval commands)."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
from plantdoc_qa.cli import main

ROOT_DIR = Path(__file__).parent.parent


def test_cli_ask_grounded(capsys):
    """'ask' command prints extractive answer, citations, and review trail snippet."""
    exit_code = main(["ask", "What is the pre-start lube oil pressure for BFP-402?"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "1.8 bar" in captured
    assert "sop_bfp402" in captured
    assert "Review Trail Logged" in captured or "REVIEW TRAIL" in captured


def test_cli_ask_refusal(capsys):
    """'ask' on out-of-domain query prints explicit warning/refusal."""
    exit_code = main(["ask", "What is the airspeed velocity of an unladen swallow?"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "INSUFFICIENT_EVIDENCE" in captured or "Warning" in captured


def test_cli_eval(capsys, tmp_path):
    """'eval' command runs evaluation, prints comparison table, and writes results JSON."""
    test_results_path = tmp_path / "results.json"
    exit_code = main(["eval", "--output", str(test_results_path)])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Citation Precision" in captured
    assert "Hallucination Rate" in captured
    assert test_results_path.exists()

    with open(test_results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["num_pairs"] == 25
    assert data["grounded_citation_precision"] > 0.8
    assert data["ungrounded_citation_precision"] == 0.0


def test_cli_module_execution():
    """Verify python -m plantdoc_qa runs successfully as a subprocess."""
    res = subprocess.run(
        [sys.executable, "-m", "plantdoc_qa", "ask", "What is the cooling tower CT-3 conductivity setpoint?"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True
    )
    assert res.returncode == 0
    assert "2400" in res.stdout
    assert "sop_ct03" in res.stdout
