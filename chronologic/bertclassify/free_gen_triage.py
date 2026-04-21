#!/usr/bin/env python3
"""free_gen_triage.py

Shared utilities for the free-generation evaluation pipeline.

Provides:
  - load_benchmark()        : load benchmark JSONL into a dict keyed by question_number
  - count_words()           : word-count helper
  - trim_quotes()           : strip surrounding quotes from a string
  - _derive_output_stem()   : derive a file-stem from an input path
  - write_deberta_tsv()     : write alternating gt/model rows for DeBERTa inference
  - write_manual_jsonl()    : write items that need human review to JSONL

These functions are imported by free_gen_eval.py.
"""

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    """Return the number of whitespace-separated tokens in *text*."""
    return len(text.split()) if text else 0


def trim_quotes(text: str) -> str:
    """Strip one layer of matching surrounding quotes ('' or \"\") from *text*."""
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return text


def _derive_output_stem(input_path: Path) -> str:
    """Return a file-stem suitable for intermediate output files.

    Strips the extension and any trailing timestamp suffix so that re-runs
    on the same evaluation file overwrite earlier intermediates.

    Example:
        free_gen_gpt-5.4_20260318_180622.json  →  free_gen_gpt-5.4_20260318_180622
    """
    return Path(input_path).stem


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------

def load_benchmark(benchmark_path: Path) -> dict:
    """Load a benchmark JSONL file into a dict keyed by question_number (str).

    Each value is the full benchmark record dict.
    """
    benchmark = {}
    with open(benchmark_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qnum = str(record.get("question_number", ""))
            if qnum:
                benchmark[qnum] = record
    return benchmark


# ---------------------------------------------------------------------------
# TSV / JSONL writers
# ---------------------------------------------------------------------------

def write_deberta_tsv(items: list, output_path: Path) -> None:
    """Write *items* as alternating ground_truth / model_answer rows for DeBERTa.

    Each pair of rows shares the same question; the first row is the authentic
    ground_truth (label 0) and the second is the model answer (label 1).
    The header is ``text\\tlabel``.

    *output_path*'s parent directory is created if it does not exist.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("text\tlabel\n")
        for item in items:
            gt = item.get("ground_truth", "").replace("\t", " ").replace("\n", " ")
            ans = item.get("model_answer", "").replace("\t", " ").replace("\n", " ")
            f.write(f"{gt}\t0\n")
            f.write(f"{ans}\t1\n")


def write_manual_jsonl(items: list, output_path: Path) -> None:
    """Write *items* that require human review to a JSONL file.

    *output_path*'s parent directory is created if it does not exist.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
