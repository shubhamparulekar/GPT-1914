"""
analyze_distractor_types.py
---------------------------
Loads the pairwise eval JSON output and produces:
  - Overall distractor-type ranking (accuracy / failure rate)
  - question_category × distractor_type breakdown
  - Detailed CSV: experiment_1/experiment1_results.csv

Usage:
    python analyze_distractor_types.py [results_pairwise.json] [benchmark_pairwise.jsonl]

If no paths given, defaults to files in this script's directory.
"""

import json
import csv
import sys
from collections import defaultdict
from pathlib import Path

EXPERIMENT_DIR   = Path(__file__).resolve().parent
DEFAULT_RESULTS  = EXPERIMENT_DIR / "results_pairwise.json"
DEFAULT_PAIRWISE = EXPERIMENT_DIR / "benchmark_pairwise.jsonl"
OUTPUT_CSV       = EXPERIMENT_DIR / "experiment1_results.csv"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def accuracy(correct, total):
    return correct / total if total else float("nan")


def load_results(path: Path) -> list[dict]:
    """Load the per-question result list from the eval JSON output."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # The eval script wraps per-question results in different keys depending
    # on the run mode.  Try the most common ones in order.
    for key in ("results", "questions", "per_question", "question_results", "per_question_results"):
        if key in data and isinstance(data[key], list):
            return data[key]

    # Fallback: if the top-level object is itself a list
    if isinstance(data, list):
        return data

    raise ValueError(
        f"Cannot find a list of per-question results in {path}. "
        f"Top-level keys: {list(data.keys())}"
    )


def is_correct(q: dict) -> bool:
    """
    Return True if the model chose the ground-truth answer.

    The eval script stores the selected answer index in different fields
    depending on version — try them all.
    """
    # Direct boolean 'correct' field (present in this eval version)
    if "correct" in q and q["correct"] is not None:
        return bool(q["correct"])

    chosen_idx = q.get("chosen_index", q.get("selected_index", q.get("model_choice_index")))
    if chosen_idx is not None:
        return int(chosen_idx) == 0          # index 0 is always ground truth

    # Some versions store the chosen letter ('A', 'B', ...) or the chosen text
    chosen_letter = q.get("chosen_letter", q.get("model_choice"))
    if chosen_letter is not None:
        return str(chosen_letter).strip().upper() == "A"  # 'A' == index 0

    chosen_text = q.get("chosen_answer", q.get("model_answer", q.get("predicted_answer")))
    if chosen_text is not None:
        gt = q.get("answer_strings", [""])[0]
        return str(chosen_text).strip() == str(gt).strip()

    raise KeyError(
        f"Cannot determine model choice from question result keys: {list(q.keys())}"
    )


def fmt_pct(v) -> str:
    if v != v:   # nan
        return "  N/A  "
    return f"{v * 100:6.1f}%"


def print_table(title: str, rows: list[tuple], col_headers: tuple):
    """Print a simple fixed-width table."""
    print(f"\n{'═'*72}")
    print(f"  {title}")
    print('═'*72)
    col_w = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
             for i, h in enumerate(col_headers)]
    header = "  ".join(str(h).ljust(col_w[i]) for i, h in enumerate(col_headers))
    print(header)
    print("─" * 72)
    for row in rows:
        print("  ".join(str(cell).ljust(col_w[i]) for i, cell in enumerate(row)))
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    results_path  = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS
    pairwise_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PAIRWISE

    if not results_path.exists():
        print(f"ERROR: results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)
    if not pairwise_path.exists():
        print(f"ERROR: pairwise JSONL not found: {pairwise_path}", file=sys.stderr)
        sys.exit(1)

    results = load_results(results_path)
    print(f"Loaded {len(results)} per-question results from {results_path}")

    # Load pairwise JSONL to get distractor_type_tested and question_category
    pairwise = []
    with open(pairwise_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairwise.append(json.loads(line))
    print(f"Loaded {len(pairwise)} questions from {pairwise_path}")

    if len(results) != len(pairwise):
        print(f"WARNING: result count ({len(results)}) != pairwise count ({len(pairwise)}). "
              "Truncating to shorter.", file=sys.stderr)

    # Merge: attach distractor_type_tested and question_category to each result
    questions = []
    for res, q in zip(results, pairwise):
        merged = dict(res)
        merged["distractor_type_tested"] = q.get("distractor_type_tested")
        merged["question_category"]      = q.get("question_category", "unknown")
        questions.append(merged)

    print(f"Merged {len(questions)} records.")

    # -----------------------------------------------------------------------
    # Aggregate: overall per distractor type
    # -----------------------------------------------------------------------
    dtype_correct = defaultdict(int)
    dtype_total   = defaultdict(int)

    # Aggregate: question_category × distractor_type
    cat_dtype_correct = defaultdict(lambda: defaultdict(int))
    cat_dtype_total   = defaultdict(lambda: defaultdict(int))

    parse_errors = 0
    for q in questions:
        dtype = q.get("distractor_type_tested")
        if dtype is None:
            # Older eval versions may not have propagated the field; skip
            continue

        cat = q.get("question_category", "unknown")

        try:
            correct = is_correct(q)
        except (KeyError, TypeError) as e:
            parse_errors += 1
            continue

        dtype_total[dtype]   += 1
        cat_dtype_total[cat][dtype] += 1
        if correct:
            dtype_correct[dtype]   += 1
            cat_dtype_correct[cat][dtype] += 1

    if parse_errors:
        print(f"WARNING: Could not determine model choice for {parse_errors} questions.")

    # -----------------------------------------------------------------------
    # Table 1 — overall distractor ranking
    # -----------------------------------------------------------------------
    all_dtypes = sorted(dtype_total.keys(),
                        key=lambda d: accuracy(dtype_correct[d], dtype_total[d]))

    ranking_rows = []
    for dtype in all_dtypes:
        n       = dtype_total[dtype]
        correct = dtype_correct[dtype]
        acc     = accuracy(correct, n)
        ranking_rows.append((dtype, fmt_pct(acc), n, fmt_pct(1 - acc)))

    print_table(
        "Distractor Type Ranking  (sorted by accuracy ascending = most fooling first)",
        ranking_rows,
        ("Distractor Type", "Accuracy", "N", "Failure Rate"),
    )

    # -----------------------------------------------------------------------
    # Table 2 — question_category × distractor_type (compact)
    # -----------------------------------------------------------------------
    all_cats = sorted(cat_dtype_total.keys())
    # Only show the 8 most common distractor types to keep width manageable
    top_dtypes = [d for d, _ in
                  sorted(dtype_total.items(), key=lambda x: -x[1])[:8]]

    cat_rows = []
    for cat in all_cats:
        row = [cat]
        for dtype in top_dtypes:
            n  = cat_dtype_total[cat].get(dtype, 0)
            ok = cat_dtype_correct[cat].get(dtype, 0)
            row.append(f"{fmt_pct(accuracy(ok, n))} (n={n})" if n else "—")
        cat_rows.append(tuple(row))

    print_table(
        "Accuracy by question_category × distractor_type  (top 8 distractor types)",
        cat_rows,
        ("question_category",) + tuple(d[:20] for d in top_dtypes),
    )

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Sheet 1: overall distractor ranking
        writer.writerow(["=== OVERALL DISTRACTOR TYPE RANKING ==="])
        writer.writerow(["distractor_type", "accuracy_pct", "n_questions", "failure_rate_pct"])
        for dtype in all_dtypes:
            n   = dtype_total[dtype]
            acc = accuracy(dtype_correct[dtype], n)
            writer.writerow([
                dtype,
                round(acc * 100, 2) if acc == acc else "",
                n,
                round((1 - acc) * 100, 2) if acc == acc else "",
            ])

        writer.writerow([])

        # Sheet 2: question_category × distractor_type
        writer.writerow(["=== ACCURACY BY QUESTION_CATEGORY x DISTRACTOR_TYPE ==="])
        header = ["question_category"] + all_dtypes
        writer.writerow(header)
        for cat in all_cats:
            row = [cat]
            for dtype in all_dtypes:
                n  = cat_dtype_total[cat].get(dtype, 0)
                ok = cat_dtype_correct[cat].get(dtype, 0)
                row.append(round(accuracy(ok, n) * 100, 2) if n else "")
            writer.writerow(row)

    print(f"Detailed results written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
