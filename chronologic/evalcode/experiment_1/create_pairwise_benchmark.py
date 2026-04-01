"""
create_pairwise_benchmark.py
----------------------------
Reads all_benchmark_questions.jsonl and expands each question into one
binary-choice (pairwise) question per distractor.

Output: experiment_1/benchmark_pairwise.jsonl
"""

import json
import sys
from collections import Counter
from pathlib import Path

BOOKSAMPLE = Path(__file__).resolve().parent.parent.parent / "booksample"
INPUT_FILE  = BOOKSAMPLE / "all_benchmark_questions.jsonl"
OUTPUT_FILE = Path(__file__).resolve().parent / "benchmark_pairwise.jsonl"


def main():
    original_questions = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                original_questions.append(json.loads(line))

    pairwise_questions = []
    distractor_counts  = Counter()
    skipped_anomalies  = 0

    for q in original_questions:
        answer_types   = q["answer_types"]
        answer_strings = q["answer_strings"]
        answer_probs   = q.get("answer_probabilities", [1.0] + [0.0] * (len(answer_types) - 1))

        # Sanity-check: index 0 must be ground truth
        if answer_types[0] != "ground_truth":
            print(f"WARNING: question '{q.get('source_title','?')}' has non-ground_truth at index 0 "
                  f"({answer_types[0]}). Skipping.", file=sys.stderr)
            skipped_anomalies += 1
            continue

        ground_truth_text = answer_strings[0]

        for i in range(1, len(answer_types)):
            dtype = answer_types[i]

            # Skip answers that are labelled ground_truth at non-zero positions (data anomaly)
            if dtype == "ground_truth":
                print(f"WARNING: answer_types[{i}] == 'ground_truth' in "
                      f"'{q.get('source_title','?')}'. Skipping this distractor.", file=sys.stderr)
                skipped_anomalies += 1
                continue

            pairwise = dict(q)                           # shallow copy of all original fields
            pairwise["distractor_type_tested"] = dtype
            pairwise["answer_types"]           = ["ground_truth", dtype]
            pairwise["answer_strings"]         = [ground_truth_text, answer_strings[i]]
            pairwise["answer_probabilities"]   = [1.0, 0.0]

            pairwise_questions.append(pairwise)
            distractor_counts[dtype] += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pq in pairwise_questions:
            f.write(json.dumps(pq, ensure_ascii=False) + "\n")

    # --- Statistics ---
    print(f"\n=== Pairwise Benchmark Creation Complete ===")
    print(f"Input  : {INPUT_FILE}")
    print(f"Output : {OUTPUT_FILE}")
    print(f"\nTotal original questions  : {len(original_questions)}")
    print(f"Total pairwise questions  : {len(pairwise_questions)}")
    print(f"Skipped anomalies         : {skipped_anomalies}")
    print(f"\nPairwise questions per distractor type:")
    for dtype, count in distractor_counts.most_common():
        print(f"  {dtype:<55} {count:>4}")


if __name__ == "__main__":
    main()
