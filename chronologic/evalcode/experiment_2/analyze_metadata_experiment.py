"""
analyze_metadata_experiment.py

Experiment 2: Metadata Ablation Analysis
Compares three conditions:
  A — Full metadata        (original benchmark)
  B — No metadata          (neutral placeholder)
  C — Shuffled metadata    (frame permuted within frame_type group)

Usage:
    python analyze_metadata_experiment.py \
        --results-A PATH_TO_FULL_EVAL_JSON \
        --results-B PATH_TO_NO_META_JSON \
        --results-C PATH_TO_SHUFFLED_JSON \
        --benchmark  PATH_TO_BENCHMARK_JSONL \
        [--output-csv evalcode/experiment2_results.csv]

The benchmark JSONL is used to attach frame_type and reasoning_type
labels to each per-question result (the eval JSON only stores probs).
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Analyze Experiment 2: Metadata Ablation")
    p.add_argument("--results-A", required=True, help="Condition A JSON (full metadata)")
    p.add_argument("--results-B", required=True, help="Condition B JSON (no metadata)")
    p.add_argument("--results-C", required=True, help="Condition C JSON (shuffled metadata)")
    p.add_argument(
        "--benchmark",
        default=os.path.join(SCRIPT_DIR, "..", "..", "booksample", "chronologic_en_0.1.jsonl"),
        help="Benchmark JSONL used for all three conditions (provides frame_type / reasoning_type)"
    )
    p.add_argument(
        "--output-csv",
        default=os.path.join(SCRIPT_DIR, "experiment2_results.csv"),
        help="Where to write the CSV summary"
    )
    return p.parse_args()


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_benchmark(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_results(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Metrics ────────────────────────────────────────────────────────────────────

def brier_score(answer_probs, model_probs):
    """
    Brier score for a single question:
      BS = sum_i (p_i - a_i)^2
    where a_i is the ground-truth probability for answer i.
    """
    return sum((mp - ap) ** 2 for mp, ap in zip(model_probs, answer_probs))


def accuracy(correct_vector):
    if not correct_vector:
        return float("nan")
    return sum(correct_vector) / len(correct_vector)


def skill_score(brier_scores, answer_prob_lists):
    """
    Brier Skill Score = 1 - BS_model / BS_climatology
    BS_climatology = average Brier score of predicting the empirical
    base-rate for each question (uniform over answers).
    """
    if not brier_scores:
        return float("nan")
    bs_model = sum(brier_scores) / len(brier_scores)
    bs_clim_list = []
    for ap in answer_prob_lists:
        n = len(ap)
        uniform = [1.0 / n] * n
        bs_clim_list.append(brier_score(ap, uniform))
    bs_clim = sum(bs_clim_list) / len(bs_clim_list)
    if bs_clim == 0:
        return float("nan")
    return 1.0 - bs_model / bs_clim


def compute_metrics(per_question, questions):
    """
    Given per_question_results (list) and questions (list, same order),
    returns a dict with overall and broken-down metrics.
    """
    briers = []
    corrects = []
    answer_prob_lists = []

    by_frame_type = defaultdict(lambda: {"briers": [], "corrects": [], "ap_lists": []})
    by_reasoning_type = defaultdict(lambda: {"briers": [], "corrects": [], "ap_lists": []})

    for res, q in zip(per_question, questions):
        model_probs = res["model_probs"]
        correct = res["correct"]
        ap = q["answer_probabilities"]

        # Normalise model_probs to sum=1 (guard against all-zero edge cases)
        total = sum(model_probs)
        if total > 0:
            model_probs = [x / total for x in model_probs]
        else:
            n = len(model_probs)
            model_probs = [1.0 / n] * n

        # Align lengths (safety)
        n = min(len(model_probs), len(ap))
        model_probs = model_probs[:n]
        ap = ap[:n]

        bs = brier_score(ap, model_probs)
        briers.append(bs)
        corrects.append(correct)
        answer_prob_lists.append(ap)

        ft = q.get("frame_type", "unknown")
        rt = q.get("reasoning_type", "unknown")
        by_frame_type[ft]["briers"].append(bs)
        by_frame_type[ft]["corrects"].append(correct)
        by_frame_type[ft]["ap_lists"].append(ap)
        by_reasoning_type[rt]["briers"].append(bs)
        by_reasoning_type[rt]["corrects"].append(correct)
        by_reasoning_type[rt]["ap_lists"].append(ap)

    def agg(brier_list, correct_list, ap_list):
        return {
            "n": len(brier_list),
            "brier": sum(brier_list) / len(brier_list) if brier_list else float("nan"),
            "accuracy": accuracy(correct_list),
            "skill": skill_score(brier_list, ap_list),
        }

    return {
        "overall": agg(briers, corrects, answer_prob_lists),
        "by_frame_type": {
            ft: agg(v["briers"], v["corrects"], v["ap_lists"])
            for ft, v in sorted(by_frame_type.items())
        },
        "by_reasoning_type": {
            rt: agg(v["briers"], v["corrects"], v["ap_lists"])
            for rt, v in sorted(by_reasoning_type.items())
        },
    }


# ── Printing ───────────────────────────────────────────────────────────────────

def fmt(val, decimals=4):
    if isinstance(val, float) and math.isnan(val):
        return "  n/a  "
    return f"{val:.{decimals}f}"


def print_table(title, breakdown_key, metrics_by_condition, condition_labels):
    print(f"\n{'═'*72}")
    print(f"  {title}")
    print(f"{'═'*72}")

    # Collect all group keys
    keys = sorted(set(
        k
        for m in metrics_by_condition
        for k in m[breakdown_key].keys()
    ))

    # Header
    hdr = f"{'Group':<30}" + "".join(f"  {lbl:>18}" for lbl in condition_labels)
    print(hdr)
    print("  " + "Brier / Acc / Skill" * len(condition_labels))
    print("-" * len(hdr))

    for k in keys:
        row = f"{k:<30}"
        for m in metrics_by_condition:
            d = m[breakdown_key].get(k, {})
            b = fmt(d.get("brier", float("nan")))
            a = fmt(d.get("accuracy", float("nan")))
            s = fmt(d.get("skill", float("nan")))
            row += f"  {b} / {a} / {s}"
        print(row)

    # Overall row
    print("-" * len(hdr))
    row = f"{'OVERALL':<30}"
    for m in metrics_by_condition:
        d = m["overall"]
        b = fmt(d.get("brier", float("nan")))
        a = fmt(d.get("accuracy", float("nan")))
        s = fmt(d.get("skill", float("nan")))
        row += f"  {b} / {a} / {s}"
    print(row)


# ── CSV output ─────────────────────────────────────────────────────────────────

def write_csv(path, metrics_A, metrics_B, metrics_C, condition_labels):
    rows = []
    header = [
        "breakdown", "group",
        f"{condition_labels[0]}_brier", f"{condition_labels[0]}_accuracy", f"{condition_labels[0]}_skill",
        f"{condition_labels[1]}_brier", f"{condition_labels[1]}_accuracy", f"{condition_labels[1]}_skill",
        f"{condition_labels[2]}_brier", f"{condition_labels[2]}_accuracy", f"{condition_labels[2]}_skill",
        "n",
    ]

    def _rows(breakdown_key, label):
        keys = sorted(set(
            list(metrics_A[breakdown_key].keys()) +
            list(metrics_B[breakdown_key].keys()) +
            list(metrics_C[breakdown_key].keys())
        ))
        for k in keys:
            dA = metrics_A[breakdown_key].get(k, {})
            dB = metrics_B[breakdown_key].get(k, {})
            dC = metrics_C[breakdown_key].get(k, {})
            rows.append([
                label, k,
                dA.get("brier", ""), dA.get("accuracy", ""), dA.get("skill", ""),
                dB.get("brier", ""), dB.get("accuracy", ""), dB.get("skill", ""),
                dC.get("brier", ""), dC.get("accuracy", ""), dC.get("skill", ""),
                dA.get("n", ""),
            ])

    # overall
    dA, dB, dC = metrics_A["overall"], metrics_B["overall"], metrics_C["overall"]
    rows.append([
        "overall", "all",
        dA.get("brier",""), dA.get("accuracy",""), dA.get("skill",""),
        dB.get("brier",""), dB.get("accuracy",""), dB.get("skill",""),
        dC.get("brier",""), dC.get("accuracy",""), dC.get("skill",""),
        dA.get("n",""),
    ])

    _rows("by_frame_type", "frame_type")
    _rows("by_reasoning_type", "reasoning_type")

    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"\nCSV written to: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("Loading benchmark questions...")
    questions = load_benchmark(args.benchmark)
    print(f"  {len(questions)} questions loaded from {os.path.basename(args.benchmark)}")

    print("\nLoading eval results...")
    res_A = load_results(args.results_A)
    res_B = load_results(args.results_B)
    res_C = load_results(args.results_C)

    label_A = f"A-Full ({res_A['metadata'].get('model_id','?').split('/')[-1]})"
    label_B = "B-NoMeta"
    label_C = "C-Shuffled"

    pq_A = res_A["per_question_results"]
    pq_B = res_B["per_question_results"]
    pq_C = res_C["per_question_results"]

    if not (len(pq_A) == len(pq_B) == len(pq_C) == len(questions)):
        print(f"WARNING: result lengths differ: A={len(pq_A)} B={len(pq_B)} C={len(pq_C)} benchmark={len(questions)}")

    print("\nComputing metrics...")
    mA = compute_metrics(pq_A, questions)
    mB = compute_metrics(pq_B, questions)
    mC = compute_metrics(pq_C, questions)

    condition_labels = [label_A, label_B, label_C]

    # ── Overall summary ────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("  OVERALL SUMMARY  (Brier score lower = better; Accuracy/Skill higher = better)")
    print(f"{'═'*72}")
    print(f"{'Condition':<30} {'Brier':>8} {'Accuracy':>10} {'SkillScore':>12} {'N':>6}")
    print("-" * 70)
    for lbl, m in zip(condition_labels, [mA, mB, mC]):
        d = m["overall"]
        print(f"{lbl:<30} {fmt(d['brier']):>8} {fmt(d['accuracy']):>10} {fmt(d['skill']):>12} {d['n']:>6}")

    # ── Breakdown tables ───────────────────────────────────────────────────────
    print_table(
        "BY FRAME TYPE  (cols: Brier / Accuracy / SkillScore)",
        "by_frame_type",
        [mA, mB, mC],
        condition_labels,
    )

    print_table(
        "BY REASONING TYPE  (cols: Brier / Accuracy / SkillScore)",
        "by_reasoning_type",
        [mA, mB, mC],
        condition_labels,
    )

    # ── CSV ────────────────────────────────────────────────────────────────────
    write_csv(args.output_csv, mA, mB, mC, condition_labels)


if __name__ == "__main__":
    main()
