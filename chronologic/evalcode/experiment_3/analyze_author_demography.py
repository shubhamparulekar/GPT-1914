"""
Experiment 3: Author Demography x Model Performance
====================================================
Disaggregates ChronoLogic eval results by author metadata:
  - author_nationality
  - author_profession
  - source_genre
  - source_date (binned 1875-1899 vs 1900-1924)
  - reasoning_type

Input files:
  - ../../booksample/chronologic_en_0.1.jsonl   (709 questions with metadata)
  - ../../booksample/eval_results_full_Qwen_Qwen2.5-7B-Instruct_20260316_174038.json
  - ../../booksample/eval_results_full_Qwen_Qwen2.5-32B-Instruct_20260322_101723.json

Output files:
  - experiment3_results.csv
  - experiment3_by_reasoning_type.csv
  - experiment3_plots/ (PNG bar charts if matplotlib available)
"""

import json
import csv
import math
import os
import sys
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BOOKSAMPLE = os.path.join(SCRIPT_DIR, "..", "..", "booksample")

QUESTIONS_FILE = os.path.join(BOOKSAMPLE, "chronologic_en_0.1.jsonl")

EVAL_FILES = {
    "Qwen-7B": os.path.join(BOOKSAMPLE,
        "eval_results_full_Qwen_Qwen2.5-7B-Instruct_20260316_174038.json"),
    "Qwen-32B": os.path.join(BOOKSAMPLE,
        "eval_results_full_Qwen_Qwen2.5-32B-Instruct_20260322_101723.json"),
}

OUTPUT_CSV         = os.path.join(SCRIPT_DIR, "experiment3_results.csv")
OUTPUT_RTYPE_CSV   = os.path.join(SCRIPT_DIR, "experiment3_by_reasoning_type.csv")
PLOTS_DIR          = os.path.join(SCRIPT_DIR, "experiment3_plots")


# ── Nationality normalisation ──────────────────────────────────────────────────
# Merge variant spellings / geographic sub-groups into canonical labels.
NATIONALITY_MAP = {
    "American":     "American",
    "United States":"American",
    "British":      "British Isles",
    "English":      "British Isles",
    "Scottish":     "British Isles",
    "Irish":        "British Isles",
    "Welsh":        "British Isles",
    "Canadian":     "Canadian",
    "Australian":   "Australian",
    "South African":"South African",
    "German":       "German",
    "French":       "French",
    "Italian":      "Italian",
    "Russian":      "Russian",
}

def normalise_nationality(raw: str) -> str:
    if not raw or raw.strip() in ("", "unknown"):
        return "Unknown"
    return NATIONALITY_MAP.get(raw.strip(), raw.strip())


def normalise_profession(raw: str) -> str:
    if not raw or raw.strip() == "":
        return "Unknown"
    return raw.strip()


# ── Date binning ───────────────────────────────────────────────────────────────
def date_bin(year) -> str:
    if year is None:
        return "Unknown"
    try:
        y = int(year)
    except (ValueError, TypeError):
        return "Unknown"
    if y <= 1899:
        return "1875-1899"
    elif y <= 1924:
        return "1900-1924"
    else:
        return "Other"


# ── Brier score ────────────────────────────────────────────────────────────────
def brier_score(model_probs, answer_probs) -> float:
    """
    Brier score = mean squared error between model probability distribution
    and the ground-truth probability distribution.
    Lower is better (0 = perfect).
    """
    n = min(len(model_probs), len(answer_probs))
    return sum((model_probs[i] - answer_probs[i]) ** 2 for i in range(n)) / n


# ── Stats helpers ──────────────────────────────────────────────────────────────
def mean(values):
    return sum(values) / len(values) if values else float("nan")


def std(values):
    if len(values) < 2:
        return float("nan")
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


# ── Load data ─────────────────────────────────────────────────────────────────
def load_questions(path):
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    print(f"Loaded {len(questions)} questions from {os.path.basename(path)}")
    return questions


def load_eval_results(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    results = data["per_question_results"]
    print(f"Loaded {len(results)} results from {os.path.basename(path)} "
          f"(model: {data['metadata']['model_id']})")
    return data["metadata"], results


# ── Merge questions + eval results → per-question records ─────────────────────
def merge(questions, per_question_results, model_name):
    """Return list of dicts with metadata + scores for each question."""
    assert len(questions) == len(per_question_results), (
        f"Length mismatch: {len(questions)} questions vs "
        f"{len(per_question_results)} results"
    )
    records = []
    for i, (q, r) in enumerate(zip(questions, per_question_results)):
        model_probs = r.get("model_probs")
        correct     = r.get("correct")
        answer_probs = q.get("answer_probabilities", [])

        bs = None
        if model_probs and answer_probs:
            bs = brier_score(model_probs, answer_probs)

        records.append({
            "idx":                i,
            "model":              model_name,
            "source_title":       q.get("source_title", ""),
            "source_author":      q.get("source_author", ""),
            "source_date":        q.get("source_date"),
            "author_nationality": normalise_nationality(q.get("author_nationality", "")),
            "author_profession":  normalise_profession(q.get("author_profession", "")),
            "source_genre":       q.get("source_genre", "Unknown") or "Unknown",
            "reasoning_type":     q.get("reasoning_type", "Unknown") or "Unknown",
            "question_category":  q.get("question_category", "Unknown") or "Unknown",
            "date_bin":           date_bin(q.get("source_date")),
            "brier_score":        bs,
            "accuracy":           int(correct) if correct is not None else None,
        })
    return records


# ── Aggregation ───────────────────────────────────────────────────────────────
def aggregate(records, group_key, models):
    """
    Group records by group_key and model, returning a list of summary dicts.
    models: list of model names expected
    """
    # group_key value → model_name → lists of scores
    groups = defaultdict(lambda: defaultdict(lambda: {"brier": [], "acc": []}))

    for rec in records:
        gval  = rec[group_key]
        mname = rec["model"]
        bs    = rec["brier_score"]
        acc   = rec["accuracy"]
        if bs is not None:
            groups[gval][mname]["brier"].append(bs)
        if acc is not None:
            groups[gval][mname]["acc"].append(acc)

    # Build rows
    rows = []
    for gval in sorted(groups.keys()):
        row = {group_key: gval}
        total_q = 0
        for mname in models:
            b_vals = groups[gval][mname]["brier"]
            a_vals = groups[gval][mname]["acc"]
            n = max(len(b_vals), len(a_vals))
            total_q = max(total_q, n)
            col = mname.replace("-", "_").replace(" ", "_")
            row[f"{col}_mean_brier"]    = round(mean(b_vals), 4) if b_vals else ""
            row[f"{col}_std_brier"]     = round(std(b_vals),  4) if b_vals else ""
            row[f"{col}_mean_accuracy"] = round(mean(a_vals), 4) if a_vals else ""
            row[f"{col}_std_accuracy"]  = round(std(a_vals),  4) if a_vals else ""
            row[f"{col}_n"]             = n
        row["n_questions"] = total_q
        rows.append(row)

    return rows


def aggregate_reasoning_cross(records, group_key, models):
    """For each (group_key value, reasoning_type) pair, compute stats."""
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: {"brier": [], "acc": []}
    )))
    for rec in records:
        gval   = rec[group_key]
        rtype  = rec["reasoning_type"]
        mname  = rec["model"]
        bs     = rec["brier_score"]
        acc    = rec["accuracy"]
        if bs is not None:
            groups[gval][rtype][mname]["brier"].append(bs)
        if acc is not None:
            groups[gval][rtype][mname]["acc"].append(acc)

    rows = []
    for gval in sorted(groups.keys()):
        for rtype in sorted(groups[gval].keys()):
            row = {group_key: gval, "reasoning_type": rtype}
            for mname in models:
                b_vals = groups[gval][rtype][mname]["brier"]
                a_vals = groups[gval][rtype][mname]["acc"]
                col = mname.replace("-", "_").replace(" ", "_")
                row[f"{col}_mean_brier"]    = round(mean(b_vals), 4) if b_vals else ""
                row[f"{col}_mean_accuracy"] = round(mean(a_vals), 4) if a_vals else ""
                row[f"{col}_n"]             = max(len(b_vals), len(a_vals))
            rows.append(row)
    return rows


# ── Pretty-print table ────────────────────────────────────────────────────────
def print_table(title, rows, group_key, models):
    if not rows:
        print(f"\n{title}: (no data)\n")
        return

    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    # Build column specs
    col_specs = [(group_key, 22)]
    for mname in models:
        col = mname.replace("-", "_").replace(" ", "_")
        col_specs += [
            (f"{col}_mean_brier",    11),
            (f"{col}_mean_accuracy", 12),
        ]
    col_specs.append(("n_questions", 10))

    # Header
    header_parts = []
    for cname, width in col_specs:
        label = cname.replace("_mean_", " ").replace("_", " ")
        header_parts.append(label[:width].ljust(width))
    print("  " + "  ".join(header_parts))

    subheader_parts = []
    for cname, width in col_specs:
        if "brier" in cname:
            sub = "(↓ better)"
        elif "accuracy" in cname:
            sub = "(↑ better)"
        else:
            sub = ""
        subheader_parts.append(sub[:width].ljust(width))
    print("  " + "  ".join(subheader_parts))
    print("  " + "-" * (sum(w for _, w in col_specs) + 2 * len(col_specs)))

    # Rows
    for row in rows:
        parts = []
        for cname, width in col_specs:
            val = row.get(cname, "")
            if isinstance(val, float):
                if "accuracy" in cname:
                    val = f"{val*100:.1f}%"
                else:
                    val = f"{val:.4f}"
            parts.append(str(val)[:width].ljust(width))
        print("  " + "  ".join(parts))

    print()


# ── Write CSV ─────────────────────────────────────────────────────────────────
def write_csv(path, rows):
    if not rows:
        return
    # Collect all field names across all rows to handle variable schemas
    all_fields = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                all_fields.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore",
                                restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {os.path.relpath(path)}")


# ── Visualisations ────────────────────────────────────────────────────────────
def make_plots(all_agg, models):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\nmatplotlib not available — skipping plots.")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for attr, rows in all_agg.items():
        if not rows:
            continue

        labels = [r[attr] for r in rows]
        x = np.arange(len(labels))
        width = 0.35 / max(len(models), 1)

        for metric_key, metric_label, better in [
            ("mean_accuracy", "Accuracy", "higher"),
            ("mean_brier",    "Brier Score", "lower"),
        ]:
            fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))

            for mi, mname in enumerate(models):
                col = mname.replace("-", "_").replace(" ", "_")
                vals = []
                for row in rows:
                    v = row.get(f"{col}_{metric_key}", "")
                    vals.append(float(v) if v != "" else 0.0)

                offset = (mi - (len(models) - 1) / 2) * width
                bars = ax.bar(x + offset, vals, width * 0.9,
                              label=mname, color=palette[mi % len(palette)],
                              alpha=0.85)

                # Value labels on bars
                for bar, v in zip(bars, vals):
                    if v > 0:
                        label_text = (f"{v*100:.1f}%" if metric_key == "mean_accuracy"
                                      else f"{v:.3f}")
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.005,
                                label_text,
                                ha="center", va="bottom", fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} by {attr.replace('_', ' ').title()} "
                         f"({better} is better)")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()

            fname = f"{attr}_{metric_key}.png"
            fpath = os.path.join(PLOTS_DIR, fname)
            fig.savefig(fpath, dpi=130)
            plt.close(fig)
            print(f"  Plot saved: {os.path.relpath(fpath)}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\nExperiment 3: Author Demography × Model Performance")
    print("=" * 60)

    # Load questions
    questions = load_questions(QUESTIONS_FILE)

    # Load eval results and merge
    all_records = []
    loaded_models = []

    for model_name, eval_path in EVAL_FILES.items():
        if not os.path.exists(eval_path):
            print(f"WARNING: {eval_path} not found — skipping {model_name}")
            continue
        meta, results = load_eval_results(eval_path)
        records = merge(questions, results, model_name)
        all_records.extend(records)
        loaded_models.append(model_name)

    if not all_records:
        print("ERROR: No eval results loaded. Check file paths.")
        sys.exit(1)

    print(f"\nModels loaded: {loaded_models}")
    print(f"Total question-result pairs: {len(all_records)}")

    # ── Aggregation dimensions ─────────────────────────────────────────────────
    dimensions = {
        "author_nationality": "Author Nationality",
        "author_profession":  "Author Profession (top professions)",
        "source_genre":       "Source Genre",
        "date_bin":           "Publication Date Bin",
    }

    all_agg = {}
    csv_rows_all = []

    for attr, title in dimensions.items():
        rows = aggregate(all_records, attr, loaded_models)

        # For profession/genre, limit to top groups by question count
        if attr in ("author_profession", "source_genre"):
            rows = sorted(rows, key=lambda r: r["n_questions"], reverse=True)[:15]

        all_agg[attr] = rows
        print_table(title, rows, attr, loaded_models)

        # Tag rows with the dimension name
        for r in rows:
            tagged = {"dimension": attr, "group_value": r[attr]}
            tagged.update(r)
            csv_rows_all.append(tagged)

    # ── Reasoning-type cross-breakdown ────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  Reasoning Type Breakdown (all questions)")
    print(f"{'='*80}")
    rt_rows = aggregate(all_records, "reasoning_type", loaded_models)
    print_table("Reasoning Type", rt_rows, "reasoning_type", loaded_models)

    # Cross: nationality × reasoning type
    nat_rt_rows = aggregate_reasoning_cross(all_records, "author_nationality", loaded_models)

    # ── Write CSVs ─────────────────────────────────────────────────────────────
    print("\nWriting output files...")
    write_csv(OUTPUT_CSV, csv_rows_all)

    # Combine reasoning-type rows for secondary CSV
    all_rt_rows = []
    for attr in ["author_nationality", "source_genre", "date_bin"]:
        rows = aggregate_reasoning_cross(all_records, attr, loaded_models)
        for r in rows:
            tagged = {"dimension": attr}
            tagged.update(r)
            all_rt_rows.append(tagged)
    all_rt_rows.extend(nat_rt_rows)

    write_csv(OUTPUT_RTYPE_CSV, all_rt_rows)

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    agg_for_plots = {k: v for k, v in all_agg.items()}
    agg_for_plots["reasoning_type"] = rt_rows
    make_plots(agg_for_plots, loaded_models)

    # ── Summary comparison ─────────────────────────────────────────────────────
    if len(loaded_models) > 1:
        print(f"\n{'='*80}")
        print("  Model Comparison Summary")
        print(f"{'='*80}")
        # Overall per-model stats
        for mname in loaded_models:
            recs = [r for r in all_records if r["model"] == mname]
            bs_vals  = [r["brier_score"] for r in recs if r["brier_score"] is not None]
            acc_vals = [r["accuracy"]    for r in recs if r["accuracy"]    is not None]
            print(f"  {mname}:")
            print(f"    Overall Brier Score : {mean(bs_vals):.4f} (±{std(bs_vals):.4f})")
            print(f"    Overall Accuracy    : {mean(acc_vals)*100:.1f}%")
            print(f"    Questions evaluated : {len(recs)}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
