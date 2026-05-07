"""
compare_gpt.py — adds GPT-4.1 results to experiment2_results_combined.csv
and prints a side-by-side comparison table.

GPT-4.1 JSON files only have accuracy and skill (no Brier, since model_probs=null).
"""

import json, csv, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GPT_FILES = {
    "A": os.path.join(SCRIPT_DIR, "GPT_output/exp2chatgpt/results_A_chatgpt/eval_results_mcq_gpt-4.1_20260411_144735.json"),
    "B": os.path.join(SCRIPT_DIR, "GPT_output/exp2chatgpt/results_B_chatgpt/eval_results_mcq_gpt-4.1_20260411_145814.json"),
    "C": os.path.join(SCRIPT_DIR, "GPT_output/exp2chatgpt/results_C_chatgpt/eval_results_mcq_gpt-4.1_20260411_151427.json"),
}

# Map CSV (breakdown, group) → GPT confidence_intervals key
def ci_key(breakdown, group):
    if breakdown == "overall":
        return "overall_benchmark"
    return f"{breakdown}:{group}"


def load_gpt_ci(path):
    with open(path) as f:
        data = json.load(f)
    # ci[key] = {"accuracy": [lo, mid, hi], "skill_score": [lo, mid, hi]}
    return data["confidence_intervals"]


def main():
    # Load GPT confidence intervals for A, B, C
    gpt_ci = {cond: load_gpt_ci(path) for cond, path in GPT_FILES.items()}

    # Load existing combined CSV
    combined_path = os.path.join(SCRIPT_DIR, "experiment2_results_combined.csv")
    with open(combined_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_header = reader.fieldnames
        rows = list(reader)

    # Build output header: existing cols + GPT-4.1 accuracy/skill per condition
    new_cols = []
    for cond in ("A", "B", "C"):
        new_cols.append(f"GPT-4.1_{cond}_accuracy")
        new_cols.append(f"GPT-4.1_{cond}_skill")
    out_header = existing_header + new_cols

    # Augment rows with GPT-4.1 values
    for row in rows:
        key = ci_key(row["breakdown"], row["group"])
        for cond in ("A", "B", "C"):
            ci = gpt_ci[cond]
            if key in ci:
                row[f"GPT-4.1_{cond}_accuracy"] = f"{ci[key]['accuracy'][1]:.6f}"
                row[f"GPT-4.1_{cond}_skill"]    = f"{ci[key]['skill_score'][1]:.6f}"
            else:
                row[f"GPT-4.1_{cond}_accuracy"] = ""
                row[f"GPT-4.1_{cond}_skill"]    = ""

    # Write updated CSV
    out_path = os.path.join(SCRIPT_DIR, "experiment2_results_with_gpt.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {out_path}\n")

    # ── Print comparison table ────────────────────────────────────────────────
    MODELS = ["Qwen2.5-7B", "Qwen2.5-14B", "Mistral-7B", "GPT-4.1"]
    CONDS  = [("A", "FullMeta"), ("B", "NoMeta"), ("C", "ShuffledMeta")]

    def acc(row, model, cond):
        col = f"{model}_{cond}_accuracy"
        v = row.get(col, "")
        return float(v) if v else None

    def skill(row, model, cond):
        col = f"{model}_{cond}_skill"
        v = row.get(col, "")
        return float(v) if v else None

    # Helper to format a value, highlight best model in each row
    def fmt(v):
        return f"{v:.3f}" if v is not None else "  — "

    # ── Overall accuracy table ────────────────────────────────────────────────
    print("=" * 70)
    print("ACCURACY — Condition A (Full Metadata) vs B (No Metadata) vs C (Shuffled)")
    print("=" * 70)
    col_w = 12
    header_line = f"{'Breakdown':<28} {'Group':<22}"
    for m in MODELS:
        header_line += f"  {m[:10]:^10}"
    print(header_line)
    print("-" * 70)

    for cond, cond_label in CONDS:
        print(f"\n  [{cond_label}]")
        for row in rows:
            bd, grp = row["breakdown"], row["group"]
            if bd not in ("overall", "frame_type", "reasoning_type"):
                continue
            line = f"  {bd:<26} {grp:<22}"
            vals = [acc(row, m, cond) for m in MODELS]
            best = max((v for v in vals if v is not None), default=None)
            for v in vals:
                cell = fmt(v)
                if v is not None and best is not None and abs(v - best) < 1e-9:
                    cell = f"[{cell}]"  # mark best
                line += f"  {cell:^10}"
            print(line)

    # ── Delta table: A vs B (full metadata benefit) ───────────────────────────
    print("\n")
    print("=" * 70)
    print("DELTA: A - B (positive = full metadata helps)")
    print("=" * 70)
    header_line = f"{'Breakdown':<28} {'Group':<22}"
    for m in MODELS:
        header_line += f"  {m[:10]:^10}"
    print(header_line)
    print("-" * 70)

    for row in rows:
        bd, grp = row["breakdown"], row["group"]
        if bd not in ("overall", "frame_type", "reasoning_type"):
            continue
        line = f"  {bd:<26} {grp:<22}"
        for m in MODELS:
            a = acc(row, m, "A")
            b = acc(row, m, "B")
            if a is not None and b is not None:
                delta = a - b
                cell = f"{delta:+.3f}"
            else:
                cell = "  —  "
            line += f"  {cell:^10}"
        print(line)

    # ── Delta table: A vs C (shuffled metadata impact) ───────────────────────
    print("\n")
    print("=" * 70)
    print("DELTA: A - C (positive = correct metadata > shuffled)")
    print("=" * 70)
    header_line = f"{'Breakdown':<28} {'Group':<22}"
    for m in MODELS:
        header_line += f"  {m[:10]:^10}"
    print(header_line)
    print("-" * 70)

    for row in rows:
        bd, grp = row["breakdown"], row["group"]
        if bd not in ("overall", "frame_type", "reasoning_type"):
            continue
        line = f"  {bd:<26} {grp:<22}"
        for m in MODELS:
            a = acc(row, m, "A")
            c = acc(row, m, "C")
            if a is not None and c is not None:
                delta = a - c
                cell = f"{delta:+.3f}"
            else:
                cell = "  —  "
            line += f"  {cell:^10}"
        print(line)

    # ── Skill score summary ───────────────────────────────────────────────────
    print("\n")
    print("=" * 70)
    print("SKILL SCORE — Overall benchmark only")
    print("=" * 70)
    overall_row = next(r for r in rows if r["breakdown"] == "overall")
    print(f"{'Model':<15} {'Cond A':>10} {'Cond B':>10} {'Cond C':>10}  {'A-B':>8}  {'A-C':>8}")
    print("-" * 70)
    for m in MODELS:
        sa = skill(overall_row, m, "A")
        sb = skill(overall_row, m, "B")
        sc = skill(overall_row, m, "C")
        def fs(v): return f"{v:.4f}" if v is not None else "  —  "
        def fd(a, b): return f"{a-b:+.4f}" if a is not None and b is not None else "  —  "
        print(f"{m:<15} {fs(sa):>10} {fs(sb):>10} {fs(sc):>10}  {fd(sa,sb):>8}  {fd(sa,sc):>8}")


if __name__ == "__main__":
    main()
