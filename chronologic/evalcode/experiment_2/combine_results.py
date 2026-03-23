"""
combine_results.py — merges per-model experiment2 CSVs into one wide CSV.

Output columns:
  breakdown, group, n,
  <model>_A_brier, <model>_A_accuracy, <model>_A_skill,
  <model>_B_brier, <model>_B_accuracy, <model>_B_skill,
  <model>_C_brier, <model>_C_accuracy, <model>_C_skill,
  ... (repeated for each model)
"""

import csv, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    ("Qwen2.5-7B",  "experiment2_results.csv"),
    ("Qwen2.5-14B", "experiment2_results_qwen14b.csv"),
    ("Mistral-7B",  "experiment2_results_mistral.csv"),
]

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    # Load all model results, keyed by (breakdown, group)
    all_data = {}   # (breakdown, group) -> {model: {A/B/C: {brier, accuracy, skill}}}
    row_order = []  # preserve row order from first file
    n_by_key = {}

    for model_name, fname in MODELS:
        path = os.path.join(SCRIPT_DIR, fname)
        rows = load_csv(path)
        for row in rows:
            key = (row["breakdown"], row["group"])
            if key not in all_data:
                all_data[key] = {}
                row_order.append(key)
                n_by_key[key] = row["n"]
            # Find the A-Full column prefix (varies by model name in header)
            a_brier_col = next(c for c in row if c.endswith("_brier") and "Full" in c)
            prefix = a_brier_col[: a_brier_col.rfind("_brier")]
            all_data[key][model_name] = {
                "A_brier":    row[f"{prefix}_brier"],
                "A_accuracy": row[f"{prefix}_accuracy"],
                "A_skill":    row[f"{prefix}_skill"],
                "B_brier":    row["B-NoMeta_brier"],
                "B_accuracy": row["B-NoMeta_accuracy"],
                "B_skill":    row["B-NoMeta_skill"],
                "C_brier":    row["C-Shuffled_brier"],
                "C_accuracy": row["C-Shuffled_accuracy"],
                "C_skill":    row["C-Shuffled_skill"],
            }

    # Build output header
    model_names = [m for m, _ in MODELS]
    header = ["breakdown", "group", "n"]
    for m in model_names:
        for cond in ("A", "B", "C"):
            for metric in ("brier", "accuracy", "skill"):
                header.append(f"{m}_{cond}_{metric}")

    # Write combined CSV
    out_path = os.path.join(SCRIPT_DIR, "experiment2_results_combined.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for key in row_order:
            breakdown, group = key
            row_out = [breakdown, group, n_by_key[key]]
            for m in model_names:
                vals = all_data[key].get(m, {})
                for cond in ("A", "B", "C"):
                    for metric in ("brier", "accuracy", "skill"):
                        row_out.append(vals.get(f"{cond}_{metric}", ""))
            writer.writerow(row_out)

    print(f"Combined CSV written to: {out_path}")
    print(f"  Models: {', '.join(model_names)}")
    print(f"  Rows:   {len(row_order)}")

if __name__ == "__main__":
    main()
