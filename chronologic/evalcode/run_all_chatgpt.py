"""
run_all_chatgpt.py
------------------
Orchestrator that runs all three ChronoLogic experiments end-to-end using
the OpenAI Chat Completions API. Invoke run_chatgpt_eval.py for model calls,
then the existing per-experiment analysis scripts for results.

Prerequisites:
    pip install openai numpy tqdm
    export OPENAI_API_KEY="sk-..."        # or use --api-key

Usage:
    # Run all three experiments with gpt-4o-mini:
    python run_all_chatgpt.py gpt-4o-mini

    # Select specific experiments:
    python run_all_chatgpt.py gpt-4o --experiments 1 2

    # Quick smoke-test with only 10 questions per condition:
    python run_all_chatgpt.py gpt-4o-mini --limit 10

    # Full run with verbose per-question markdown:
    python run_all_chatgpt.py gpt-4o --verbose-report

Experiment overview:
    1 — Pairwise distractor types
        Creates benchmark_pairwise.jsonl from booksample/all_benchmark_questions.jsonl
        Runs MCQ eval → analyze_distractor_types.py
        Output: experiment_1/results_chatgpt/  +  experiment_1/experiment1_results.csv

    2 — Metadata ablation (three conditions A / B / C)
        Condition A: original benchmark (chronologic_en_0.1.jsonl)
        Condition B: no metadata      (benchmark_no_metadata.jsonl)
        Condition C: shuffled metadata (benchmark_shuffled_metadata.jsonl)
        Files B and C are created automatically if missing.
        Runs MCQ eval on each → analyze_metadata_experiment.py
        Output: experiment_2/results_{A,B,C}_chatgpt/  +  experiment_2/experiment2_results_chatgpt.csv

    3 — Author demography breakdown
        Runs MCQ eval on main benchmark → analyze_author_demography.py
        Output: experiment_3/results_chatgpt/  +  experiment_3/experiment3_results.csv
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Resolve paths relative to this script so the orchestrator can be run from anywhere.
SCRIPT_DIR  = Path(__file__).resolve().parent
BOOKSAMPLE  = SCRIPT_DIR.parent / "booksample"

MAIN_BENCHMARK    = BOOKSAMPLE / "chronologic_en_0.1.jsonl"
ALL_BENCHMARK     = BOOKSAMPLE / "all_benchmark_questions.jsonl"
NO_META_FILE      = SCRIPT_DIR / "experiment_2" / "benchmark_no_metadata.jsonl"
SHUFFLED_META_FILE = SCRIPT_DIR / "experiment_2" / "benchmark_shuffled_metadata.jsonl"
PAIRWISE_FILE     = SCRIPT_DIR / "experiment_1" / "benchmark_pairwise.jsonl"

RUN_EVAL_SCRIPT   = SCRIPT_DIR / "run_chatgpt_eval.py"

# Analysis scripts (existing, unchanged)
ANALYZE_EXP1 = SCRIPT_DIR / "experiment_1" / "analyze_distractor_types.py"
ANALYZE_EXP2 = SCRIPT_DIR / "experiment_2" / "analyze_metadata_experiment.py"
ANALYZE_EXP3 = SCRIPT_DIR / "experiment_3" / "analyze_author_demography.py"

# Condition-creation scripts (existing, unchanged)
CREATE_PAIRWISE   = SCRIPT_DIR / "experiment_1" / "create_pairwise_benchmark.py"
CREATE_CONDITIONS = SCRIPT_DIR / "experiment_2" / "create_metadata_conditions.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd, description):
    """Run a subprocess command, printing the description and output."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  $ {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run([str(c) for c in cmd], check=True)
    return result


def find_latest_json(directory):
    """Return the most recently modified .json file in directory, or None."""
    jsons = sorted(
        Path(directory).glob("eval_results_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(jsons[0]) if jsons else None


def build_eval_cmd(model_id, jsonl_path, output_dir, api_key=None,
                   verbose_report=False, limit=None, extra_args=None):
    """Assemble the command list for run_chatgpt_eval.py."""
    cmd = [sys.executable, RUN_EVAL_SCRIPT, model_id, jsonl_path,
           "--output-dir", output_dir]
    if api_key:
        cmd += ["--api-key", api_key]
    if verbose_report:
        cmd.append("--verbose-report")
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if extra_args:
        cmd += extra_args
    return cmd


# ---------------------------------------------------------------------------
# Per-experiment runners
# ---------------------------------------------------------------------------

def run_experiment_1(model_id, api_key, verbose_report, limit, n_bootstrap):
    """Experiment 1: Pairwise distractor type ranking."""
    print("\n" + "#" * 60)
    print("  EXPERIMENT 1: Pairwise Distractor Types")
    print("#" * 60)

    # 1a. Create pairwise benchmark if missing
    if not PAIRWISE_FILE.exists():
        run([sys.executable, CREATE_PAIRWISE],
            "Creating pairwise benchmark from all_benchmark_questions.jsonl")
    else:
        print(f"\n  Pairwise benchmark already exists: {PAIRWISE_FILE}")

    if not PAIRWISE_FILE.exists():
        print("ERROR: pairwise benchmark not created. "
              "Check that all_benchmark_questions.jsonl exists in booksample/.")
        return None

    # 1b. Run MCQ eval
    out_dir = SCRIPT_DIR / "experiment_1" / "results_chatgpt"
    extra = ["--n-bootstrap", str(n_bootstrap)]
    run(build_eval_cmd(model_id, PAIRWISE_FILE, out_dir, api_key,
                       verbose_report, limit, extra),
        f"Running MCQ eval on pairwise benchmark ({PAIRWISE_FILE.name})")

    results_file = find_latest_json(out_dir)
    if not results_file:
        print("ERROR: No eval results found after running experiment 1.")
        return None

    # 1c. Analyze
    run([sys.executable, ANALYZE_EXP1, results_file, str(PAIRWISE_FILE)],
        "Analyzing distractor type results")

    return results_file


def run_experiment_2(model_id, api_key, verbose_report, limit, n_bootstrap):
    """Experiment 2: Metadata ablation (conditions A / B / C)."""
    print("\n" + "#" * 60)
    print("  EXPERIMENT 2: Metadata Ablation")
    print("#" * 60)

    # 2a. Create condition files B and C if missing
    if not NO_META_FILE.exists() or not SHUFFLED_META_FILE.exists():
        run([sys.executable, CREATE_CONDITIONS],
            "Creating no-metadata and shuffled-metadata condition files")
    else:
        print(f"\n  Condition files already exist:\n"
              f"    {NO_META_FILE}\n    {SHUFFLED_META_FILE}")

    for label, jsonl_file in [
        ("A (full metadata)",    MAIN_BENCHMARK),
        ("B (no metadata)",      NO_META_FILE),
        ("C (shuffled metadata)", SHUFFLED_META_FILE),
    ]:
        if not Path(jsonl_file).exists():
            print(f"ERROR: {jsonl_file} not found. Skipping condition {label}.")
            continue

        out_dir = SCRIPT_DIR / "experiment_2" / f"results_{label[0]}_chatgpt"
        extra = ["--n-bootstrap", str(n_bootstrap)]
        run(build_eval_cmd(model_id, jsonl_file, out_dir, api_key,
                           verbose_report, limit, extra),
            f"Running MCQ eval — condition {label}")

    # Find the three result files
    def latest(subdir):
        return find_latest_json(
            SCRIPT_DIR / "experiment_2" / subdir
        )

    res_A = latest("results_A_chatgpt")
    res_B = latest("results_B_chatgpt")
    res_C = latest("results_C_chatgpt")

    if not all([res_A, res_B, res_C]):
        print("WARNING: one or more condition result files missing; "
              "skipping analysis.")
        return

    out_csv = SCRIPT_DIR / "experiment_2" / "experiment2_results_chatgpt.csv"
    run([
        sys.executable, ANALYZE_EXP2,
        "--results-A", res_A,
        "--results-B", res_B,
        "--results-C", res_C,
        "--benchmark",  str(MAIN_BENCHMARK),
        "--output-csv", str(out_csv),
    ], "Analyzing metadata ablation results")


def run_experiment_3(model_id, api_key, verbose_report, limit, n_bootstrap):
    """Experiment 3: Author demography breakdown."""
    print("\n" + "#" * 60)
    print("  EXPERIMENT 3: Author Demography")
    print("#" * 60)

    if not MAIN_BENCHMARK.exists():
        print(f"ERROR: main benchmark not found at {MAIN_BENCHMARK}.")
        return

    out_dir = SCRIPT_DIR / "experiment_3" / "results_chatgpt"
    extra = ["--n-bootstrap", str(n_bootstrap)]
    run(build_eval_cmd(model_id, MAIN_BENCHMARK, out_dir, api_key,
                       verbose_report, limit, extra),
        f"Running MCQ eval on main benchmark for experiment 3")

    results_file = find_latest_json(out_dir)
    if not results_file:
        print("ERROR: No eval results found after running experiment 3.")
        return

    # analyze_author_demography.py reads its eval files from hardcoded paths.
    # We patch those paths by passing the ChatGPT result file in via a
    # temporary environment variable that the analysis script can optionally
    # read from, or we run it with updated paths via the existing EVAL_FILES
    # mechanism.  For now we print guidance; the script can also be run
    # manually with the found result file.
    print(f"\n  New ChatGPT eval results: {results_file}")
    print(f"  To include in the demography analysis, either:\n"
          f"    a) Update EVAL_FILES in experiment_3/analyze_author_demography.py "
          f"to add this model, or\n"
          f"    b) Run: python experiment_3/analyze_author_demography.py "
          f"(after adding the path to EVAL_FILES)")

    # Attempt to run with existing eval files if they exist alongside ChatGPT results
    try:
        run([sys.executable, ANALYZE_EXP3],
            "Running author demography analysis (existing model results)")
    except subprocess.CalledProcessError:
        print("  (Analysis skipped — pre-existing Qwen result files may be absent.)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run all ChronoLogic experiments using the OpenAI Chat Completions API.\n\n"
            "Authentication: set OPENAI_API_KEY env var, or use --api-key."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_id",
                        help="OpenAI model ID (e.g. gpt-4o-mini, gpt-4o)")
    parser.add_argument("--api-key", default=None, metavar="KEY",
                        help="OpenAI API key (overrides OPENAI_API_KEY env var)")
    parser.add_argument("--experiments", nargs="+", type=int,
                        choices=[1, 2, 3], default=[1, 2, 3], metavar="N",
                        help="Which experiments to run (default: 1 2 3)")
    parser.add_argument("--verbose-report", action="store_true",
                        help="Pass --verbose-report to each eval run")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Evaluate only the first N questions per condition "
                             "(useful for quick testing)")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap iterations per eval run (default: 1000)")

    args = parser.parse_args()

    # Resolve API key (the eval script handles the same resolution, but we
    # resolve here too so we can fail fast before launching any subprocesses).
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        # Check for key file as fallback
        key_file = SCRIPT_DIR / "openai_api_key.txt"
        if key_file.exists():
            api_key = key_file.read_text().strip()
    if not api_key:
        print(
            "ERROR: No OpenAI API key found.\n"
            "  Set OPENAI_API_KEY env var, use --api-key, or write the key to\n"
            f"  {SCRIPT_DIR / 'openai_api_key.txt'}"
        )
        sys.exit(1)

    print(f"Model:       {args.model_id}")
    print(f"Experiments: {args.experiments}")
    if args.limit:
        print(f"Limit:       {args.limit} questions per condition (test mode)")

    if 1 in args.experiments:
        run_experiment_1(args.model_id, api_key,
                         args.verbose_report, args.limit, args.n_bootstrap)

    if 2 in args.experiments:
        run_experiment_2(args.model_id, api_key,
                         args.verbose_report, args.limit, args.n_bootstrap)

    if 3 in args.experiments:
        run_experiment_3(args.model_id, api_key,
                         args.verbose_report, args.limit, args.n_bootstrap)

    print("\n" + "=" * 60)
    print("  All requested experiments complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
