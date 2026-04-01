"""
run_chatgpt_eval.py
-------------------
Self-contained MCQ evaluation script using the OpenAI Chat Completions API
(compatible with gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, etc.).

This is a shareable, dependency-light alternative to benchmark_evaluation.py.
It does NOT require PyTorch, HuggingFace, or a local GPU — only the `openai`
package and a valid API key.

Setup:
    pip install openai numpy tqdm

Authentication (pick one):
    export OPENAI_API_KEY="sk-..."          # recommended
    echo "sk-..." > openai_api_key.txt      # or write key to file
    python run_chatgpt_eval.py ... --api-key sk-...  # or pass directly

Usage examples:
    # Evaluate gpt-4o-mini on the main benchmark (MCQ mode):
    python run_chatgpt_eval.py gpt-4o-mini ../booksample/chronologic_en_0.1.jsonl

    # Full verbose report, custom output directory:
    python run_chatgpt_eval.py gpt-4o-mini ../booksample/chronologic_en_0.1.jsonl \\
        --verbose-report --output-dir results/

    # Experiment 1 — pairwise benchmark:
    python run_chatgpt_eval.py gpt-4o-mini experiment_1/benchmark_pairwise.jsonl \\
        --output-dir experiment_1/results_chatgpt/

    # Experiment 2 — metadata ablation (run once per condition):
    python run_chatgpt_eval.py gpt-4o-mini ../booksample/chronologic_en_0.1.jsonl \\
        --output-dir experiment_2/results_A_chatgpt/
    python run_chatgpt_eval.py gpt-4o-mini experiment_2/benchmark_no_metadata.jsonl \\
        --output-dir experiment_2/results_B_chatgpt/
    python run_chatgpt_eval.py gpt-4o-mini experiment_2/benchmark_shuffled_metadata.jsonl \\
        --output-dir experiment_2/results_C_chatgpt/

Output:
    eval_results_mcq_<model>_<timestamp>.json   — machine-readable (compatible with
                                                   existing analysis scripts)
    eval_report_mcq_<model>_<timestamp>.md      — human-readable (with --verbose-report)
"""

import json
import math
import os
import random
import re
import datetime
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------

def load_api_key(api_key_arg=None, key_file=None):
    """Resolve the OpenAI API key from multiple sources (in priority order):
      1. --api-key command-line argument
      2. OPENAI_API_KEY environment variable
      3. openai_api_key.txt in this script's directory
      4. key_file path (if provided)

    Args:
        api_key_arg: value of --api-key CLI argument, or None.
        key_file:    path to a file containing the API key, or None.

    Returns:
        str: the API key, stripped of whitespace.

    Raises:
        ValueError: if no key can be found.
    """
    if api_key_arg:
        return api_key_arg.strip()

    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    candidates = []
    if key_file:
        candidates.append(Path(key_file))
    candidates.append(Path(__file__).parent / "openai_api_key.txt")

    for path in candidates:
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            if content:
                return content

    raise ValueError(
        "No OpenAI API key found. Provide one via:\n"
        "  1. export OPENAI_API_KEY='sk-...'\n"
        "  2. --api-key sk-... on the command line\n"
        "  3. A file named openai_api_key.txt in the evalcode directory"
    )


# ---------------------------------------------------------------------------
# MCQ prompt construction and response parsing
# ---------------------------------------------------------------------------

def _build_mcq_prompt(question, use_metadata=True, include_negation=False):
    """Build a multiple-choice prompt for a benchmark question.

    Args:
        question:         dict conforming to the benchmark question format.
        use_metadata:     if True, prepend metadata_frame to the prompt.
        include_negation: if False (default), drop negation-type answers.

    Returns:
        tuple: (prompt_text, answer_order, correct_letter)
    """
    entries = list(zip(
        question["answer_strings"],
        question["answer_types"],
        question["answer_probabilities"],
    ))

    if not include_negation:
        entries = [(s, t, p) for s, t, p in entries if t != "negation"]

    random.shuffle(entries)

    letters = [chr(ord("A") + i) for i in range(len(entries))]
    answer_order = [(letter, s, p) for letter, (s, t, p) in zip(letters, entries)]

    correct_letter = None
    for letter, s, p in answer_order:
        if p == 1.0:
            correct_letter = letter
            break

    lines = []
    if use_metadata:
        lines.append(question.get("metadata_frame", ""))
    lines.append(f"QUESTION: {question['main_question']}")
    for letter, s, p in answer_order:
        lines.append(f"{letter}) {s}")
    lines.append("Respond only with the letter of the correct answer:")

    prompt_text = "\n".join(lines)
    return prompt_text, answer_order, correct_letter


def _parse_mcq_response(response_text):
    """Extract the first uppercase letter from a model response.

    Returns:
        str: the first uppercase letter, or None if none found.
    """
    match = re.search(r"\b[A-Z]\b", response_text)
    return match.group(0) if match else None


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

def _generate_chatgpt(prompt, model_id, client, max_tokens=10,
                      temperature=0, max_retries=5):
    """Call the Chat Completions API and return the model's response text.

    Uses exponential backoff for rate-limit (429) errors.

    Args:
        prompt:      the full MCQ prompt string.
        model_id:    OpenAI model identifier (e.g. 'gpt-4o-mini').
        client:      an openai.OpenAI client instance.
        max_tokens:  maximum tokens to generate (MCQ answer is 1 letter).
        temperature: sampling temperature (0 = deterministic).
        max_retries: number of retry attempts after initial rate-limit failure.

    Returns:
        str: the model's response content.
    """
    try:
        from openai import RateLimitError
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required. Install it with: pip install openai"
        ) from exc

    messages = [
        {
            "role": "system",
            "content": "Choose the best answer. Respond with only the letter of the correct answer.",
        },
        {"role": "user", "content": prompt},
    ]

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except RateLimitError as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  Rate limit hit; retrying in {wait}s "
                      f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise RateLimitError(
                    f"Rate limit exceeded after {max_retries} retries."
                ) from last_exc


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def calculate_mcq_skill_score(is_correct, k, r=1):
    """Chance-adjusted skill score for a single MCQ question.

    score = (observed – chance) / (1 – chance)
    Expected value at chance = 0.0; perfect = 1.0.

    Args:
        is_correct: bool.
        k:          number of answer choices.
        r:          number of correct answers (default 1).

    Returns:
        float
    """
    observed = 1.0 if is_correct else 0.0
    chance = r / k
    if chance == 1.0:
        return 1.0
    return (observed - chance) / (1.0 - chance)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (MCQ mode only)
# ---------------------------------------------------------------------------

DEFAULT_SUBSET_FIELDS = [
    "question_category", "answer_length", "frame_type", "reasoning_type",
]


def build_subset_index(questions, fields=None):
    """Build a dict mapping 'field:value' → list of question indices."""
    if fields is None:
        fields = DEFAULT_SUBSET_FIELDS
    index = {}
    for i, q in enumerate(questions):
        for field in fields:
            if field not in q:
                continue
            key = f"{field}:{q[field]}"
            index.setdefault(key, []).append(i)
    return index


def bootstrap_evaluate_mcq(mcq_results, subset_index, n_bootstrap=1000):
    """Bootstrap confidence intervals (accuracy and skill score) for MCQ mode.

    Args:
        mcq_results:  list of dicts {"is_correct": bool, "k": int}.
        subset_index: dict from build_subset_index().
        n_bootstrap:  number of bootstrap iterations.

    Returns:
        dict: {"overall_benchmark": {"accuracy": [p2.5, p50, p97.5],
                                     "skill_score": [p2.5, p50, p97.5]}, ...}
    """
    n = len(mcq_results)
    rng = np.random.default_rng()

    per_q_correct = np.array([r["is_correct"] for r in mcq_results], dtype=float)
    per_q_k = np.array([r["k"] for r in mcq_results])
    per_q_skill = np.array([
        calculate_mcq_skill_score(bool(c), int(k))
        for c, k in zip(per_q_correct, per_q_k)
    ])

    all_subsets = {"overall_benchmark": np.arange(n)}
    for key, indices in subset_index.items():
        all_subsets[key] = np.array(indices)

    collectors = {key: {"accuracy": [], "skill_score": []}
                  for key in all_subsets}

    for _ in range(n_bootstrap):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        weights = counts.astype(float)

        for key, indices in all_subsets.items():
            w = weights[indices]
            w_sum = w.sum()
            if w_sum < 1e-12:
                collectors[key]["accuracy"].append(0.0)
                collectors[key]["skill_score"].append(0.0)
                continue
            collectors[key]["accuracy"].append(
                (w * per_q_correct[indices]).sum() / w_sum
            )
            collectors[key]["skill_score"].append(
                (w * per_q_skill[indices]).sum() / w_sum
            )

    result = {}
    for key in all_subsets:
        result[key] = {}
        for metric in ("accuracy", "skill_score"):
            vals = np.array(collectors[key][metric])
            p2_5, p50, p97_5 = np.percentile(vals, [2.5, 50, 97.5])
            result[key][metric] = [float(p2_5), float(p50), float(p97_5)]

    return result


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_json_report(report_path, metadata, per_question_results,
                      confidence_intervals, correct_vector):
    """Write the machine-readable JSON evaluation report.

    Output format is compatible with the existing analysis scripts
    (analyze_distractor_types.py, analyze_metadata_experiment.py, etc.).
    """
    report = {
        "metadata": metadata,
        "per_question_results": per_question_results,
        "confidence_intervals": confidence_intervals,
        "correct_vector": correct_vector,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def mcq_eval_chatgpt(model_id, path_to_jsonl, api_key,
                     include_negation=False, verbose_report=False,
                     n_bootstrap=1000, output_dir=None,
                     limit=None, trace=False):
    """Evaluate a ChatGPT model on multiple-choice benchmark questions.

    For each question: build MCQ prompt → call Chat Completions API →
    parse chosen letter → compare to ground truth.

    Always writes a JSON report. Writes a markdown report if verbose_report=True.

    Args:
        model_id:        OpenAI model ID (e.g. 'gpt-4o-mini', 'gpt-4o').
        path_to_jsonl:   path to benchmark JSONL file.
        api_key:         OpenAI API key string.
        include_negation: if True, include negation-type answers as distractors.
        verbose_report:  if True, write per-question markdown report.
        n_bootstrap:     bootstrap iterations for confidence intervals.
        output_dir:      directory for output files; defaults to JSONL's parent.
        limit:           if set, evaluate only the first N questions.
        trace:           if True, print each prompt and response to stdout.

    Returns:
        str: path to the JSON report file.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required. Install it with: pip install openai"
        ) from exc

    client = OpenAI(api_key=api_key)
    path = Path(path_to_jsonl)

    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    if limit is not None:
        questions = questions[:limit]

    model_safe = model_id.replace(":", "_").replace("/", "_")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir) if output_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_report_path = out_dir / f"eval_results_mcq_{model_safe}_{timestamp}.json"
    md_report_path = out_dir / f"eval_report_mcq_{model_safe}_{timestamp}.md"

    total = len(questions)
    try:
        from tqdm import tqdm
        question_iter = tqdm(enumerate(questions, 1), total=total, desc=model_id)
        _tqdm_active = True
    except ImportError:
        question_iter = enumerate(questions, 1)
        _tqdm_active = False

    correct_total = 0
    no_response_total = 0
    category_results = {}
    category_no_response = {}
    all_skill_scores = []
    category_skill = {}
    per_question_blocks = []
    all_mcq_results = []

    for i, question in question_iter:
        if not _tqdm_active and i % 25 == 0:
            print(f"  Processed {i} of {total} questions...")

        prompt_text, answer_order, correct_letter = _build_mcq_prompt(
            question, use_metadata=True, include_negation=include_negation
        )
        response_text = _generate_chatgpt(prompt_text, model_id, client)
        chosen_letter = _parse_mcq_response(response_text)

        if trace:
            sep = "=" * 60
            print(f"\n{sep}\nQUESTION {i}")
            print("── PROMPT ──────────────────────────────────────────────")
            print(prompt_text)
            print("── RESPONSE ────────────────────────────────────────────")
            print(repr(response_text))
            print(f"  chosen: {chosen_letter!r}  correct: {correct_letter!r}")
            print(sep)

        if chosen_letter is None:
            no_response_total += 1
            category = question.get("question_category", "unknown")
            print(f"  WARNING: no letter in response for question {i} "
                  f"(category: {category}) — raw: {repr(response_text)}")

        is_correct = chosen_letter == correct_letter
        if is_correct:
            correct_total += 1

        category = question.get("question_category", "unknown")
        category_results.setdefault(category, [0, 0])
        if is_correct:
            category_results[category][0] += 1
        category_results[category][1] += 1
        if chosen_letter is None:
            category_no_response[category] = category_no_response.get(category, 0) + 1

        k = len(answer_order)
        r = sum(1 for _, _, p in answer_order if p == 1.0)
        skill = calculate_mcq_skill_score(is_correct, k, r)
        all_skill_scores.append(skill)
        category_skill.setdefault(category, []).append(skill)
        all_mcq_results.append({"is_correct": is_correct, "k": k})

        if verbose_report:
            result_str = "TRUE" if is_correct else "FALSE"
            no_resp_str = "  ⚠ NO RESPONSE" if chosen_letter is None else ""
            block = [
                f"## Question {i}\n",
                f"**Metadata:** {question.get('metadata_frame', '')}\n",
                f"**Question:** {question.get('main_question', '')}\n",
                f"**Category:** {category}\n",
                "**Choices:**",
            ]
            for letter, ans_str, prob in answer_order:
                markers = []
                if letter == chosen_letter:
                    markers.append("←")
                if prob == 1.0:
                    markers.append("(ground truth)")
                block.append(f"- {letter}) {ans_str} {' '.join(markers)}".strip())
            block.append(f"\n**Model response:** `{response_text.strip()}`")
            block.append(
                f"**Chosen:** {chosen_letter}  **Correct:** {correct_letter}  "
                f"**Result:** {result_str}{no_resp_str}"
            )
            block.append(f"**Skill Score:** {skill:.4f}\n")
            per_question_blocks.append("\n".join(block))

    overall_accuracy = correct_total / total if total > 0 else 0.0
    no_response_rate = no_response_total / total if total > 0 else 0.0
    overall_mean_skill = (
        sum(all_skill_scores) / len(all_skill_scores) if all_skill_scores else 0.0
    )

    correct_vector = [r["is_correct"] for r in all_mcq_results]
    per_question_results = [
        {"model_probs": None, "chosen_letter": None, "correct": r["is_correct"]}
        for r in all_mcq_results
    ]

    subset_index = build_subset_index(questions)
    confidence_intervals = bootstrap_evaluate_mcq(
        all_mcq_results, subset_index, n_bootstrap=n_bootstrap
    )

    meta = {
        "model_id": model_id,
        "benchmark_file": str(path),
        "eval_mode": "mcq",
        "api": "openai_chat_completions",
        "timestamp": timestamp,
        "n_questions": total,
        "n_bootstrap": n_bootstrap,
    }
    write_json_report(json_report_path, meta, per_question_results,
                      confidence_intervals, correct_vector)

    if verbose_report:
        by_category = {
            cat: counts[0] / counts[1] if counts[1] > 0 else 0.0
            for cat, counts in category_results.items()
        }
        by_category_skill = {cat: sum(s) / len(s) for cat, s in category_skill.items()}

        summary_lines = [
            f"# MCQ Eval — {model_id}\n",
            "## Summary\n",
            f"**No-response rate:** {no_response_rate:.1%} "
            f"({no_response_total} of {total} questions returned no parseable letter)\n",
            "| Metric | Accuracy | Skill Score | No Response |",
            "|--------|----------|-------------|-------------|",
            f"| **Overall** | {overall_accuracy:.1%} | {overall_mean_skill:.4f} "
            f"| {no_response_total} / {total} |",
        ]
        for cat in sorted(by_category.keys()):
            acc = by_category[cat]
            skill_avg = by_category_skill.get(cat, 0.0)
            cat_total = category_results[cat][1]
            cat_no_resp = category_no_response.get(cat, 0)
            summary_lines.append(
                f"| {cat} | {acc:.1%} | {skill_avg:.4f} "
                f"| {cat_no_resp} / {cat_total} |"
            )
        summary_lines.append("")

        report_text = (
            "\n".join(summary_lines) + "\n" + "\n\n".join(per_question_blocks)
        )
        with open(md_report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Markdown report: {md_report_path}")

    print(f"Overall MCQ accuracy:    {overall_accuracy:.1%}")
    print(f"Overall MCQ skill score: {overall_mean_skill:.4f}")
    print(f"No-response rate:        {no_response_rate:.1%} "
          f"({no_response_total} of {total})")
    ci = confidence_intervals.get("overall_benchmark", {})
    if "accuracy" in ci:
        lo, mid, hi = ci["accuracy"]
        print(f"95% CI accuracy:         [{lo:.1%}, {hi:.1%}]")
    print(f"JSON report: {json_report_path}")
    return str(json_report_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "MCQ evaluation of ChronoLogic benchmark questions via the OpenAI\n"
            "Chat Completions API (gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.).\n\n"
            "Authentication (pick one):\n"
            "  export OPENAI_API_KEY='sk-...'\n"
            "  echo 'sk-...' > evalcode/openai_api_key.txt\n"
            "  --api-key sk-..."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_id",
                        help="OpenAI model ID, e.g. gpt-4o-mini, gpt-4o, gpt-4-turbo")
    parser.add_argument("path_to_jsonl",
                        help="Path to benchmark JSONL file")
    parser.add_argument("--api-key", default=None, metavar="KEY",
                        help="OpenAI API key (overrides OPENAI_API_KEY env var)")
    parser.add_argument("--key-file", default=None, metavar="PATH",
                        help="Path to file containing the API key")
    parser.add_argument("--output-dir", default=None, metavar="DIR",
                        help="Directory for output files (default: same as JSONL)")
    parser.add_argument("--verbose-report", action="store_true",
                        help="Also write a per-question markdown report")
    parser.add_argument("--include-negation", action="store_true",
                        help="Include negation-type answers as MCQ distractors")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap iterations for confidence intervals (default: 1000)")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Evaluate only the first N questions (useful for testing)")
    parser.add_argument("--trace", action="store_true",
                        help="Print each prompt and raw response to stdout")

    args = parser.parse_args()

    api_key = load_api_key(args.api_key, args.key_file)

    mcq_eval_chatgpt(
        model_id=args.model_id,
        path_to_jsonl=args.path_to_jsonl,
        api_key=api_key,
        include_negation=args.include_negation,
        verbose_report=args.verbose_report,
        n_bootstrap=args.n_bootstrap,
        output_dir=args.output_dir,
        limit=args.limit,
        trace=args.trace,
    )
