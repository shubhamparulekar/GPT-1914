#!/usr/bin/env python3
"""
prepare_data.py

Metadata-instruction fine-tuning data preparation for the ChronoLogic experiment.

Experiment design
-----------------
This prepares a single-variable fine-tuning dataset that teaches a model to
generate period-appropriate text when conditioned on book metadata. The format is:

  User:      "Write a passage from <title>, a <genre> published in <year> by
              <author>, a <nationality> <profession>."
  Assistant: <verbatim passage from that book>

This is a CLEAN SINGLE-VARIABLE TEST. Unlike the hybrid SFT in qwentuning/
(which mixes historical-text continuation, instruction following, and
motive-detection tasks), this dataset contains ONLY metadata-conditioned
passage generation examples. The hypothesis being tested: does exposing the
model to (metadata → historical passage) pairs at fine-tuning time cause it to
attend to metadata more effectively at inference time?

Data source
-----------
chronologic_en_0.1.jsonl — the ChronoLogic benchmark. Each question with a
non-empty `passage` field contributes one training example. The `passage` field
is the verbatim historical text excerpt used to construct each benchmark question.

Question types that typically have passages:
  - phrase_cloze        (passage is the surrounding context paragraph)
  - sentence_cloze      (same)
  - character_modeling  (passage is a long narrative excerpt)
  - topic_sentence      (passage is the headless paragraph)

Split strategy
--------------
The split is done by source book (source_htid), NOT by question, to prevent
data leakage where the model sees passages from the same book in both train
and eval. Approximate ratios: 80% train / 10% val / 10% test.

Deduplication
-------------
Multiple questions can share the same passage (e.g. several cloze questions
drawn from the same paragraph). Exact-duplicate passages within the same book
are kept only once in the training data.

Usage
-----
  python prepare_data.py                   # full run → data/train.jsonl etc.
  python prepare_data.py --smoke           # 50 examples → sample_train.jsonl
  python prepare_data.py --stats           # print stats and exit (no files written)
  python prepare_data.py --source PATH     # override default benchmark path
  python prepare_data.py --min_passage N   # minimum passage length (default: 100 chars)
  python prepare_data.py --seed N          # random seed (default: 42)
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
DEFAULT_SOURCE = SCRIPT_DIR.parent / "booksample" / "chronologic_en_0.2.jsonl"
DATA_DIR = SCRIPT_DIR / "data"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list:
    """Read a JSONL file and return a list of dicts."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def write_jsonl(examples: list, path: Path) -> None:
    """Write a list of message-format dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            record = make_example(ex["instruction"], ex["passage"])
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(examples):,} examples -> {path}")


# ---------------------------------------------------------------------------
# Instruction builder
# ---------------------------------------------------------------------------

def build_instruction(q: dict) -> str:
    """
    Build a metadata-conditioned generation instruction from a question's fields.

    Example output:
        "Write a passage from 'The Indian Eye on English Life', a travelogue
         published in 1891 by Behramji M. Malabari, a Indian poet and social reformer."
    """
    title = (q.get("source_title") or "").strip()
    genre = (q.get("source_genre") or "").strip()
    year = q.get("source_date")  # int or None
    author = (q.get("source_author") or "").strip()
    nationality = (q.get("author_nationality") or "").strip()
    profession = (q.get("author_profession") or "").strip()

    # Describe the book
    if genre and title:
        book_desc = f"a {genre} called '{title}'"
    elif genre:
        book_desc = f"a {genre}"
    elif title:
        book_desc = f"'{title}'"
    else:
        book_desc = "a historical work"

    if year:
        book_desc += f", published in {year}"

    if author:
        if nationality and profession:
            author_desc = f"{author}, a {nationality} {profession}"
        elif nationality:
            author_desc = f"{author} ({nationality})"
        elif profession:
            author_desc = f"{author}, a {profession}"
        else:
            author_desc = author
        book_desc += f" by {author_desc}"

    return f"Write a passage from {book_desc}."


# ---------------------------------------------------------------------------
# OCR artifact cleaning
# ---------------------------------------------------------------------------

def clean_passage(text: str) -> str:
    """
    Remove OCR artifacts from digitised historical book passages.

    The raw passages in chronologic_en_0.1.jsonl come from HathiTrust OCR
    and contain several common artifacts that corrupt fine-tuning:

    1. Standalone page numbers — lines that are only a number, e.g. "\n123\n"
    2. Running headers — ALL-CAPS lines like "THE CELLAR-HOUSE" or "CHAPTER III"
       that repeat the book/chapter title mid-passage
    3. Numeric repetition runs — "126. 127. 128. 129..." (OCR'd page/footnote refs)
    4. Excess whitespace left behind after removal

    Prose content and inline numbers (dates, statistics) are preserved.
    """
    # 1. Remove standalone page numbers on their own line, e.g. "\n123\n"
    text = re.sub(r'\n[ \t]*\d{1,4}[ \t]*\n', '\n', text)

    # 2. Remove ALL-CAPS header + page-number on their own line,
    #    e.g. "\nTHE CELLAR-HOUSE 123\n"
    text = re.sub(r'\n[ \t]*[A-Z][A-Z\s\-:\'\.]{3,40}[ \t]+\d{1,4}[ \t]*\n', '\n', text)

    # 3. Remove lines that are purely an ALL-CAPS running header,
    #    e.g. "\nTHE CELLAR-HOUSE\n" or "\nCHAPTER III\n"
    text = re.sub(r'\n[ \t]*[A-Z][A-Z\s\-:\'\.]{3,60}[ \t]*\n', '\n', text)

    # 4. Remove inline "HEADER. PAGE_NUM " prefix merged into a prose line,
    #    e.g. "\nLONDON. 253 fingers..." or "\nROSINE. 67 this or that..."
    #    Pattern: newline, ALL-CAPS word(s), period, 1-4 digit page number, space.
    text = re.sub(r'\n[A-Z][A-Z\s_]{2,30}\.\s*\d{1,4}\s+', '\n', text)

    # 5. Remove "HEADER. " prefix without a page number (e.g. "\nROSINE. text")
    #    only when HEADER is a single capitalised token (book/character name used
    #    as a running header), not a sentence start.
    text = re.sub(r'\n[A-Z]{3,20}\.\s+(?=[a-z"\'«])', '\n', text)

    # 6. Remove numeric repetition runs: three or more numbers in sequence,
    #    e.g. "126. 127. 128. 129." — OCR'd page/footnote reference lists
    text = re.sub(r'(\b\d{1,4}[\.\,\s]+){3,}', ' ', text)

    # 7. Collapse multiple blank lines and strip leading/trailing whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


# ---------------------------------------------------------------------------
# Example format
# ---------------------------------------------------------------------------

def make_example(instruction: str, passage: str) -> dict:
    """
    Return a messages-format training example compatible with train_qlora.py.

    The normalize_examples() function in train_qlora.py will apply the Qwen
    chat template to convert this to a plain text string at training time.
    """
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": passage},
        ]
    }


# ---------------------------------------------------------------------------
# Data collection and deduplication
# ---------------------------------------------------------------------------

def collect_examples(questions: list, min_passage: int = 100) -> list:
    """
    Extract training examples from benchmark questions.

    Filters to questions with a non-empty `passage` field of at least
    `min_passage` characters. Deduplicates: if two questions from the same
    book have identical passage text, only the first is kept.

    Returns a list of dicts with keys: instruction, passage, htid.
    """
    examples = []
    seen: dict = defaultdict(set)  # htid → set of seen passage strings

    for q in questions:
        passage = clean_passage((q.get("passage") or "").strip())
        if len(passage) < min_passage:
            continue

        htid = q.get("source_htid") or "unknown"
        if passage in seen[htid]:
            continue
        seen[htid].add(passage)

        instruction = build_instruction(q)
        examples.append({
            "instruction": instruction,
            "passage": passage,
            "htid": htid,
            "reasoning_type": q.get("reasoning_type", "unknown"),
        })

    return examples


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def split_by_book(examples: list, seed: int = 42) -> tuple:
    """
    Split examples 80 / 10 / 10 by source book (htid) to prevent leakage.

    Books are assigned entirely to one split, so the model never sees a
    passage from book X during training and then must generate a passage
    from book X during evaluation.

    Returns (train_examples, val_examples, test_examples).
    """
    by_book: dict = defaultdict(list)
    for ex in examples:
        by_book[ex["htid"]].append(ex)

    books = sorted(by_book.keys())
    rng = random.Random(seed)
    rng.shuffle(books)

    n = len(books)
    n_val = max(1, round(n * 0.10))
    n_test = max(1, round(n * 0.10))
    n_train = n - n_val - n_test

    train_books = set(books[:n_train])
    val_books = set(books[n_train: n_train + n_val])
    test_books = set(books[n_train + n_val:])

    train_ex = [ex for ex in examples if ex["htid"] in train_books]
    val_ex = [ex for ex in examples if ex["htid"] in val_books]
    test_ex = [ex for ex in examples if ex["htid"] in test_books]

    return train_ex, val_ex, test_ex


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(questions: list, examples: list) -> None:
    """Print a summary of the benchmark and the extracted examples."""
    n_q = len(questions)
    n_ex = len(examples)
    n_books = len(set(ex["htid"] for ex in examples))

    # Count questions with passages (before dedup) by reasoning_type
    rt_counts: dict = defaultdict(int)
    for q in questions:
        if (q.get("passage") or "").strip():
            rt_counts[q.get("reasoning_type", "unknown")] += 1

    q_with_passage = sum(rt_counts.values())

    sep = "-" * 55
    print(f"\n{sep}")
    print(f"Benchmark questions total:      {n_q}")
    print(f"Questions with passage field:   {q_with_passage}")
    print(f"Training examples (after dedup):{n_ex:>5}")
    print(f"Unique source books:            {n_books}")
    print(f"\nPassage-bearing questions by reasoning_type:")
    for rt, cnt in sorted(rt_counts.items(), key=lambda x: -x[1]):
        print(f"  {rt:<30} {cnt:>4}")

    if examples:
        lengths = sorted(len(ex["passage"]) for ex in examples)
        n = len(lengths)
        print(f"\nPassage char length -- "
              f"min: {lengths[0]}, "
              f"median: {lengths[n // 2]}, "
              f"max: {lengths[-1]}")

        # Example instruction
        print(f"\nExample instruction:")
        print(f"  {examples[0]['instruction']}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare metadata-instruction fine-tuning data (ChronoLogic experiment).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", default=str(DEFAULT_SOURCE), metavar="PATH",
        help=f"Path to benchmark JSONL (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--min_passage", type=int, default=100, metavar="N",
        help="Minimum passage length in characters (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val/test split (default: 42)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Write 50 shuffled examples to sample_train.jsonl instead of full split",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print statistics and exit without writing files",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Benchmark not found: {source_path}\n"
            "Pass --source to specify its location."
        )

    print(f"Loading benchmark from {source_path} ...")
    questions = load_jsonl(source_path)
    examples = collect_examples(questions, min_passage=args.min_passage)

    print_stats(questions, examples)

    if args.stats:
        return

    if args.smoke:
        rng = random.Random(args.seed)
        rng.shuffle(examples)
        sample = examples[:50]
        out = SCRIPT_DIR / "sample_train.jsonl"
        write_jsonl(sample, out)
        print("Smoke sample written. To test training:")
        print(f"  python ../qwentuning/train_qlora.py --smoke")
        return

    train_ex, val_ex, test_ex = split_by_book(examples, seed=args.seed)
    print(f"Split: {len(train_ex)} train / {len(val_ex)} val / {len(test_ex)} test "
          f"(by source book)")
    write_jsonl(train_ex, DATA_DIR / "train.jsonl")
    write_jsonl(val_ex, DATA_DIR / "val.jsonl")
    write_jsonl(test_ex, DATA_DIR / "test.jsonl")

    print("\nNext steps:")
    print("  1. Inspect a few examples:  head -n 2 data/train.jsonl | python -m json.tool")
    print("  2. Submit training job:     sbatch h100.slurm")


if __name__ == "__main__":
    main()
