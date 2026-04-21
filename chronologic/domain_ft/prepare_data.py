#!/usr/bin/env python3
"""
prepare_data.py -- Domain fine-tuning data preparation from Project Gutenberg.

Downloads ~100 English books (1860-1925), cleans boilerplate, chunks into
passages, splits 80/10/10 by book, writes {"text": "..."} JSONL for training.

Genre split: ~40% fiction (novels) + ~60% non-fiction (essays, history, memoir).
Books that overlap with the ChronoLogic benchmark are excluded.

Usage:
    python prepare_data.py              # full run (~100 books)
    python prepare_data.py --smoke      # 10 books, quick test
    python prepare_data.py --stats      # show stats without writing files
    python prepare_data.py --num-books 50
"""



import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CATALOG_CACHE = DATA_DIR / "gutenberg_catalog.csv"
BENCHMARK_PATH = SCRIPT_DIR.parent / "booksample" / "chronologic_en_0.2.jsonl"

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
BOOK_URL_TEMPLATES = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ERA_START = 1855
ERA_END   = 1930

MIN_PASSAGE_CHARS = 200
MAX_PASSAGE_CHARS = 1000

FICTION_LOCC    = ("PR", "PS", "PZ")
NONFICTION_LOCC = ("D", "E", "F", "H", "B", "G", "CT", "PE")


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def normalize_author(author: str) -> str:
    name = re.sub(r",?\s*\d{4}[-–]\d{4}", "", author)
    name = re.sub(r",?\s*\d{4}[-–]\?", "", name)
    return re.sub(r"[^a-z ]", "", name.lower()).strip()


# ---------------------------------------------------------------------------
# Era and genre classification
# ---------------------------------------------------------------------------

def in_era(authors_str: str) -> bool:
    years = [int(m) for m in re.findall(r"\b(1[6-9]\d{2})\b", authors_str)]
    return any(ERA_START <= y <= ERA_END for y in years)


def classify_genre(locc: str) -> str:
    locc = locc.strip()
    for prefix in FICTION_LOCC:
        if locc.startswith(prefix):
            return "fiction"
    for prefix in NONFICTION_LOCC:
        if locc.startswith(prefix):
            return "nonfiction"
    return "other"


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def download_catalog() -> list:
    if not CATALOG_CACHE.exists():
        print(f"Downloading Gutenberg catalog ...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        r = requests.get(CATALOG_URL, timeout=120)
        r.raise_for_status()
        CATALOG_CACHE.write_bytes(r.content)
        print(f"  Saved -> {CATALOG_CACHE}")
    else:
        print(f"Using cached catalog: {CATALOG_CACHE}")

    books = []
    with open(CATALOG_CACHE, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            books.append(row)
    print(f"  {len(books)} catalog entries")
    return books


def filter_catalog(books: list, bench_titles: set, bench_authors: set) -> dict:
    fiction, nonfiction = [], []

    for b in books:
        if b.get("Language", "").strip() != "en":
            continue
        if b.get("Type", "").strip() != "Text":
            continue
        if not in_era(b.get("Authors", "")):
            continue

        title_norm  = normalize_title(b.get("Title", ""))
        author_norm = normalize_author(b.get("Authors", ""))

        # Exclude if title AND author both match benchmark
        overlap = False
        for bt in bench_titles:
            if bt and (bt in title_norm or title_norm in bt):
                for ba in bench_authors:
                    if ba and ba in author_norm:
                        overlap = True
                        break
            if overlap:
                break
        if overlap:
            continue

        genre = classify_genre(b.get("LoCC", ""))
        if genre == "fiction":
            fiction.append(b)
        elif genre == "nonfiction":
            nonfiction.append(b)

    print(f"  Candidates: {len(fiction)} fiction, {len(nonfiction)} nonfiction")
    return {"fiction": fiction, "nonfiction": nonfiction}


def select_books(candidates: dict, n_fiction: int, n_nonfiction: int, seed: int) -> list:
    rng = random.Random(seed)
    fic = rng.sample(candidates["fiction"],    min(n_fiction,    len(candidates["fiction"])))
    nf  = rng.sample(candidates["nonfiction"], min(n_nonfiction, len(candidates["nonfiction"])))
    selected = fic + nf
    print(f"  Selected: {len(fic)} fiction + {len(nf)} nonfiction = {len(selected)} books")
    return selected


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_book(gutenberg_id: str, delay: float = 0.5) -> str | None:
    raw_path = RAW_DIR / f"{gutenberg_id}.txt"
    if raw_path.exists():
        return raw_path.read_text(encoding="utf-8", errors="replace")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for template in BOOK_URL_TEMPLATES:
        url = template.format(id=gutenberg_id)
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 2000:
                text = r.content.decode("utf-8", errors="replace")
                raw_path.write_text(text, encoding="utf-8")
                time.sleep(delay)
                return text
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_book(text: str) -> str:
    """Strip Gutenberg boilerplate and structural noise."""
    # 1. Extract between START/END markers (find both before slicing)
    start = re.search(
        r"\*{3}\s*START OF (?:THE )?PROJECT GUTENBERG EBOOK[^\*]*\*{3}",
        text, re.IGNORECASE
    )
    end = re.search(
        r"\*{3}\s*END OF (?:THE )?PROJECT GUTENBERG EBOOK[^\*]*\*{3}",
        text, re.IGNORECASE
    )
    start_pos = start.end()   if start else 0
    end_pos   = end.start()   if end   else len(text)
    text = text[start_pos:end_pos]

    # 2. Transcriber / editor notes
    text = re.sub(
        r"\[(?:Transcriber|Editor|Illustrator)[^\]]{0,300}\]",
        "", text, flags=re.IGNORECASE
    )
    # 3. Illustration markers
    text = re.sub(
        r"\[(?:Illustration|Image|Fig\.?|Figure)[^\]]{0,200}\]",
        "", text, flags=re.IGNORECASE
    )
    # 4. Inline footnote references [1], [2]
    text = re.sub(r"\[\d{1,3}\]", "", text)

    # 5. Footnote blocks [Footnote N: ...]
    text = re.sub(
        r"\[Footnote \d+:[^\]]*\]", "", text,
        flags=re.IGNORECASE | re.DOTALL
    )
    # 6. Chapter / section headings
    text = re.sub(
        r"\n[ \t]*(?:CHAPTER|SECTION|PART|BOOK|VOLUME)[\s\w.\-]{0,60}\n",
        "\n", text, flags=re.IGNORECASE
    )
    # 7. Standalone Roman numerals
    text = re.sub(r"\n[ \t]*[IVXLCDM]{1,8}\.?\s*\n", "\n", text)

    # 8. ALL-CAPS lines (headers)
    lines = text.split("\n")
    lines = [
        l for l in lines
        if not (l.strip()
                and l.strip() == l.strip().upper()
                and 3 < len(l.strip()) < 80
                and re.search(r"[A-Z]", l))
    ]
    text = "\n".join(lines)

    # 9. Normalize whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, min_chars: int = MIN_PASSAGE_CHARS,
               max_chars: int = MAX_PASSAGE_CHARS) -> list:
    """Group paragraphs into passage-sized chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    chunks, current, current_len = [], [], 0
    for para in paragraphs:
        if len(para) < 30:          # skip very short (stray headers etc.)
            continue
        if current_len + len(para) > max_chars and current_len >= min_chars:
            chunks.append(" ".join(current))
            current, current_len = [para], len(para)
        else:
            current.append(para)
            current_len += len(para) + 1

    if current and current_len >= min_chars:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_examples(selected: list) -> list:
    examples = []
    failed = []

    for i, book in enumerate(selected):
        gid   = book["Text#"].strip()
        title = book["Title"].strip()
        print(f"  [{i+1}/{len(selected)}] {gid}: {title[:55]}", end=" ... ", flush=True)

        raw = download_book(gid)
        if raw is None:
            print("FAILED (download)")
            failed.append(gid)
            continue

        cleaned = clean_book(raw)
        chunks  = chunk_text(cleaned)

        if not chunks:
            print("FAILED (no chunks)")
            failed.append(gid)
            continue

        for chunk in chunks:
            examples.append({"text": chunk, "gutenberg_id": gid, "title": title})

        print(f"OK ({len(chunks)} chunks)")

    if failed:
        print(f"\n  Failed: {failed}")
    return examples


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_by_book(examples: list, seed: int = 42) -> tuple:
    """80/10/10 split by gutenberg_id — no book appears in two splits."""
    rng   = random.Random(seed)
    books = list({ex["gutenberg_id"] for ex in examples})
    rng.shuffle(books)

    n       = len(books)
    n_test  = max(1, int(n * 0.1))
    n_val   = max(1, int(n * 0.1))

    test_books  = set(books[:n_test])
    val_books   = set(books[n_test:n_test + n_val])
    train_books = set(books[n_test + n_val:])

    train = [ex for ex in examples if ex["gutenberg_id"] in train_books]
    val   = [ex for ex in examples if ex["gutenberg_id"] in val_books]
    test  = [ex for ex in examples if ex["gutenberg_id"] in test_books]
    return train, val, test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_jsonl(examples: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"text": ex["text"]}, ensure_ascii=False) + "\n")


def load_benchmark_books() -> tuple:
    titles, authors = set(), set()
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            t = normalize_title(q.get("source_title", ""))
            a = normalize_author(q.get("source_author", ""))
            if t:
                titles.add(t)
            if a:
                authors.add(a)
    return titles, authors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare domain FT data from Project Gutenberg."
    )
    parser.add_argument("--stats",       action="store_true", help="Show stats only, no file writes")
    parser.add_argument("--smoke",       action="store_true", help="10 books only")
    parser.add_argument("--num-books",   type=int,   default=100)
    parser.add_argument("--fiction-ratio", type=float, default=0.4)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--min-passage", type=int,   default=MIN_PASSAGE_CHARS)
    parser.add_argument("--max-passage", type=int,   default=MAX_PASSAGE_CHARS)
    args = parser.parse_args()

    n_total     = 10 if args.smoke else args.num_books
    n_fiction   = int(n_total * args.fiction_ratio)
    n_nonfiction = n_total - n_fiction

    print("\nStep 1: Loading benchmark books for exclusion ...")
    bench_titles, bench_authors = load_benchmark_books()
    print(f"  {len(bench_titles)} titles / {len(bench_authors)} authors to exclude")

    print("\nStep 2: Loading Gutenberg catalog ...")
    catalog = download_catalog()

    print("\nStep 3: Filtering catalog ...")
    candidates = filter_catalog(catalog, bench_titles, bench_authors)

    print("\nStep 4: Selecting books ...")
    selected = select_books(candidates, n_fiction, n_nonfiction, args.seed)

    print("\nStep 5: Downloading and processing books ...")
    examples = collect_examples(selected)

    total_chars  = sum(len(ex["text"]) for ex in examples)
    unique_books = len({ex["gutenberg_id"] for ex in examples})

    print(f"\n{'='*55}")
    print(f"Books processed:    {unique_books}")
    print(f"Total passages:     {len(examples)}")
    print(f"Total chars:        {total_chars:,}")
    print(f"Approx tokens:      {total_chars // 4:,}")
    print(f"Avg passage chars:  {total_chars // max(len(examples), 1):.0f}")

    if args.stats:
        return

    print("\nStep 6: Splitting by book ...")
    train, val, test = split_by_book(examples, args.seed)
    print(f"  {len(train)} train / {len(val)} val / {len(test)} test")

    print("\nStep 7: Writing files ...")
    write_jsonl(train, DATA_DIR / "train.jsonl")
    write_jsonl(val,   DATA_DIR / "val.jsonl")
    write_jsonl(test,  DATA_DIR / "test.jsonl")

    # Save selected book list for reproducibility
    sel_path = DATA_DIR / "gutenberg_selected.csv"
    genre_map = {b["Text#"]: "fiction"    for b in candidates["fiction"]}
    genre_map.update({b["Text#"]: "nonfiction" for b in candidates["nonfiction"]})
    with open(sel_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Text#", "Title", "Authors", "LoCC", "genre"])
        w.writeheader()
        for b in selected:
            w.writerow({
                "Text#":   b["Text#"],
                "Title":   b["Title"],
                "Authors": b["Authors"],
                "LoCC":    b["LoCC"],
                "genre":   genre_map.get(b["Text#"], "unknown"),
            })

    print(f"  train.jsonl        ({len(train)} passages)")
    print(f"  val.jsonl          ({len(val)} passages)")
    print(f"  test.jsonl         ({len(test)} passages)")
    print(f"  gutenberg_selected.csv ({len(selected)} books)")
    print(f"\nNext: push to git, then run training on H100.")


if __name__ == "__main__":
    main()
