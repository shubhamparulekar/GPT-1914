#!/usr/bin/env python3
"""Tests for domain_ft/prepare_data.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from prepare_data import (
    clean_book,
    chunk_text,
    classify_genre,
    in_era,
    normalize_title,
    normalize_author,
    split_by_book,
)

# ---------------------------------------------------------------------------
# normalize_title / normalize_author
# ---------------------------------------------------------------------------

def test_normalize_title_lowercase():
    assert normalize_title("The Land of the Czar") == "the land of the czar"

def test_normalize_title_strips_punctuation():
    assert "!" not in normalize_title("Hello, World!")

def test_normalize_author_strips_years():
    assert "1843" not in normalize_author("Dickens, Charles, 1812-1870")

def test_normalize_author_lowercase():
    result = normalize_author("Twain, Mark, 1835-1910")
    assert result == result.lower()

# ---------------------------------------------------------------------------
# in_era
# ---------------------------------------------------------------------------

def test_in_era_valid():
    assert in_era("Dickens, Charles, 1812-1870") is True

def test_in_era_too_early():
    assert in_era("Shakespeare, William, 1564-1616") is False

def test_in_era_boundary():
    assert in_era("Author, Some, 1855-1920") is True

def test_in_era_no_years():
    assert in_era("Unknown Author") is False

# ---------------------------------------------------------------------------
# classify_genre
# ---------------------------------------------------------------------------

def test_fiction_PR():
    assert classify_genre("PR") == "fiction"

def test_fiction_PS():
    assert classify_genre("PS") == "fiction"

def test_fiction_PZ():
    assert classify_genre("PZ") == "fiction"

def test_nonfiction_D():
    assert classify_genre("D") == "nonfiction"

def test_nonfiction_E():
    assert classify_genre("E") == "nonfiction"

def test_nonfiction_H():
    assert classify_genre("H") == "nonfiction"

def test_other():
    assert classify_genre("Q") == "other"

def test_empty_locc():
    assert classify_genre("") == "other"

# ---------------------------------------------------------------------------
# clean_book
# ---------------------------------------------------------------------------

SAMPLE_BOOK = """
Some legal preamble text here.

*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE BOOK ***

This is the actual book content.
It spans multiple paragraphs.

Here is a second paragraph with real prose.

CHAPTER I

The story begins here with more prose content.

[Illustration: A picture of something]

The narrative continues after the illustration.

[Transcriber's note: Some note here]

More real content follows here in the text.

*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE BOOK ***

More legal text here that should be stripped.
"""

def test_clean_strips_preamble():
    result = clean_book(SAMPLE_BOOK)
    assert "Some legal preamble" not in result

def test_clean_strips_footer():
    result = clean_book(SAMPLE_BOOK)
    assert "More legal text here" not in result

def test_clean_keeps_content():
    result = clean_book(SAMPLE_BOOK)
    assert "actual book content" in result

def test_clean_removes_illustration():
    result = clean_book(SAMPLE_BOOK)
    assert "[Illustration:" not in result

def test_clean_removes_transcriber_note():
    result = clean_book(SAMPLE_BOOK)
    assert "Transcriber's note" not in result

def test_clean_removes_chapter_heading():
    result = clean_book(SAMPLE_BOOK)
    assert "CHAPTER I" not in result

def test_clean_removes_footnote_refs():
    text = "*** START OF THE PROJECT GUTENBERG EBOOK X ***\nSome text[1] here[2].\n*** END OF THE PROJECT GUTENBERG EBOOK X ***"
    result = clean_book(text)
    assert "[1]" not in result
    assert "[2]" not in result

def test_clean_no_markers_keeps_all():
    text = "This is plain text with no Gutenberg markers at all."
    result = clean_book(text)
    assert "plain text" in result

def test_clean_normalizes_whitespace():
    text = "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n\n\n\nParagraph.\n*** END OF THE PROJECT GUTENBERG EBOOK X ***"
    result = clean_book(text)
    assert "\n\n\n" not in result

# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

def _make_text(n_paragraphs: int, chars_each: int = 300) -> str:
    para = "a" * chars_each
    return "\n\n".join([para] * n_paragraphs)

def test_chunk_returns_list():
    assert isinstance(chunk_text(_make_text(5)), list)

def test_chunk_min_filter():
    short = "x" * 50
    result = chunk_text(short, min_chars=200)
    assert result == []

def test_chunk_respects_max():
    text = _make_text(10, 200)
    chunks = chunk_text(text, min_chars=100, max_chars=500)
    for c in chunks:
        assert len(c) <= 600  # some slack for joining

def test_chunk_no_empty():
    text = _make_text(5)
    chunks = chunk_text(text)
    assert all(c.strip() for c in chunks)

def test_chunk_preserves_content():
    text = "*** START ***\n\nThe quick brown fox " + ("word " * 50) + "\n\n" + ("more words " * 50) + "\n\n*** END ***"
    cleaned = clean_book(text)
    chunks = chunk_text(cleaned)
    combined = " ".join(chunks)
    assert "quick brown fox" in combined

# ---------------------------------------------------------------------------
# split_by_book
# ---------------------------------------------------------------------------

def _make_examples(n_books: int, passages_per_book: int = 10) -> list:
    examples = []
    for b in range(n_books):
        for _ in range(passages_per_book):
            examples.append({
                "text": "some passage",
                "gutenberg_id": str(b),
                "title": f"Book {b}",
            })
    return examples

def test_split_sizes():
    examples = _make_examples(20)
    train, val, test = split_by_book(examples, seed=42)
    assert len(train) + len(val) + len(test) == len(examples)

def test_split_no_book_leakage():
    examples = _make_examples(20)
    train, val, test = split_by_book(examples, seed=42)
    train_books = {ex["gutenberg_id"] for ex in train}
    val_books   = {ex["gutenberg_id"] for ex in val}
    test_books  = {ex["gutenberg_id"] for ex in test}
    assert train_books.isdisjoint(val_books)
    assert train_books.isdisjoint(test_books)
    assert val_books.isdisjoint(test_books)

def test_split_train_is_majority():
    examples = _make_examples(20)
    train, val, test = split_by_book(examples, seed=42)
    assert len(train) > len(val)
    assert len(train) > len(test)

def test_split_reproducible():
    examples = _make_examples(20)
    t1, v1, te1 = split_by_book(examples, seed=99)
    t2, v2, te2 = split_by_book(examples, seed=99)
    assert [ex["gutenberg_id"] for ex in t1] == [ex["gutenberg_id"] for ex in t2]

def test_split_different_seeds():
    examples = _make_examples(20)
    t1, _, _ = split_by_book(examples, seed=1)
    t2, _, _ = split_by_book(examples, seed=2)
    train_books_1 = {ex["gutenberg_id"] for ex in t1}
    train_books_2 = {ex["gutenberg_id"] for ex in t2}
    assert train_books_1 != train_books_2


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
