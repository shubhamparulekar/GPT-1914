#!/usr/bin/env python3
"""
test_prepare_data.py

Unit tests for prepare_data.py.

Run from the repo root:
    pytest chronologic/metadata_ft/tests/

Or from metadata_ft/:
    pytest tests/
"""

import json
import sys
from pathlib import Path

import pytest

# Allow importing prepare_data from the parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare_data import (
    build_instruction,
    collect_examples,
    make_example,
    split_by_book,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

FULL_QUESTION = {
    "source_title": "The Indian Eye on English Life",
    "source_genre": "travelogue",
    "source_date": 1891,
    "source_author": "Behramji M. Malabari",
    "author_nationality": "Indian",
    "author_profession": "poet and social reformer",
    "source_htid": "htid_001",
    "passage": "A long passage of historical text that exceeds the minimum length threshold.",
    "reasoning_type": "constrained_generation",
    "question_category": "parallax",
}


def make_q(**overrides):
    """Return a copy of FULL_QUESTION with any fields overridden."""
    q = dict(FULL_QUESTION)
    q.update(overrides)
    return q


def long_passage(n: int = 150) -> str:
    return "Historical prose. " * (n // 17 + 1)  # ~150+ chars


# ---------------------------------------------------------------------------
# build_instruction
# ---------------------------------------------------------------------------

class TestBuildInstruction:

    def test_contains_title(self):
        instr = build_instruction(make_q())
        assert "The Indian Eye on English Life" in instr

    def test_contains_genre(self):
        instr = build_instruction(make_q())
        assert "travelogue" in instr

    def test_contains_year(self):
        instr = build_instruction(make_q())
        assert "1891" in instr

    def test_contains_author(self):
        instr = build_instruction(make_q())
        assert "Behramji M. Malabari" in instr

    def test_contains_nationality(self):
        instr = build_instruction(make_q())
        assert "Indian" in instr

    def test_contains_profession(self):
        instr = build_instruction(make_q())
        assert "poet and social reformer" in instr

    def test_starts_with_write_a_passage(self):
        instr = build_instruction(make_q())
        assert instr.startswith("Write a passage from")

    def test_ends_with_period(self):
        instr = build_instruction(make_q())
        assert instr.endswith(".")

    def test_missing_nationality_omits_it(self):
        # "Indian" also appears in the book title, so check that nationality
        # is not appended to the author name (the "(Indian)" parenthetical form)
        instr = build_instruction(make_q(author_nationality=""))
        assert "(Indian)" not in instr
        assert "a Indian" not in instr
        # Author and profession should still appear
        assert "Behramji M. Malabari" in instr

    def test_missing_profession_omits_it(self):
        instr = build_instruction(make_q(author_profession=""))
        assert "poet" not in instr
        assert "Behramji M. Malabari" in instr

    def test_missing_nationality_and_profession(self):
        instr = build_instruction(make_q(author_nationality="", author_profession=""))
        assert "Behramji M. Malabari" in instr

    def test_missing_author_entirely(self):
        instr = build_instruction(make_q(source_author="", author_nationality="", author_profession=""))
        assert "Write a passage from" in instr

    def test_missing_genre(self):
        instr = build_instruction(make_q(source_genre=""))
        assert "The Indian Eye on English Life" in instr
        assert "travelogue" not in instr

    def test_missing_title(self):
        instr = build_instruction(make_q(source_title=""))
        assert "Write a passage from" in instr

    def test_missing_year(self):
        instr = build_instruction(make_q(source_date=None))
        assert "1891" not in instr

    def test_all_fields_missing_returns_string(self):
        q = {k: "" for k in FULL_QUESTION}
        q["source_date"] = None
        instr = build_instruction(q)
        assert isinstance(instr, str)
        assert len(instr) > 0

    def test_year_as_integer(self):
        instr = build_instruction(make_q(source_date=1905))
        assert "1905" in instr

    def test_no_duplicate_spaces(self):
        instr = build_instruction(make_q())
        assert "  " not in instr


# ---------------------------------------------------------------------------
# make_example
# ---------------------------------------------------------------------------

class TestMakeExample:

    def test_has_messages_key(self):
        ex = make_example("Instruction.", "Response.")
        assert "messages" in ex

    def test_two_messages(self):
        ex = make_example("Instruction.", "Response.")
        assert len(ex["messages"]) == 2

    def test_user_role_first(self):
        ex = make_example("Instruction.", "Response.")
        assert ex["messages"][0]["role"] == "user"

    def test_assistant_role_second(self):
        ex = make_example("Instruction.", "Response.")
        assert ex["messages"][1]["role"] == "assistant"

    def test_instruction_is_user_content(self):
        ex = make_example("My instruction.", "My response.")
        assert ex["messages"][0]["content"] == "My instruction."

    def test_passage_is_assistant_content(self):
        ex = make_example("My instruction.", "My response.")
        assert ex["messages"][1]["content"] == "My response."

    def test_json_serializable(self):
        ex = make_example("Instr.", "Resp.")
        serialized = json.dumps(ex)
        assert json.loads(serialized) == ex

    def test_unicode_preserved(self):
        ex = make_example("Instr.", "Élève à l'école — naïve café.")
        assert "Élève" in ex["messages"][1]["content"]


# ---------------------------------------------------------------------------
# collect_examples
# ---------------------------------------------------------------------------

class TestCollectExamples:

    def test_returns_example_with_long_passage(self):
        q = make_q(passage=long_passage())
        result = collect_examples([q], min_passage=100)
        assert len(result) == 1

    def test_filters_passage_below_min_length(self):
        q = make_q(passage="Too short.")
        result = collect_examples([q], min_passage=100)
        assert len(result) == 0

    def test_filters_empty_passage(self):
        q = make_q(passage="")
        result = collect_examples([q], min_passage=100)
        assert len(result) == 0

    def test_filters_none_passage(self):
        q = make_q(passage=None)
        result = collect_examples([q], min_passage=100)
        assert len(result) == 0

    def test_missing_passage_key(self):
        q = {k: v for k, v in FULL_QUESTION.items() if k != "passage"}
        result = collect_examples([q], min_passage=100)
        assert len(result) == 0

    def test_deduplicates_same_passage_same_book(self):
        passage = long_passage()
        qs = [
            make_q(passage=passage, source_htid="book_A"),
            make_q(passage=passage, source_htid="book_A"),
        ]
        result = collect_examples(qs, min_passage=100)
        assert len(result) == 1

    def test_keeps_same_passage_different_books(self):
        passage = long_passage()
        qs = [
            make_q(passage=passage, source_htid="book_A"),
            make_q(passage=passage, source_htid="book_B"),
        ]
        result = collect_examples(qs, min_passage=100)
        assert len(result) == 2

    def test_keeps_different_passages_same_book(self):
        qs = [
            make_q(passage=long_passage() + "AAA", source_htid="book_A"),
            make_q(passage=long_passage() + "BBB", source_htid="book_A"),
        ]
        result = collect_examples(qs, min_passage=100)
        assert len(result) == 2

    def test_example_has_instruction_key(self):
        q = make_q(passage=long_passage())
        result = collect_examples([q], min_passage=100)
        assert "instruction" in result[0]

    def test_example_has_passage_key(self):
        q = make_q(passage=long_passage())
        result = collect_examples([q], min_passage=100)
        assert "passage" in result[0]

    def test_example_has_htid_key(self):
        q = make_q(passage=long_passage())
        result = collect_examples([q], min_passage=100)
        assert "htid" in result[0]

    def test_htid_matches_source(self):
        q = make_q(passage=long_passage(), source_htid="my_book_123")
        result = collect_examples([q], min_passage=100)
        assert result[0]["htid"] == "my_book_123"

    def test_instruction_contains_metadata(self):
        q = make_q(passage=long_passage())
        result = collect_examples([q], min_passage=100)
        instr = result[0]["instruction"]
        assert "The Indian Eye on English Life" in instr
        assert "1891" in instr

    def test_passage_stripped(self):
        q = make_q(passage="   " + long_passage() + "   ")
        result = collect_examples([q], min_passage=100)
        assert not result[0]["passage"].startswith(" ")
        assert not result[0]["passage"].endswith(" ")

    def test_min_passage_exact_boundary(self):
        # passage of exactly min_passage chars should be INCLUDED
        passage = "A" * 100
        q = make_q(passage=passage)
        result = collect_examples([q], min_passage=100)
        assert len(result) == 1

    def test_min_passage_one_below_boundary(self):
        passage = "A" * 99
        q = make_q(passage=passage)
        result = collect_examples([q], min_passage=100)
        assert len(result) == 0

    def test_empty_input(self):
        result = collect_examples([], min_passage=100)
        assert result == []

    def test_missing_htid_uses_unknown(self):
        q = make_q(passage=long_passage())
        del q["source_htid"]
        result = collect_examples([q], min_passage=100)
        assert result[0]["htid"] == "unknown"


# ---------------------------------------------------------------------------
# split_by_book
# ---------------------------------------------------------------------------

def _make_example_set(n_books: int, per_book: int = 3) -> list:
    """Create a simple list of examples spread across n_books."""
    examples = []
    for i in range(n_books):
        htid = f"book_{i:04d}"
        for j in range(per_book):
            examples.append({
                "instruction": f"Instr {i}-{j}",
                "passage": f"Passage {i}-{j}",
                "htid": htid,
                "reasoning_type": "phrase_cloze",
            })
    return examples


class TestSplitByBook:

    def test_no_leakage_train_val(self):
        examples = _make_example_set(30)
        train, val, test = split_by_book(examples, seed=42)
        train_books = {ex["htid"] for ex in train}
        val_books = {ex["htid"] for ex in val}
        assert train_books.isdisjoint(val_books)

    def test_no_leakage_train_test(self):
        examples = _make_example_set(30)
        train, val, test = split_by_book(examples, seed=42)
        train_books = {ex["htid"] for ex in train}
        test_books = {ex["htid"] for ex in test}
        assert train_books.isdisjoint(test_books)

    def test_no_leakage_val_test(self):
        examples = _make_example_set(30)
        train, val, test = split_by_book(examples, seed=42)
        val_books = {ex["htid"] for ex in val}
        test_books = {ex["htid"] for ex in test}
        assert val_books.isdisjoint(test_books)

    def test_all_examples_accounted_for(self):
        examples = _make_example_set(30)
        train, val, test = split_by_book(examples, seed=42)
        assert len(train) + len(val) + len(test) == len(examples)

    def test_train_is_largest_split(self):
        examples = _make_example_set(30)
        train, val, test = split_by_book(examples, seed=42)
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_approximate_train_fraction(self):
        """Train should be at least 60% of total examples."""
        examples = _make_example_set(50)
        train, val, test = split_by_book(examples, seed=42)
        assert len(train) >= len(examples) * 0.60

    def test_deterministic_with_same_seed(self):
        examples = _make_example_set(30)
        train1, val1, test1 = split_by_book(examples, seed=7)
        train2, val2, test2 = split_by_book(examples, seed=7)
        assert [e["htid"] for e in train1] == [e["htid"] for e in train2]
        assert [e["htid"] for e in val1] == [e["htid"] for e in val2]
        assert [e["htid"] for e in test1] == [e["htid"] for e in test2]

    def test_different_seeds_produce_different_splits(self):
        examples = _make_example_set(30)
        train1, _, _ = split_by_book(examples, seed=1)
        train2, _, _ = split_by_book(examples, seed=99)
        books1 = frozenset(ex["htid"] for ex in train1)
        books2 = frozenset(ex["htid"] for ex in train2)
        # With 30 books two randomly-chosen orderings very likely differ
        assert books1 != books2

    def test_val_and_test_are_nonempty_with_many_books(self):
        examples = _make_example_set(20)
        train, val, test = split_by_book(examples, seed=42)
        assert len(val) > 0
        assert len(test) > 0

    def test_small_dataset_three_books(self):
        """3 books: each split gets 1 book."""
        examples = _make_example_set(3, per_book=2)
        train, val, test = split_by_book(examples, seed=42)
        assert len(train) + len(val) + len(test) == len(examples)
        # Each split must have at least one book
        assert len(train) >= 2   # 1 book × 2 examples
        assert len(val) >= 2
        assert len(test) >= 2

    def test_single_book_goes_to_train(self):
        """With 1 book and minimum splits, all examples must land somewhere."""
        examples = _make_example_set(1, per_book=5)
        train, val, test = split_by_book(examples, seed=42)
        total = len(train) + len(val) + len(test)
        assert total == len(examples)

    def test_returns_three_lists(self):
        examples = _make_example_set(10)
        result = split_by_book(examples, seed=42)
        assert len(result) == 3
        train, val, test = result
        assert isinstance(train, list)
        assert isinstance(val, list)
        assert isinstance(test, list)
