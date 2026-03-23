"""
create_metadata_conditions.py

Produces two ablation versions of the ChronoLogic benchmark:
  B — benchmark_no_metadata.jsonl   : neutral placeholder replaces metadata_frame
  C — benchmark_shuffled_metadata.jsonl : metadata_frame shuffled within frame_type groups
      (no question receives its own original metadata_frame)

Source: chronologic/booksample/chronologic_en_0.1.jsonl  (709 questions)
Output: chronologic/evalcode/  (same directory as this script)
"""

import json
import os
import random
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(SCRIPT_DIR, "..", "booksample", "chronologic_en_0.1.jsonl")
OUT_NO_META = os.path.join(SCRIPT_DIR, "benchmark_no_metadata.jsonl")
OUT_SHUFFLED = os.path.join(SCRIPT_DIR, "benchmark_shuffled_metadata.jsonl")

NEUTRAL_FRAME = "The following is a question about a text."

random.seed(42)


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(questions, path):
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def make_no_metadata(questions):
    out = []
    for q in questions:
        q2 = dict(q)
        q2["metadata_frame"] = NEUTRAL_FRAME
        out.append(q2)
    return out


def derange(indices):
    """
    Return a derangement of `indices` (list of ints): a permutation where
    no element stays in its original position.  Uses random shuffles;
    converges in O(1) expected tries (probability ~1/e per trial).
    """
    n = len(indices)
    if n == 1:
        return list(indices)   # can't derange — return as-is
    perm = list(range(n))
    while True:
        random.shuffle(perm)
        if all(perm[j] != j for j in range(n)):
            return perm


def shuffle_within_groups(questions):
    """
    For each frame_type group, permute metadata_frame values so that
    no question ends up with the frame that was originally at its own
    index within the group (standard derangement by position).

    Shuffling within frame_type ensures a cloze question always gets
    a cloze-style frame (passage context), isolating metadata content
    from metadata format.
    """
    # group indices by frame_type
    groups = defaultdict(list)
    for i, q in enumerate(questions):
        groups[q["frame_type"]].append(i)

    shuffled_frames = {}
    for ft, idxs in groups.items():
        original_frames = [questions[i]["metadata_frame"] for i in idxs]
        perm = derange(idxs)   # perm[j] = which slot's frame goes to position j
        for j, global_idx in enumerate(idxs):
            shuffled_frames[global_idx] = original_frames[perm[j]]

    out = []
    for i, q in enumerate(questions):
        q2 = dict(q)
        q2["metadata_frame"] = shuffled_frames[i]
        out.append(q2)
    return out


def main():
    questions = load_questions(SOURCE)
    print(f"Loaded {len(questions)} questions from {os.path.basename(SOURCE)}")

    # ── Condition B: no metadata ──────────────────────────────────────────────
    no_meta = make_no_metadata(questions)
    write_jsonl(no_meta, OUT_NO_META)
    print(f"\nWrote {OUT_NO_META}")

    # ── Condition C: shuffled metadata ────────────────────────────────────────
    shuffled = shuffle_within_groups(questions)
    write_jsonl(shuffled, OUT_SHUFFLED)
    print(f"Wrote {OUT_SHUFFLED}")

    # ── Summary ───────────────────────────────────────────────────────────────
    from collections import Counter
    ft_counts = Counter(q["frame_type"] for q in questions)
    print("\n── frame_type distribution (all conditions share same structure) ──")
    for ft, cnt in sorted(ft_counts.items()):
        print(f"  {ft}: {cnt}")

    # Verify derangement: every question should get a different frame
    changed = sum(
        1 for orig, shuf in zip(questions, shuffled)
        if orig["metadata_frame"] != shuf["metadata_frame"]
    )
    unchanged = len(questions) - changed
    print(f"\nShuffled condition: {changed}/{len(questions)} frames changed "
          f"({unchanged} string-identical matches — expected ~0 for well-separated groups)")


if __name__ == "__main__":
    main()
