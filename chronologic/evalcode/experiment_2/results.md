Here’s a clean, data-driven read of your experiment.

---

# 1) Does full metadata (A) consistently beat no metadata (B)?

**Short answer: yes, almost everywhere, but not uniformly.**

### Overall accuracy

* **Qwen2.5-7B:** A = 0.202 vs B = 0.158 → **clear gain**
* **Qwen2.5-14B:** A = 0.209 vs B = 0.178 → **clear gain**
* **Mistral-7B:** A = 0.162 vs B = 0.145 → **small gain**

So:

* All models benefit from metadata **on average**
* The effect size scales with model capability (bigger model → stronger gain)

### Takeaway

Metadata is **net helpful**, but it is **not universally critical**. Some tasks barely change.

---

# 2) Does shuffled metadata (C) hurt vs full metadata (A)?

This is where things get interesting.

### Overall comparison (A vs C)

* **Qwen2.5-7B:** A = C (0.202) → **no effect**
* **Qwen2.5-14B:** C (0.224) > A (0.209) → **improves with wrong metadata**
* **Mistral-7B:** A (0.162) > C (0.157) → **slight degradation**

### Interpretation

* **Qwen2.5-7B:** largely **ignores metadata**
* **Qwen2.5-14B:** **uses metadata, but not reliably grounded**

  * wrong metadata can still help → suggests **heuristic pattern matching**
* **Mistral-7B:** mildly sensitive, but not strongly

### Key insight

> Metadata is not being used as “truth”. It’s being used as a **soft prior or stylistic signal**.

This is quite counterintuitive: **incorrect metadata does not reliably hurt**, and can even help.

---

# 3) Which question types benefit most from metadata?

## By frame type

Metadata helps **most in:**

* **book_context**

  * Largest gains across all models
  * Example: +0.08 (Qwen2.5-7B)

Moderate gains:

* world_context
* passage_context (smallest effect)

### Why?

* Book-level framing aligns strongly with metadata fields (author, genre, era)
* Passage-level questions rely more on local text

---

## By reasoning type

### Biggest gains from metadata:

* **topic_sentence** (huge for smaller models, +0.16)
* **constrained_generation**
* **inference**
* **character_modeling**

These tasks benefit from:

* stylistic priors
* author/genre expectations
* narrative/world assumptions

---

### Minimal or mixed impact:

* sentence_cloze
* phrase_cloze

These are more **local language tasks**, so metadata matters less.

---

# 4) Which are hurt most by removing metadata?

Look at A − B drops:

### Most impacted:

* **book_context**
* **topic_sentence**
* **constrained_generation**
* **inference**

### Least impacted:

* **passage_context**
* **cloze tasks**

### Surprising negative cases:

* Qwen2.5-14B shows **negative impact** (metadata hurts) for:

  * knowledge
  * phrase_cloze

This suggests:

* metadata can **mislead factual recall**
* model may overfit to priors instead of retrieving knowledge

---

# 5) Differences between models

## Qwen2.5-14B (most interesting)

* **Most sensitive to metadata**
* Gains strongly from A vs B
* But also **benefits from wrong metadata (C > A)**

 Interpretation:

* Uses metadata actively
* But not in a grounded way
* Likely treating metadata as **semantic hints**, not constraints

---

## Qwen2.5-7B

* Gains from metadata
* But **C = A exactly (overall)**

 Interpretation:

* Either:

  * ignores metadata often, or
  * uses it only weakly

---

## Mistral-7B

* Small improvements from metadata
* Slight drop with shuffled metadata

 Interpretation:

* More **conservative**
* Less prone to being misled
* But also **extracts less benefit**

---

# 6) Most surprising findings

### 1. Wrong metadata can HELP

* Qwen2.5-14B performs **better with shuffled metadata than correct metadata**

This strongly suggests:

* metadata acts as a **regularizer or prompt enrichment**
* not as a factual grounding signal

---

### 2. Smaller model ignores metadata noise

* Qwen2.5-7B: identical A and C

Implies:

* either robustness or under-utilization

---

### 3. Metadata hurts factual tasks sometimes

* “knowledge” category drops for Qwen2.5-14B

This is important:

> Metadata can bias the model away from correct answers when the task requires **objective recall**

---

### 4. Metadata matters most at higher abstraction levels

* High gains in:

  * topic_sentence
  * inference
* Low gains in:

  * cloze tasks

So:

> Metadata helps when the model must **interpret**, not just **read**

---

# Final synthesis

* Metadata is **useful but not trusted**
* Models treat it as a **soft contextual prior**
* Larger models:

  * use it more
  * but are also more **easily misled**
* Smaller models:

  * benefit less
  * are more stable to noise

---

If you want, I can:

* quantify statistical significance (bootstrap / CI)
* visualize A vs B vs C gaps
* or help turn this into a strong paper-ready result section
