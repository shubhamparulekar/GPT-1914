

***

# Distractor Type Analysis: Qwen2.5-7B-Instruct

This analysis examines the performance of the **Qwen2.5-7B-Instruct** model on the **ChronoLogic** benchmark (English literature $1875\text{--}1924$), specifically focusing on how different distractor types influence model accuracy.

## 1. Distractor Type Ranking and Tiers
The model's performance suggests a clear hierarchy of difficulty among distractor types, categorized into three distinct tiers.

### Tier 1: High Difficulty / Adversarial ($0\% \text{--} 50\%$ Accuracy)
The most effective distractors involve manual intervention or LLM-generated anachronisms.
* **Lowest Accuracy:** `manual_anachronistic_distractor` ($0.0\%$, $n=1$) and hybrid manual/LLM distractors like `manual_anachronistic_metadataless_gpt-oss:20b` ($33.33\%$, $n=9$).
* **Strongest LLM Distractor:** `anachronistic_mistral-small:24b` ($38.71\%$, $n=62$) significantly outperforms its competitors in confusing the model.

### Tier 2: Moderate Difficulty / In-Corpus ($55\% \text{--} 70\%$ Accuracy)
Distractors pulled from the same context or generated with distortions.
* **Entity Confusion:** `same_character` ($57.45\%$) and `same_book` ($62.75\%$) show the model occasionally loses track of narrative details when presented with familiar names or settings.
* **Manual Anachronisms:** `anachronistic_manual` ($59.65\%$) (human-written) are slightly less effective than the best LLM-generated variants.

### Tier 3: Low Difficulty / Logic & Fluency ($80\% \text{--} 100\%$ Accuracy)
* **Logic Mastery:** The model is perfect at identifying `negation` ($100.0\%$, $n=78$) and `manual_anachronistic_distort2` ($100.0\%$).
* **Surface Safety:** It performs well against `manual_distractor` ($84.15\%$) and `manual_negation` ($80.77\%$).

## 2. Impact of Metadata on Distractor Performance
Anachronistic distractors generated **with metadata** (author, date, genre) consistently outperform those generated **without metadata**.

| Generator Model | Accuracy (Without Metadata) | Accuracy (With Metadata) |
| :--- | :--- | :--- |
| **Mistral-small:24b** | $49.02\%$ | **$38.71\%$** |
| **GPT-oss:20b** | $64.1\%$ | **$54.01\%$** |

**Observation:** Generator models effectively utilize historical metadata to "stylize" output. By knowing the era and author, they produce text that mimics period-specific surface linguistic features, making it harder for Qwen2.5-7B to distinguish fake text from genuine historical prose.

## 3. Vulnerable vs. Robust Question Categories
Robustness varies significantly based on the reasoning required for the question category.

* **Most Vulnerable:**
    * `handcrafted` ($36.11\%$ avg): Tailored questions that bypass standard model patterns.
    * `knowledge` ($42.08\%$ avg): Factual retrieval is easily disrupted by plausible-sounding distractors.
    * `cloze_concessiveclause` ($43.27\%$ avg): Weakest linguistic cloze test, indicating difficulty with complex logical relationships (e.g., "although").
* **Most Robust:**
    * `attribution` ($92.86\%$ avg): The model is highly skilled at identifying specific authorial styles.
    * `poetic_form` ($75.0\%$ avg) and `textbook` ($73.16\%$ avg): These rely on rigid structural or academic patterns harder to spoof.

## 4. Noteworthy Category × Distractor Combinations
Specific intersections reveal critical "blind spots":

* **Critical Weakness:** The `knowledge` category paired with `manual_anachronistic_distractor` resulted in **$0.0\%$ accuracy**. Similarly, `cloze_causalclause` vs `manual_anachronistic_metadataless_gpt-oss:20b` was **$0.0\%$**.
* **Surprising Robustness:** Despite `anachronistic_mistral-small` being strong overall, the model achieved **$100.0\%$ accuracy** against it in the `cloze_causalsentence` category.
* **Refusal Bias:** In the `refusal` category, the model had only **$51.69\%$ accuracy**, often being "tricked" into choosing a distractor that sounds like a safe refusal rather than the correct literary answer.

## 5. Reasoning Mechanism: Surface vs. Substance
The results suggest Qwen2.5-7B relies heavily on **surface fluency and temporal plausibility** rather than deep semantic verification.

1.  **Logical Strength:** The $100\%$ score on `negation` proves the model understands basic logical reversals and isn't fooled by simple "not" insertions.
2.  **Temporal Weakness:** The $\approx 11\%$ performance gap between metadata-aware and metadataless distractors shows the model is highly sensitive to "historical vibes." If a distractor uses era-appropriate vocabulary, the model often treats it as genuine.
3.  **The Adversarial Gap:** Human-edited LLM text (`manual_anachronistic_*`) yields the lowest scores ($\approx 33\%$), suggesting that while LLMs provide a "historical surface," human editors are needed to remove the semantic "tells" that the model otherwise detects.