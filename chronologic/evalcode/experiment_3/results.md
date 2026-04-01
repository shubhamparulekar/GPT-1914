# Experiment 3: Author Demography × Model Performance

**Date:** 2026-03-25
**Models:** Qwen2.5-7B-Instruct, Qwen2.5-32B-Instruct
**Questions:** 709 (chronologic_en_0.1.jsonl)
**Eval mode:** Probabilistic (Brier score) + correctness flag

> Note: GPT-5-mini results exist only as a markdown report (no JSON), so this experiment covers Qwen-7B vs Qwen-32B.

---

## Overall Model Performance

| Model      | Brier Score (↓) | Accuracy (↑) | Questions |
|------------|-----------------|--------------|-----------|
| Qwen-7B    | 0.1980 ±0.085   | 20.2%        | 709       |
| Qwen-32B   | 0.1857 ±0.080   | 24.0%        | 709       |

Qwen-32B outperforms Qwen-7B across the board (~4pp accuracy gain, ~0.012 Brier improvement).

---

## By Author Nationality

| Nationality    | Qwen-7B Brier | Qwen-7B Acc | Qwen-32B Brier | Qwen-32B Acc | N   |
|----------------|---------------|-------------|----------------|--------------|-----|
| American       | 0.1886        | 22.2%       | 0.1775         | 25.6%        | 418 |
| British Isles  | 0.2095        | 18.3%       | 0.1967         | 23.7%        | 219 |
| Unknown        | 0.2396        | 13.3%       | 0.2189         | 10.0%        | 30  |
| Canadian       | 0.2187        | 0.0%        | 0.1841         | 18.2%        | 11  |
| South African  | 0.2444        | 12.5%       | 0.2180         | 25.0%        | 8   |

**Finding:** Both models perform best on American authors, worst on South African / Unknown. British Isles authors consistently score ~4pp lower accuracy than American authors. The gap is consistent across model sizes — neither model has a differential nationality bias.

---

## By Author Profession

| Profession           | Qwen-7B Acc | Qwen-32B Acc | N   |
|----------------------|-------------|--------------|-----|
| mathematics teacher  | 50.0%       | 64.3%        | 14  |
| mathematician        | 45.5%       | 59.1%        | 22  |
| author               | 29.4%       | 35.3%        | 17  |
| literary scholar     | 28.6%       | 21.4%        | 14  |
| Unknown              | 21.2%       | 24.9%        | 321 |
| writer               | 8.1%        | 10.8%        | 37  |
| politician           | 9.1%        | 0.0%         | 11  |

**Finding:** Mathematical/technical authors score highest — likely because questions come from textbooks with more deterministic answers. Literary writers (novels, essays) score lowest. Qwen-32B gains most on mathematical content (+13.6pp for mathematicians vs +4pp for writers).

---

## By Source Genre

| Genre              | Qwen-7B Acc | Qwen-32B Acc | N   |
|--------------------|-------------|--------------|-----|
| examination manual | 50.0%       | 64.3%        | 14  |
| textbook           | 35.2%       | 42.6%        | 54  |
| book of poems      | 23.1%       | 30.8%        | 13  |
| encyclopedia       | 25.0%       | 30.6%        | 144 |
| novel              | 19.7%       | 19.7%        | 61  |
| history            | 9.5%        | 14.3%        | 21  |
| debate             | 9.1%        | 0.0%         | 11  |
| book of essays     | 6.7%        | 6.7%         | 15  |

**Finding:** Textbook/examination material is ~2× easier than fiction/essays. Novels are particularly hard (19.7% for both models — no scale gain). Debates score near chance.

---

## By Publication Date

| Era       | Qwen-7B Brier | Qwen-7B Acc | Qwen-32B Brier | Qwen-32B Acc | N   |
|-----------|---------------|-------------|----------------|--------------|-----|
| 1875-1899 | 0.1920        | 24.7%       | 0.1832         | 28.1%        | 417 |
| 1900-1924 | 0.2066        | 13.8%       | 0.1891         | 18.2%        | 291 |

**Finding:** Older texts (1875-1899) are easier than later ones (1900-1924) by ~10pp. This is unexpected — one hypothesis is that older canonical texts appear more in pretraining data.

---

## By Reasoning Type

| Reasoning Type        | Qwen-7B Acc | Qwen-32B Acc | N   |
|-----------------------|-------------|--------------|-----|
| inference             | 41.9%       | 47.3%        | 74  |
| knowledge             | 35.3%       | 44.7%        | 85  |
| topic_sentence        | 32.0%       | 16.0%        | 25  |
| constrained_generation| 22.1%       | 26.3%        | 95  |
| character_modeling    | 18.7%       | 22.0%        | 91  |
| phrase_cloze          | 12.2%       | 15.2%        | 164 |
| sentence_cloze        | 8.3%        | 15.0%        | 120 |
| abstention            | 10.9%       | 9.1%         | 55  |

**Finding:** Both models handle inference and knowledge questions best. Cloze tasks (phrase/sentence completion) are much harder — models struggle to reproduce exact period-appropriate language. Qwen-32B regresses on topic_sentence (−16pp) suggesting sensitivity to that format.

---

## Output Files

| File | Description |
|------|-------------|
| [analyze_author_demography.py](analyze_author_demography.py) | Main analysis script |
| [experiment3_results.csv](experiment3_results.csv) | Aggregated stats by dimension × group |
| [experiment3_by_reasoning_type.csv](experiment3_by_reasoning_type.csv) | Cross-breakdown: dimension × reasoning type |
| [experiment3_plots/](experiment3_plots/) | PNG bar charts (requires matplotlib) |

## Adding More Models

To add a model (e.g. GPT-5-mini once a JSON result exists), add it to `EVAL_FILES` in the script:

```python
EVAL_FILES = {
    "Qwen-7B":  "...",
    "Qwen-32B": "...",
    "GPT-5-mini": os.path.join(BOOKSAMPLE, "eval_results_mcq_gpt-5-mini_....json"),
}
```

The script handles any number of models automatically.
