# Domain Fine-Tuning Experiment — H100 Runbook

Fine-tune Qwen2.5-7B-Instruct on 100 Project Gutenberg historical books
(1860-1925) and evaluate on the ChronoLogic benchmark (condA vs condC).

**Goal:** Test whether reading historical prose improves the model's ability
to prefer historically-accurate answers over anachronistic distractors.

---

## 0. Prerequisites

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## 1. Pull latest code

```bash
cd <repo-root>
git pull
ls chronologic/domain_ft/data/
# should show: train.jsonl  val.jsonl  test.jsonl  gutenberg_selected.csv
wc -l chronologic/domain_ft/data/train.jsonl   # expect ~28960
```

---

## 2. Activate environment

```bash
cd chronologic/metadata_ft
source ft-env/bin/activate
```

If environment doesn't exist, create it:
```bash
python -m venv ft-env
source ft-env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets peft trl bitsandbytes accelerate
```

---

## 3. Spot-check training data

```bash
python -c "
import json
ex = json.loads(open('data/train.jsonl').readline())
print(ex['text'][:400])
" 
```
Should be clean historical prose — no Gutenberg headers, no `[Illustration:]` markers.

---

## 4. Train

```bash
MODEL_PATH="\$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

# If not found:
find ~/.cache/huggingface -name config.json | grep -i qwen
```

```bash
cd chronologic/domain_ft
source ../metadata_ft/ft-env/bin/activate

python ../qwentuning/train_qlora.py \
    --model_path      "\$MODEL_PATH"          \
    --train_data      data/train.jsonl        \
    --val_data        data/val.jsonl          \
    --output_dir      checkpoints/run1        \
    --max_seq_length  2048   \
    --batch_size      8      \
    --grad_accum      2      \
    --learning_rate   2e-4   \
    --num_epochs      5      \
    --max_steps       -1     \
    --save_steps      200    \
    --lora_r          16     \
    --lora_alpha      32     \
    --lora_dropout    0.05   \
    --packing
```

Adapter saved to `checkpoints/run1/final_adapter/`.

**If OOM:** reduce `--batch_size 4`, increase `--grad_accum 4`.

---

## 5. Evaluate

```bash
cd chronologic
source metadata_ft/ft-env/bin/activate

MODEL_PATH="\$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
ADAPTER="domain_ft/checkpoints/run1/final_adapter"
```

### Condition A — correct metadata

```bash
python evalcode/benchmark_free_generation.py \
    "\$MODEL_PATH" \
    booksample/chronologic_en_0.2.jsonl \
    --lora-adapter "\$ADAPTER" \
    --output booksample/free_gen_qwen7b_domainft_run1_condA.json
```

### Condition C — permuted metadata

```bash
python evalcode/benchmark_free_generation.py \
    "\$MODEL_PATH" \
    booksample/chronologic_en_0.2_permuted.jsonl \
    --lora-adapter "\$ADAPTER" \
    --output booksample/free_gen_qwen7b_domainft_run1_condC.json
```

---

## 6. Push results

```bash
cd <repo-root>

git add \
    chronologic/booksample/free_gen_qwen7b_domainft_run1_condA.json \
    chronologic/booksample/free_gen_qwen7b_domainft_run1_condC.json \
    chronologic/domain_ft/checkpoints/run1/final_adapter/

git commit -m "Domain FT run1: 100 Gutenberg books, 5 epochs

- free_gen_qwen7b_domainft_run1_condA.json
- free_gen_qwen7b_domainft_run1_condC.json
- domain_ft/checkpoints/run1/final_adapter/"

git push
```

---

## Claude Code prompt

Open a terminal in the repo root and run `claude`, then paste:

```
Read chronologic/domain_ft/INSTRUCTIONS.md and follow steps 1-6 in order.
Ask me before starting step 4 (training) so I can confirm the model path.
Output files should be named free_gen_qwen7b_domainft_run1_condA.json and
free_gen_qwen7b_domainft_run1_condC.json.
```
