# Metadata Fine-Tuning Experiment — H100 Runbook (Run 2)

This is a self-contained guide for running Experiment 4 (metadata-instruction
fine-tuning) on an H100 GPU and pushing the results back to git.

**Goal:** Test whether fine-tuning Qwen2.5-7B-Instruct on
`(metadata → historical passage)` pairs causes it to use metadata more
effectively when answering ChronoLogic benchmark questions.

**Background — Run 1 (completed, discarded):**
Run 1 trained on raw passages from chronologic_en_0.1.jsonl which contained
HathiTrust OCR artifacts (embedded page numbers, ALL-CAPS running headers,
numeric repetition runs). The model learned to reproduce these artifacts,
producing degenerate outputs like "126. 127. 128. 129..." and "168 168 168 168".
The training data has since been cleaned and the benchmark updated to v0.2.
Run 2 trains a fresh adapter on the clean data.

**You will run three things:**
1. `prepare_data.py` — confirm/regenerate clean training data from v0.2 benchmark
2. `train_qlora.py` — fine-tune with QLoRA, saving to `checkpoints/run2`
3. `benchmark_free_generation.py` — evaluate twice (correct metadata + permuted metadata)

---

## 0. Prerequisites

Check that you have a GPU:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## 1. Pull the latest code

```bash
cd <wherever-you-cloned-the-repo>
git pull
```

Verify the files are there:
```bash
ls chronologic/metadata_ft/
# should show: prepare_data.py  h100.slurm  tests/  INSTRUCTIONS.md  data/
ls chronologic/metadata_ft/data/
# should show: train.jsonl  val.jsonl  test.jsonl
```

---

## 2. Activate the Python environment

```bash
cd chronologic/metadata_ft
source ft-env/bin/activate     # Windows: ft-env\Scripts\activate
```

If the environment does not exist yet, create it:
```bash
python -m venv ft-env
source ft-env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets peft trl bitsandbytes accelerate
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

---

## 3. Confirm training data

The cleaned training data should already be present from the last git push.
Verify:
```bash
wc -l data/train.jsonl   # expect ~338
```

If the file is missing or empty, regenerate it:
```bash
python prepare_data.py
```

Expected output:
```
Training examples (after dedup):  402
Unique source books:               76
Split: 338 train / 38 val / 26 test (by source book)
  Wrote 338 examples -> data/train.jsonl
  Wrote 38 examples -> data/val.jsonl
  Wrote 26 examples -> data/test.jsonl
```

Spot-check an example to confirm no OCR artifacts:
```bash
python -c "
import json
ex = json.loads(open('data/train.jsonl').readline())
print('USER:', ex['messages'][0]['content'])
print()
print('ASSISTANT:', ex['messages'][1]['content'][:300])
"
```
The assistant text should be clean prose — no page numbers, no ALL-CAPS headers.

---

## 4. Run the fine-tuning

### Set the model path

```bash
# Most likely location (already cached from Run 1):
MODEL_PATH="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

# If that path does not exist, find it:
find ~/.cache/huggingface -name config.json | grep -i qwen

# Or download fresh from HuggingFace:
# MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
```

### Option A: Direct GPU (VSCode terminal, no SLURM)

```bash
cd chronologic/metadata_ft
source ft-env/bin/activate

python ../qwentuning/train_qlora.py \
    --model_path      "$MODEL_PATH"          \
    --train_data      data/train.jsonl       \
    --val_data        data/val.jsonl         \
    --output_dir      checkpoints/run2       \
    --max_seq_length  4096   \
    --batch_size      8      \
    --grad_accum      2      \
    --learning_rate   2e-4   \
    --num_epochs      3      \
    --max_steps       -1     \
    --save_steps      50     \
    --lora_r          16     \
    --lora_alpha      32     \
    --lora_dropout    0.05
```

The adapter will be saved to `checkpoints/run2/final_adapter/`.

**If you run out of VRAM**, reduce `--batch_size` to 4 and increase `--grad_accum` to 4.

### Option B: SLURM (HPC cluster)

Edit `h100.slurm`:
- Change `--output_dir` to `checkpoints/run2`
- Set `MODEL_PATH` to the path above

Then submit:
```bash
sbatch h100.slurm
squeue --me
tail -f metadata_ft_<JOBID>.out
```

---

## 5. Evaluate the fine-tuned model

Run the benchmark **twice** — correct metadata (Condition A) and permuted
metadata (Condition C) — to measure whether the model is actually using metadata.

```bash
cd ..   # back to chronologic/
source metadata_ft/ft-env/bin/activate

ADAPTER_PATH="metadata_ft/checkpoints/run2/final_adapter"
```

### 5a. Condition A — correct metadata

```bash
python evalcode/benchmark_free_generation.py \
    "$MODEL_PATH" \
    booksample/chronologic_en_0.2.jsonl \
    --lora-adapter "$ADAPTER_PATH" \
    --output booksample/free_gen_qwen7b_metaft_run2_condA.json
```

### 5b. Condition C — permuted metadata

```bash
python evalcode/benchmark_free_generation.py \
    "$MODEL_PATH" \
    booksample/chronologic_en_0.2_permuted.jsonl \
    --lora-adapter "$ADAPTER_PATH" \
    --output booksample/free_gen_qwen7b_metaft_run2_condC.json
```

**What these two runs tell you:**
- **Condition A** accuracy: fine-tuned model with correct metadata
- **Condition C** accuracy: fine-tuned model with wrong metadata
- A large **A - C gap** means the model is actually using metadata
- Compare to baseline (Run 1 condA without degenerate filtering) to see net effect

---

## 6. Commit results and push

```bash
cd ..   # repo root

git add \
    chronologic/booksample/free_gen_qwen7b_metaft_run2_condA.json \
    chronologic/booksample/free_gen_qwen7b_metaft_run2_condC.json \
    chronologic/metadata_ft/checkpoints/run2/final_adapter/

git commit -m "Experiment 4 run2: fine-tuning on cleaned v0.2 data

- free_gen_qwen7b_metaft_run2_condA.json  -- fine-tuned, correct metadata
- free_gen_qwen7b_metaft_run2_condC.json  -- fine-tuned, permuted metadata
- checkpoints/run2/final_adapter/         -- LoRA adapter weights

Training: 338 examples, 3 epochs, Qwen2.5-7B-Instruct + QLoRA (r=16)
Data: cleaned OCR artifacts, chronologic_en_0.2.jsonl"

git push
```

---

## 7. What to analyze back home

| File | What it shows |
|------|---------------|
| `free_gen_qwen7b_metaft_run2_condA.json` | Run 2 answers with correct metadata |
| `free_gen_qwen7b_metaft_run2_condC.json` | Run 2 answers with permuted metadata |
| `free_gen_qwen7b_metaft_condA.json` | Run 1 answers (degenerate, for reference only) |

Score the new files:
```bash
python bertclassify/free_gen_eval.py \
    booksample/free_gen_qwen7b_metaft_run2_condA.json --auto-only

python bertclassify/free_gen_eval.py \
    booksample/free_gen_qwen7b_metaft_run2_condC.json --auto-only
```

Key hypothesis (H3): fine-tuning improves phrase_cloze and sentence_cloze,
with the gap increasing with answer_length.

Secondary check (H1): Condition A >> Condition C accuracy gap is larger after
fine-tuning than in the untuned baseline.

---

## Troubleshooting

**CUDA out of memory:**
```bash
--batch_size 4 --grad_accum 4   # effective batch stays 16
# or:
--max_seq_length 2048
```

**Model not found:**
```bash
find ~/.cache/huggingface -name config.json | grep -i qwen
# or force download:
export HF_HUB_OFFLINE=0
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

**bitsandbytes error:**
```bash
pip uninstall bitsandbytes -y && pip install bitsandbytes
python -c "import bitsandbytes; print('OK')"
```

**Generation looks wrong / no adapter effect:**
Make sure `--lora-adapter` points at `checkpoints/run2/final_adapter`, not run1.

---

## Using Claude Code

Open a terminal in `chronologic/metadata_ft/` and run `claude`, then paste:

```
Read INSTRUCTIONS.md. We are doing Run 2 of the metadata fine-tuning experiment.
Run 1 is already done but produced degenerate outputs (OCR artifacts in training data).
The cleaned data and updated benchmark (v0.2) are already committed.

Follow steps 1-6 in INSTRUCTIONS.md in order. Ask me before starting the training
step (step 4) so I can confirm the model path is correct.
```
