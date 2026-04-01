# Metadata Fine-Tuning Experiment — H100 Runbook

This is a self-contained guide for running Experiment 4 (metadata-instruction
fine-tuning) on an H100 GPU and pushing the results back to git.

**Goal:** Test whether fine-tuning Qwen2.5-7B-Instruct on
`(metadata → historical passage)` pairs causes it to use metadata more
effectively when answering ChronoLogic benchmark questions.

**You will run three things:**
1. `prepare_data.py` — generate training data from the benchmark (~2 minutes)
2. `train_qlora.py` — fine-tune the model with QLoRA (~30–60 minutes on H100)
3. `benchmark_free_generation.py` — evaluate the fine-tuned model on the benchmark
   twice (with original metadata and with permuted metadata)

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

Verify the new directory is there:
```bash
ls chronologic/metadata_ft/
# should show: prepare_data.py  h100.slurm  tests/  INSTRUCTIONS.md
```

---

## 2. Create the Python environment

**Run once.** If the environment already exists, skip to step 3.

```bash
cd chronologic/metadata_ft

# Create venv
python -m venv ft-env
source ft-env/bin/activate     # Windows: ft-env\Scripts\activate

# Install PyTorch matching your CUDA version
# Check CUDA version first:
nvcc --version                  # or: nvidia-smi | grep "CUDA Version"

# For CUDA 12.x (most H100 setups):
pip install torch --index-url https://download.pytorch.org/whl/cu128

# For CUDA 11.8 (older setup):
# pip install torch --index-url https://download.pytorch.org/whl/cu118

# Core training packages
pip install transformers datasets peft trl bitsandbytes accelerate

# Verify GPU is visible:
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

---

## 3. Generate training data

```bash
cd chronologic/metadata_ft
source ft-env/bin/activate

python prepare_data.py
```

Expected output:
```
Training examples (after dedup):  404
Unique source books:               76
Split: 340 train / 38 val / 26 test (by source book)
  Wrote 340 examples -> data/train.jsonl
  Wrote 38 examples -> data/val.jsonl
  Wrote 26 examples -> data/test.jsonl
```

Spot-check an example:
```bash
python -c "
import json
ex = json.loads(open('data/train.jsonl').readline())
print('USER:', ex['messages'][0]['content'])
print()
print('ASSISTANT:', ex['messages'][1]['content'][:300])
"
```

---

## 4. Run the fine-tuning

### Option A: If you have SLURM (HPC cluster)

Edit `h100.slurm`:
- Change `#SBATCH --account=bdfx-delta-gpu` to your account
- Change `#SBATCH --partition=gpuH100x8` to your partition
- Set `MODEL_PATH` to the model location (see below)

Then submit:
```bash
sbatch h100.slurm
squeue --me             # monitor job
tail -f metadata_ft_<JOBID>.out
```

### Option B: Direct GPU (VSCode terminal, no SLURM)

```bash
cd chronologic/metadata_ft
source ft-env/bin/activate

# Choose your model path — one of:
#   a) Download from HF (requires internet):
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

#   b) Already cached on this machine:
MODEL_PATH="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

#   c) Local directory:
MODEL_PATH="./Qwen2.5-7B-Instruct"   # adjust path as needed

python ../qwentuning/train_qlora.py \
    --model_path      "$MODEL_PATH"              \
    --train_data      data/train.jsonl           \
    --val_data        data/val.jsonl             \
    --output_dir      checkpoints/run1           \
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

The adapter will be saved to `checkpoints/run1/final_adapter/`.

**If you run out of VRAM**, reduce `--batch_size` to 4 (effective batch stays 16
with `--grad_accum 4`) or reduce `--max_seq_length` to 2048.

**Quick smoke test** (verifies the pipeline in ~5 minutes):
```bash
python prepare_data.py --smoke          # writes sample_train.jsonl
python ../qwentuning/train_qlora.py --smoke --model_path "$MODEL_PATH"
```

---

## 5. Evaluate the fine-tuned model on the benchmark

We run the benchmark **twice** — with correct metadata and with permuted metadata —
to measure whether fine-tuning made the model more sensitive to metadata content.

```bash
cd ..    # back to chronologic/
source metadata_ft/ft-env/bin/activate

# Make sure eval dependencies are installed:
pip install torch transformers peft accelerate    # already installed above
```

### 5a. Baseline (no adapter) — skip if already done in Experiment 1/3

```bash
python evalcode/benchmark_free_generation.py \
    "$MODEL_PATH" \
    booksample/chronologic_en_0.1.jsonl \
    --output booksample/free_gen_qwen7b_baseline.json
```

### 5b. Fine-tuned model — original metadata (Condition A)

```bash
ADAPTER_PATH="metadata_ft/checkpoints/run1/final_adapter"

python evalcode/benchmark_free_generation.py \
    "$MODEL_PATH" \
    booksample/chronologic_en_0.1.jsonl \
    --lora-adapter "$ADAPTER_PATH" \
    --output booksample/free_gen_qwen7b_metaft_condA.json
```

### 5c. Fine-tuned model — permuted metadata (Condition C)

First, generate the permuted benchmark if it doesn't exist yet:
```bash
python evalcode/experiment_2/create_metadata_conditions.py
# creates: evalcode/experiment_2/benchmark_shuffled_metadata.jsonl
```

Then run generation on the permuted version:
```bash
python evalcode/benchmark_free_generation.py \
    "$MODEL_PATH" \
    evalcode/experiment_2/benchmark_shuffled_metadata.jsonl \
    --lora-adapter "$ADAPTER_PATH" \
    --output booksample/free_gen_qwen7b_metaft_condC.json
```

**What these two runs tell you:**
- **Condition A** accuracy: how well the fine-tuned model answers with correct metadata
- **Condition C** accuracy: how well it answers with wrong metadata
- A large **A − C gap** means the model is actually using the metadata
- Compare both to the baseline to see if fine-tuning helped or hurt

---

## 6. Commit results and push

```bash
cd <repo-root>

# Add the output files
git add \
    chronologic/booksample/free_gen_qwen7b_metaft_condA.json \
    chronologic/booksample/free_gen_qwen7b_metaft_condC.json \
    chronologic/metadata_ft/checkpoints/run1/final_adapter/

# If you also have the baseline:
# git add chronologic/booksample/free_gen_qwen7b_baseline.json

git commit -m "Experiment 4 results: metadata instruction fine-tuning

- free_gen_qwen7b_metaft_condA.json  -- fine-tuned, correct metadata
- free_gen_qwen7b_metaft_condC.json  -- fine-tuned, permuted metadata
- checkpoints/run1/final_adapter/    -- LoRA adapter weights

Training: 340 examples, 3 epochs, Qwen2.5-7B-Instruct + QLoRA (r=16)"

git push
```

**Note on adapter size:** The LoRA adapter is small (~100 MB). If the repo has
a file-size policy or git-lfs is set up, follow those conventions. If it's too
large to push, push just the JSON result files and keep the adapter locally.

---

## 7. What to analyze back home

After pulling, the key files to look at are:

| File | What it shows |
|------|---------------|
| `free_gen_qwen7b_metaft_condA.json` | Fine-tuned answers with correct metadata |
| `free_gen_qwen7b_metaft_condC.json` | Fine-tuned answers with permuted metadata |
| (existing) `free_gen_gpt4.1_score.json` | GPT-4.1 scores for comparison |

Run scoring on the new files using the existing pipeline:
```bash
python bertclassify/free_gen_eval.py \
    booksample/free_gen_qwen7b_metaft_condA.json --auto-only

python bertclassify/free_gen_eval.py \
    booksample/free_gen_qwen7b_metaft_condC.json --auto-only
```

The key hypothesis (H3 from ExperimentPlan.md):
> Fine-tuning improves performance on phrase_cloze and sentence_cloze,
> with the gap increasing with answer_length.

A secondary check (H1):
> Metadata permutation still hurts performance (Condition A >> Condition C),
> but hopefully the gap is larger after fine-tuning than before.

---

## Troubleshooting

**CUDA out of memory:**
```bash
# Reduce batch size (effective batch = batch_size * grad_accum stays 16):
--batch_size 4 --grad_accum 4

# Or reduce sequence length:
--max_seq_length 2048
```

**Model not found / download fails:**
```bash
# Force download:
export HF_HUB_OFFLINE=0
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# Or find the cached version:
find ~/.cache/huggingface -name "config.json" | grep -i qwen
```

**bitsandbytes error on loading:**
```bash
# bitsandbytes sometimes needs to be reinstalled to match CUDA:
pip uninstall bitsandbytes -y
pip install bitsandbytes
python -c "import bitsandbytes; print('OK')"
```

**Generation looks wrong / no adapter effect:**
Make sure you're passing `--lora-adapter` to the same model ID used for training.
The adapter is model-specific: a Qwen adapter won't work on Mistral.
