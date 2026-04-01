#!/bin/bash
# h100.slurm
#
# Metadata-instruction fine-tuning — ChronoLogic Experiment 4
# ─────────────────────────────────────────────────────────────
# Single-variable test: does training on (metadata → historical passage) pairs
# cause a small LLM to attend to metadata more effectively at inference time?
#
# Training format (instruction-tuning only, no mixed tasks):
#   User:      "Write a passage from '<title>', a <genre> published in <year>
#               by <author>, a <nationality> <profession>."
#   Assistant: <verbatim historical passage>
#
# This reuses qwentuning/train_qlora.py with metadata_ft/ data so the training
# infrastructure is the same tested pipeline — only the data differs.
#
# ─── Before submitting ────────────────────────────────────────────────────────
#
# 1. Edit #SBATCH --account and --partition below to match your allocation.
#
# 2. Set MODEL_PATH (search for "MODEL_PATH=" near the bottom of this file).
#    Options:
#      a) HuggingFace Hub ID  (will download on first run):
#           MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
#      b) Local HF cache (faster, no download):
#           MODEL_PATH="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<hash>"
#      c) Local directory:
#           MODEL_PATH="$SCRIPT_DIR/Qwen2.5-7B-Instruct"
#
# 3. Create the Python environment (run ONCE on a login node):
#
#      module load cudatoolkit/25.3_12.8   # adjust to your cluster's CUDA module
#      cd <path-to-metadata_ft>
#      python -m venv ft-env
#      source ft-env/bin/activate
#      pip install torch --index-url https://download.pytorch.org/whl/cu128
#      pip install transformers datasets peft trl bitsandbytes accelerate
#
# 4. Generate the training data (run ONCE from metadata_ft/):
#
#      source ft-env/bin/activate
#      python prepare_data.py
#      # Creates: data/train.jsonl, data/val.jsonl, data/test.jsonl
#
# 5. Submit:
#      sbatch h100.slurm
#
# ─── Monitoring ───────────────────────────────────────────────────────────────
#
#   squeue --me
#   tail -f metadata_ft_<jobid>.out
#
# ─── After training ───────────────────────────────────────────────────────────
#
#   The trained LoRA adapter will be at: checkpoints/run1/final_adapter/
#
#   To evaluate on the ChronoLogic benchmark, use the existing evaluation
#   pipeline in evalcode/benchmark_free_generation.py with the adapter loaded
#   via PEFT:
#
#      from peft import PeftModel
#      from transformers import AutoModelForCausalLM
#      base = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
#      model = PeftModel.from_pretrained(base, "checkpoints/run1/final_adapter")
#
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=metadata-ft
#SBATCH --output=metadata_ft_%j.out
#SBATCH --error=metadata_ft_%j.err
#SBATCH --partition=gpuH100x8       # adjust: gpuA100x4 / gpuH100x8 / etc.
#SBATCH --account=bdfx-delta-gpu    # adjust to your active allocation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80g                   # H100 SXM5 has 80 GB HBM; host RAM headroom
#SBATCH --time=04:00:00             # ~400 examples × 3 epochs is fast; adjust if needed

echo "══════════════════════════════════════════════"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "══════════════════════════════════════════════"

# ─── Environment ──────────────────────────────────────────────────────────────

module load cudatoolkit/25.3_12.8   # adjust to your cluster's CUDA module

SCRIPT_DIR="$SLURM_SUBMIT_DIR"     # directory where this .slurm file lives (metadata_ft/)
REPO_ROOT="$(dirname "$SCRIPT_DIR")"  # chronologic/

cd "$SCRIPT_DIR"

source ft-env/bin/activate

# Quick sanity check
python - <<'EOF'
import torch, sys
print(f"Python:  {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF

# ─── Model path ───────────────────────────────────────────────────────────────
# !! EDIT THIS LINE to point at your model weights !!

MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"   # or a local path — see header above

# ─── Data ─────────────────────────────────────────────────────────────────────

if [ ! -f "$SCRIPT_DIR/data/train.jsonl" ]; then
    echo ""
    echo "data/train.jsonl not found — running prepare_data.py ..."
    python prepare_data.py
    echo "Data preparation complete."
fi

TRAIN_ROWS=$(wc -l < "$SCRIPT_DIR/data/train.jsonl")
VAL_ROWS=$(wc -l   < "$SCRIPT_DIR/data/val.jsonl")
echo ""
echo "Training data: $TRAIN_ROWS examples"
echo "Val data:      $VAL_ROWS examples"

# ─── Training ─────────────────────────────────────────────────────────────────
#
# Hyperparameter rationale:
#   Dataset is small (~300-500 examples after dedup). To avoid overfitting:
#     - 3 epochs with cosine LR decay (warmup_steps=50 from train_qlora.py)
#     - lr=2e-4 (standard for QLoRA on instruction data)
#     - lora_r=16 with alpha=32 (2× scaling, mid-size adapter)
#   H100 80GB can handle batch_size=8 with 4096 seq length comfortably.
#   Effective batch = batch_size(8) × grad_accum(2) = 16.
#
# Adjust --max_steps or --num_epochs if you want to run longer/shorter.

echo ""
echo "Starting training ..."
python "$REPO_ROOT/qwentuning/train_qlora.py" \
    --model_path      "$MODEL_PATH"                      \
    --train_data      "$SCRIPT_DIR/data/train.jsonl"     \
    --val_data        "$SCRIPT_DIR/data/val.jsonl"       \
    --output_dir      "$SCRIPT_DIR/checkpoints/run1"     \
    --max_seq_length  4096    \
    --batch_size      8       \
    --grad_accum      2       \
    --learning_rate   2e-4    \
    --num_epochs      3       \
    --max_steps       -1      \
    --save_steps      50      \
    --lora_r          16      \
    --lora_alpha      32      \
    --lora_dropout    0.05

echo ""
echo "══════════════════════════════════════════════"
echo "Job completed at $(date)"
echo "Adapter saved to: $SCRIPT_DIR/checkpoints/run1/final_adapter"
echo "══════════════════════════════════════════════"
