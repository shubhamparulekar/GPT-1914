#!/bin/bash
#SBATCH --job-name=domainft_eval
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=domainft_eval_%j.out

MODEL_PATH="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
ADAPTER="chronologic/domain_ft/checkpoints/run1/final_adapter"

cd $HOME/GPT-1914/chronologic
source metadata_ft/ft-env/bin/activate

echo "=== condA ==="
python evalcode/benchmark_free_generation.py \
    "$MODEL_PATH" \
    booksample/chronologic_en_0.2.jsonl \
    --lora-adapter "$ADAPTER" \
    --output booksample/free_gen_qwen7b_domainft_run1_condA.json

echo "=== condC ==="
python evalcode/benchmark_free_generation.py \
    "$MODEL_PATH" \
    booksample/chronologic_en_0.2_permuted.jsonl \
    --lora-adapter "$ADAPTER" \
    --output booksample/free_gen_qwen7b_domainft_run1_condC.json

echo "=== Done ==="
