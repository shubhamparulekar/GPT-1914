#!/bin/bash
cd /home/sdp7/GPT-1914/chronologic
source metadata_ft/ft-env/bin/activate

MODEL_PATH="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
ADAPTER="domain_ft/checkpoints/run1/final_adapter"

echo "[$(date)] Starting condA..."
python evalcode/benchmark_free_generation.py \
    "$MODEL_PATH" \
    booksample/chronologic_en_0.2.jsonl \
    --lora-adapter "$ADAPTER" \
    --device-map auto \
    --output booksample/free_gen_qwen7b_domainft_run1_condA.json

echo "[$(date)] condA done. Starting condC..."
python evalcode/benchmark_free_generation.py \
    "$MODEL_PATH" \
    booksample/chronologic_en_0.2_permuted.jsonl \
    --lora-adapter "$ADAPTER" \
    --device-map auto \
    --output booksample/free_gen_qwen7b_domainft_run1_condC.json

echo "[$(date)] condC done. All finished."
