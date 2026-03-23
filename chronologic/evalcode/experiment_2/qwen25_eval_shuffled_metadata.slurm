#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdfx-delta-gpu
#SBATCH --time=08:00:00
#SBATCH --job-name=qwen25-eval-shuffled
#SBATCH --error=qwen25-eval-shuffled-%j.err
#SBATCH --output=qwen25-eval-shuffled-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=64g
#SBATCH --constraint="scratch"

echo "Job starting on $(hostname)"
date

module reset
module load cudatoolkit/25.3_12.8
module list

source ../qwentuning/qlora-env/bin/activate

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
cd "$SCRIPT_DIR"

# Condition C: shuffled metadata frame (permuted within frame_type groups)
python benchmark_evaluation.py \
    Qwen/Qwen2.5-7B-Instruct \
    evalcode/benchmark_shuffled_metadata.jsonl \
    --full-eval \
    --verbose-report \
    --output-dir evalcode/results_shuffled_metadata

echo "Job completed"
date
