#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdfx-delta-gpu
#SBATCH --time=08:00:00
#SBATCH --job-name=qwen25-pairwise-eval
#SBATCH --error=qwen25-pairwise-eval-%j.err
#SBATCH --output=qwen25-pairwise-eval-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=64g
#SBATCH --constraint="scratch"

echo "Job starting on $(hostname)"
date

module reset
module load cudatoolkit/25.3_12.8
module list

source ../../qwentuning/qlora-env/bin/activate

# Move to evalcode directory so relative paths in benchmark_evaluation.py work
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR/.."

python benchmark_evaluation.py \
    Qwen/Qwen2.5-7B-Instruct \
    experiment_1/benchmark_pairwise.jsonl \
    --mcq \
    --output-dir experiment_1 \
    --verbose-report

echo "Job completed"
date
