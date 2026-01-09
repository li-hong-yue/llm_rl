#!/bin/bash
#SBATCH --mem=300g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=20:00:00
#SBATCH --job-name=llm_rl
#SBATCH --account=bfyv-dtai-gh

#SBATCH --output=/u/hli48/slurm_logs/llm_rl_%j.out
#SBATCH --error=/u/hli48/slurm_logs/llm_rl_%j.err
#SBATCH --gpus-per-node=4

source /sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh
conda activate base

unset ROCR_VISIBLE_DEVICES


export WANDB_API_KEY="58c63245738680ba7ef05553b320c054b04fa049"
export PYTHONPATH=$HOME:$PYTHONPATH
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

cd /u/hli48/sft


NUM_GPUS=${1:-4}  # Default to 4 GPUs if not specified
CONFIG=${2:-config/default.yaml}  # Default config

echo "Running distributed training on $NUM_GPUS GPUs with config $CONFIG"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    scripts/train.py \
    --config $CONFIG

# Usage:
# bash scripts/run_distributed.sh 4 config/qwen_math.yaml  # 4 GPUs
# bash scripts/run_distributed.sh 2  # 2 GPUs with default config
