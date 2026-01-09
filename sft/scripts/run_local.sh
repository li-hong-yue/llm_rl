#!/bin/bash

export WANDB_API_KEY="58c63245738680ba7ef05553b320c054b04fa049"
export PYTHONPATH=$HOME:$PYTHONPATH

cd /u/hli48/sft


NUM_GPUS=${1:-2}  # Default to 2 GPUs if not specified
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
bash scripts/run_local.sh 2 config/math.yaml 