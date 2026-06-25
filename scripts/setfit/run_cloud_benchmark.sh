#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
mkdir -p "$HF_HOME"
mkdir -p logs

MODEL="sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE=8
NUM_ITERATIONS=20
NUM_EPOCHS=1
MAX_SEQ_LENGTH=128

echo "==== GPU CHECK ===="
nvidia-smi || true

echo "==== DEV BENCHMARK: sst2 ag_news trec ===="
python run_fewshot.py \
  --model "$MODEL" \
  --datasets sst2 ag_news trec \
  --sample_sizes 4 8 16 \
  --is_dev_set true \
  --batch_size "$BATCH_SIZE" \
  --num_iterations "$NUM_ITERATIONS" \
  --num_epochs "$NUM_EPOCHS" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --override_results

echo "==== SUMMARIZE RESULTS ===="
python summarize_aug_results.py \
  --results_dir ../../results \
  --output_dir ../../results

echo "==== DONE ===="
