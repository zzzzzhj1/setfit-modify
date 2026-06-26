#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
mkdir -p "$HF_HOME"
mkdir -p logs

MODEL="sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE=2
NUM_ITERATIONS=5
NUM_EPOCHS=1
MAX_SEQ_LENGTH=128
OUTPUT_DIR="../../results_ag_news_safe"

echo "==== GPU CHECK ===="
nvidia-smi || true

echo "==== AG News safe benchmark, split process ===="
for SHOT in 4 8 16; do
  for SEED in 0 1 2 3 4; do
    echo "==== CASE: ag_news shot=${SHOT} seed=${SEED} ===="
    python run_fewshot.py \
      --model "$MODEL" \
      --datasets ag_news \
      --sample_sizes "$SHOT" \
      --seeds "$SEED" \
      --is_dev_set true \
      --batch_size "$BATCH_SIZE" \
      --num_iterations "$NUM_ITERATIONS" \
      --num_epochs "$NUM_EPOCHS" \
      --max_seq_length "$MAX_SEQ_LENGTH" \
      --output_dir "$OUTPUT_DIR" \
      2>&1 | tee "logs/ag_news_shot${SHOT}_seed${SEED}_safe.log"
    sleep 5
    nvidia-smi || true
  done
done

echo "==== SUMMARIZE AG News SAFE RESULTS ===="
python summarize_aug_results.py \
  --results_dir "$OUTPUT_DIR" \
  --output_dir "$OUTPUT_DIR"

echo "==== DONE ===="
