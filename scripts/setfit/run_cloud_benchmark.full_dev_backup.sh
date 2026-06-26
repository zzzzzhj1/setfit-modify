#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL="sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE=8
NUM_ITERATIONS=10
NUM_EPOCHS=1
MAX_SEQ_LENGTH=128

python run_fewshot.py \
  --model "$MODEL" \
  --sample_sizes 4 8 16 \
  --is_dev_set true \
  --batch_size "$BATCH_SIZE" \
  --num_iterations "$NUM_ITERATIONS" \
  --num_epochs "$NUM_EPOCHS" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --override_results

python summarize_aug_results.py \
  --results_dir ../../results \
  --output_dir ../../results

# Test benchmark, enable only after dev results are stable:
# python run_fewshot.py \
#   --model "$MODEL" \
#   --sample_sizes 4 8 16 \
#   --is_test_set true \
#   --batch_size "$BATCH_SIZE" \
#   --num_iterations "$NUM_ITERATIONS" \
#   --num_epochs "$NUM_EPOCHS" \
#   --max_seq_length "$MAX_SEQ_LENGTH" \
#   --override_results
