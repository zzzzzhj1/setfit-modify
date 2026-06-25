$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

python run_fewshot.py `
  --model sentence-transformers/all-MiniLM-L6-v2 `
  --datasets sst2 ag_news trec `
  --sample_sizes 4 8 16 `
  --batch_size 4 `
  --num_iterations 5 `
  --num_epochs 1 `
  --max_seq_length 128 `
  --override_results

python summarize_aug_results.py `
  --results_dir ..\..\results `
  --output_dir ..\..\results
