#!/bin/bash
# End-to-end smoke test of the LLM fine-tuning pipeline on a tiny slice of data.
# Verifies that each stage runs, saves, and reloads before committing GPU-days.
set -euo pipefail

smoke_dir=${1:-./results/llm_smoke}
base_model=meta-llama/Llama-3.1-8B

echo "############ 1/4  SFT synthesizer ############"
python -m spec_decomposition.llm_finetune.sft_train \
  --role=synthesizer --base_model=${base_model} \
  --output_dir=${smoke_dir} \
  --max_train_records=600 --max_eval_records=100 \
  --num_epochs=1 --grad_accum=8 --eval_steps=30 --save_steps=1000 \
  --log_steps=5

echo "############ 2/4  SFT decomposer ############"
python -m spec_decomposition.llm_finetune.sft_train \
  --role=decomposer --base_model=${base_model} \
  --output_dir=${smoke_dir} \
  --max_train_records=600 --max_eval_records=100 \
  --num_epochs=1 --grad_accum=8 --eval_steps=30 --save_steps=1000 \
  --log_steps=5

echo "############ 3/4  reward sanity ############"
# Must pass before SAD training means anything.
python -m spec_decomposition.llm_finetune.reward_check \
  --base_model=${base_model} \
  --synthesizer_adapter=${smoke_dir}/synthesizer_NONE/adapter \
  --num_records=40

echo "############ 4/4  SAD (SCST) ############"
python -m spec_decomposition.llm_finetune.sad_train \
  --base_model=${base_model} \
  --decomposer_adapter=${smoke_dir}/decomposer_NONE/adapter \
  --synthesizer_adapter=${smoke_dir}/synthesizer_NONE/adapter \
  --output_dir=${smoke_dir}/sad \
  --max_train_records=400 \
  --num_train_steps=10 --batch_size=4 --grad_accum=1 \
  --log_steps=1 --save_steps=10

echo "############ ALL SMOKE TESTS PASSED ############"
