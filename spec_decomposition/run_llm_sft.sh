#!/bin/bash
# LoRA SFT of the ExeDec roles on the DeepCoder-Pythonic (v4) DSL.
#
# Usage:
#   ./run_llm_sft.sh                # both roles, sequentially
#   ./run_llm_sft.sh synthesizer    # one role, so the two can run on two GPUs
#
# Each role is a single-GPU job, ~3.6 h per epoch (measured). Running them as
# two jobs halves wall clock; request --gres=gpu:1 per job, because with two
# GPUs visible HF Trainer silently falls back to nn.DataParallel.
#
# Build the data first:
#   python -m spec_decomposition.llm_finetune.build_sft_data \
#     --generalization_task=NONE --split=train --max_tasks=20000
#   python -m spec_decomposition.llm_finetune.build_sft_data \
#     --generalization_task=NONE --split=valid --max_tasks=500
set -euo pipefail

base_model=meta-llama/Llama-3.1-8B
generalization_task=NONE
output_dir=${RESULTS_DIR:-./results}/llm_sft

roles=${1:-"synthesizer decomposer"}

for role in ${roles}; do
  echo "=== SFT: ${role} on ${generalization_task} ==="
  # batch_size x grad_accum = 16 is the effective batch. Note batch_size is NOT
  # a speed knob here: measured 0.573 s/record at 4 vs 0.613 at 16, because the
  # GPU is already compute-bound on the sequence dimension and a bigger batch
  # only adds padding. It was left at 4 (23.7 GB of 93 GB) for that reason --
  # low VRAM use is the correct outcome, not something to fix.
  python -m spec_decomposition.llm_finetune.sft_train \
    --role=${role} \
    --base_model=${base_model} \
    --generalization_task=${generalization_task} \
    --output_dir=${output_dir} \
    --num_epochs=2 \
    --learning_rate=1e-4 \
    --batch_size=4 \
    --grad_accum=4 \
    --max_seq_len=4096 \
    --save_steps=1000 \
    --save_total_limit=10 \
    "${@:2}"
done
