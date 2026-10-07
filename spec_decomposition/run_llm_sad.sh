#!/bin/bash
# SAD training of the decomposer LLM against the frozen SFT synthesizer.
#
# Mirrors spec_decomposition/train.py: --rl_loss=grpo is the self-critical
# REINFORCE loss, --rl_loss=supervised is the CE-only control that shares the
# exact same code path. Run both to attribute any gain to the RL term -- because
# the grpo loss always includes the CE term, the control is what holds the
# number of extra CE steps fixed so the difference is the RL term alone.
#
# Usage:
#   ./run_llm_sad.sh            # both arms, sequentially
#   ./run_llm_sad.sh grpo       # one arm, so the two can run on two GPUs
#
# The starting adapters default to the finished SFT runs. Override them to start
# from a mid-training checkpoint instead, and set EXP_SUFFIX so the run does not
# overwrite the one started from the final adapters:
#   DECOMPOSER_ADAPTER=./results/llm_sft/decomposer_NONE/checkpoint-1000 \
#   SYNTHESIZER_ADAPTER=./results/llm_sft/synthesizer_NONE/checkpoint-1000 \
#   EXP_SUFFIX=_sft1000 ./run_llm_sad.sh grpo
set -euo pipefail

# At batch_size=16 the run sits at ~74% of a 93 GB H100, where allocator
# fragmentation starts to matter -- the probe needed this to avoid a 9 GB
# allocation failing with 4 GB free but 12 GB reserved-and-unallocated.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

base_model=meta-llama/Llama-3.1-8B
generalization_task=NONE
results_dir=${RESULTS_DIR:-./results}
sft_dir=${results_dir}/llm_sft

# Set either to the empty string to run that role off the untuned base model:
#   DECOMPOSER_ADAPTER= SYNTHESIZER_ADAPTER= EXP_SUFFIX=_base ./run_llm_sad.sh grpo
# `:-` (not `-`) so an explicitly empty value falls through to the default; use
# the `+` test below to tell "unset" from "set but empty".
decomposer_adapter=${DECOMPOSER_ADAPTER-${sft_dir}/decomposer_${generalization_task}/adapter}
synthesizer_adapter=${SYNTHESIZER_ADAPTER-${sft_dir}/synthesizer_${generalization_task}/adapter}
exp_suffix=${EXP_SUFFIX:-}

adapter_flags=()
[ -n "${decomposer_adapter}" ] && \
  adapter_flags+=(--decomposer_adapter="${decomposer_adapter}")
[ -n "${synthesizer_adapter}" ] && \
  adapter_flags+=(--synthesizer_adapter="${synthesizer_adapter}")

arms=${1:-"grpo supervised"}

# batch_size x grad_accum = 64 is the effective batch, and since sad_train centers
# the SCST advantage over the *whole* accumulation window, 64 is also the baseline's
# sample size. That matters: with adv_std ~ 7 a 4-sample baseline has a standard
# error of 3.5, half the signal it subtracts; at 64 it is 0.88. Raising grad_accum
# alone would not have helped when centering was per micro-batch.
# batch_size is 8 rather than 4 because generation is decode-bound and batches well,
# unlike the training forward/backward where batching did nothing.
for rl_loss in ${arms}; do
  echo "=== SAD: rl_loss=${rl_loss} on ${generalization_task} ==="
  echo "    decomposer:  ${decomposer_adapter:-<untuned base, fresh LoRA>}"
  echo "    synthesizer: ${synthesizer_adapter:-<untuned base>}"
  python -m spec_decomposition.llm_finetune.sad_train \
    --base_model=${base_model} \
    "${adapter_flags[@]}" \
    --exp_title=${rl_loss}_${generalization_task}${exp_suffix} \
    --generalization_task=${generalization_task} \
    --output_dir=${results_dir}/llm_sad \
    --rl_loss=${rl_loss} \
    --temperature=1.0 \
    --ent_coef=0.001 \
    --ce_weight=1.0 \
    --learning_rate=5e-6 \
    --batch_size=16 \
    --grad_accum=8 \
    --num_train_steps=250 \
    --log_steps=1 \
    --save_steps=25 \
    "${@:2}"
done
