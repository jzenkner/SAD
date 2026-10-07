#!/bin/bash
# Evaluates fine-tuned decomposer/synthesizer adapters on all six DeepCoder
# generalization splits, through the same harness the Ollama runs use.
#
# Set stage= to pick what is evaluated:
#   base  - untuned Llama-3.1-8B with the two-stage split. This is the ablation
#           that separates the effect of fine-tuning from the effect of turning
#           one generation into two; run it before drawing any conclusions.
#   sft   - both SFT adapters
#   sad   - SAD decomposer + SFT synthesizer
#
# The *_sft1000 stages evaluate the SAD runs started from SFT checkpoint-1000,
# against the sft_ckpt1000 baseline. All three hold the synthesizer at
# checkpoint-1000 -- the frozen reward model those SAD runs trained against --
# so the decomposer is the only thing that differs between the numbers:
#   sft_ckpt1000            - the baseline the SAD arms have to beat
#   sad_grpo_sft1000        - SAD with the self-critical REINFORCE loss
#   sad_supervised_sft1000  - the CE-only control, same number of CE steps
#
# The *_basedec and *_base stages evaluate the "does SAD work without SFT
# first" runs, whose decomposer started from a fresh LoRA on the untuned base.
# Each holds the synthesizer at the frozen reward model that run trained
# against, so the decomposer adapter is the only thing that differs:
#   sad_grpo_basedec / sad_supervised_basedec  - SFT synthesizer, as in `sft`
#   sad_grpo_base / sad_supervised_base        - untuned synthesizer, as in `base`
# Read the first pair against `base` too: the untuned decomposer paired with the
# SFT synthesizer has not been run, so `base` is the closest floor there is.
set -euo pipefail

stage=${1:-sft}
base_model=meta-llama/Llama-3.1-8B
generalization_task=NONE   # the split the adapters were trained on
results_dir=${RESULTS_DIR:-./results}
sft_dir=${results_dir}/llm_sft
sad_dir=${results_dir}/llm_sad

adapter_flags=()
case "${stage}" in
  base)
    ;;
  sft)
    adapter_flags=(
      --decomposer_adapter=${sft_dir}/decomposer_${generalization_task}/adapter
      --synthesizer_adapter=${sft_dir}/synthesizer_${generalization_task}/adapter
    )
    ;;
  sad)
    # sad_train saves through PEFT's named-adapter path, i.e. <run>/decomposer.
    adapter_flags=(
      --decomposer_adapter=${sad_dir}/grpo_${generalization_task}/decomposer
      --synthesizer_adapter=${sft_dir}/synthesizer_${generalization_task}/adapter
    )
    ;;
  sft_ckpt1000)
    adapter_flags=(
      --decomposer_adapter=${sft_dir}/decomposer_${generalization_task}/checkpoint-1000
      --synthesizer_adapter=${sft_dir}/synthesizer_${generalization_task}/checkpoint-1000
    )
    ;;
  sad_grpo_sft1000)
    adapter_flags=(
      --decomposer_adapter=${sad_dir}/grpo_${generalization_task}_sft1000/decomposer
      --synthesizer_adapter=${sft_dir}/synthesizer_${generalization_task}/checkpoint-1000
    )
    ;;
  sad_supervised_sft1000)
    adapter_flags=(
      --decomposer_adapter=${sad_dir}/supervised_${generalization_task}_sft1000/decomposer
      --synthesizer_adapter=${sft_dir}/synthesizer_${generalization_task}/checkpoint-1000
    )
    ;;
  sad_grpo_basedec)
    adapter_flags=(
      --decomposer_adapter=${sad_dir}/grpo_${generalization_task}_basedec/decomposer
      --synthesizer_adapter=${sft_dir}/synthesizer_${generalization_task}/adapter
    )
    ;;
  sad_supervised_basedec)
    adapter_flags=(
      --decomposer_adapter=${sad_dir}/supervised_${generalization_task}_basedec/decomposer
      --synthesizer_adapter=${sft_dir}/synthesizer_${generalization_task}/adapter
    )
    ;;
  sad_grpo_base)
    # No --synthesizer_adapter: the untuned base is the reward model this run
    # trained against, so it is also the synthesizer it has to be scored with.
    adapter_flags=(
      --decomposer_adapter=${sad_dir}/grpo_${generalization_task}_base/decomposer
    )
    ;;
  sad_supervised_base)
    adapter_flags=(
      --decomposer_adapter=${sad_dir}/supervised_${generalization_task}_base/decomposer
    )
    ;;
  *)
    echo "Unknown stage: ${stage} (expected base, sft, sad, sft_ckpt1000, " \
         "sad_grpo_sft1000, sad_supervised_sft1000, sad_grpo_basedec, " \
         "sad_supervised_basedec, sad_grpo_base or sad_supervised_base)" >&2
    exit 1
    ;;
esac

# The LLM cache keys only on the prompt text and temperature, so every
# checkpoint needs its own directory or it will serve another model's samples.
cache_dir=./llm_cache/llama3.1-8b_exedec_${stage}_${generalization_task}

python -m spec_decomposition.run_llm_experiment \
  --model=llama-3.1-8b-${stage} \
  --llm_backend=hf \
  --base_model=${base_model} \
  "${adapter_flags[@]}" \
  --prompt_format=exedec \
  --two_stage_exedec=true \
  --max_dec_steps=5 \
  --num_workers=1 \
  --dataset_types=deepcoder \
  --llm_cache_dir=${cache_dir} \
  --version_deepcoder=4 \
  --version_robustfill=1 \
  "${@:2}"
