#!/bin/bash
# Few-shot prompting baseline: queries a local Ollama server (start it with
# `ollama serve` and pull the model first).

model_type=llama3.1:70b

for prediction_type in exedec; do
    python -m spec_decomposition.run_llm_experiment \
    --model=${model_type} \
    --prompt_format=${prediction_type} \
    --llm_cache_dir=./llm_cache/${model_type}_${prediction_type} \
    --version_deepcoder=4 \
    --version_robustfill=1
done