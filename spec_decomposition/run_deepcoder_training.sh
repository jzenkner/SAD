#!/bin/bash
# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Trains the DeepCoder spec decomposer against ExeDec's frozen, pretrained
# synthesizer. The two arms share one code path:
#   supervised - cross-entropy on the ground-truth subgoals only (the control)
#   grpo       - SAD: cross-entropy + self-critical REINFORCE on the
#                synthesizer's loss + an entropy bonus (--entropy_coef)
#
# By default this runs the paper's sweep (NONE and LENGTH_GENERALIZATION, five
# seeds, both arms) sequentially; the other four ExeDec generalization tasks are
# supported too. To run one job per
# task/seed/arm, override the sweep from the environment, e.g.
#   EXPERIMENTS=NONE SEEDS=10 RL_OPTIONS=grpo ./spec_decomposition/run_deepcoder_training.sh
# Extra arguments are passed through to launch_train.py, e.g. --entropy_coef=0.1.

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NCCL_P2P_DISABLE=1

DATA_DIR=${DATA_DIR:-./generated_data}
RESULTS_DIR=${RESULTS_DIR:-./results}

# Which generalization tasks, SAD arms and seeds to train.
read -r -a experiments_array <<< "${EXPERIMENTS:-NONE LENGTH_GENERALIZATION}"
read -r -a rl_options <<< "${RL_OPTIONS:-supervised grpo}"
read -r -a seeds <<< "${SEEDS:-10 20 30 40 50}"

model_type=spec_decomposer_model
decomposition_mode=coupled

# Which dataset to train on.
examples=4  # Number of I/O examples in specifications.
list_length=5  # Max length of lists.
max_int=50  # Max integer.
data_dir=${DATA_DIR}/deepcoder_data

# The frozen synthesizer that provides the SAD reward. Seed s of the decomposer
# is paired with seed s of the synthesizer.
synthesizer_path_format=gs://exedec/trained_models/deepcoder/joint_model/checkpoints/adr=0.1,ara=False,dr=0.1,e={experiment},ed=512,hd=1024,l=0.0002,md=60,mpced=180,npb=32,s={seed},scnpr=0.0,ura=True/
synthesizer_max_distance=60
synthesizer_max_program_cross_embed_distance=180

# This training run.
run=1
save_dir=${RESULTS_DIR}/exedec_train_deepcoder_run-${run}

# Generate comma-separated strings to pass as an argument.
experiments=$(printf ",%s" "${experiments_array[@]}")
experiments=${experiments:1}
seed_flags=()
for seed in "${seeds[@]}"; do
  seed_flags+=("--seed=${seed}")
done

for rl_loss in "${rl_options[@]}"; do
  echo "=== ${rl_loss} on ${experiments} ==="
  python -m spec_decomposition.launch_train \
    --save_dir=${save_dir} \
    --rl_loss=${rl_loss} \
    --decomposition_mode=${decomposition_mode} \
    --dataset_type=deepcoder \
    --experiments=${experiments} \
    --dataset_dir=${data_dir} \
    --num_examples=${examples} \
    --max_program_arity=2 \
    --max_num_statements=5 \
    --max_list_length=${list_length} \
    --max_int=${max_int} \
    --num_train_steps=500000 \
    --num_eval_steps=10 \
    --model_type=${model_type} \
    --per_device_batch_size=128 \
    --lr=1e-4 \
    --embedding_dim=512 \
    --hidden_dim=1024 \
    --dropout_rate=0.1 \
    --attention_dropout_rate=0.1 \
    --num_position_buckets=32 \
    --aligned_relative_attention=1 \
    --synthesizer_corrupted_next_part_rate=0.0 \
    "${seed_flags[@]}" \
    --log_freq=2000 \
    --eval_freq=10000 \
    --predict_freq=50000 \
    --checkpoint_freq=50000 \
    --synthesizer_path_format=${synthesizer_path_format} \
    --synthesizer_num_position_buckets=32 \
    --synthesizer_max_target_length=6 \
    --synthesizer_max_distance=${synthesizer_max_distance} \
    --synthesizer_max_program_cross_embed_distance=${synthesizer_max_program_cross_embed_distance} \
    --exp_title=${rl_loss} \
    "$@"
done
