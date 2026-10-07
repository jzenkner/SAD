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

# End-to-end evaluation of the DeepCoder spec decomposers trained by
# run_deepcoder_training.sh, paired with ExeDec's pretrained synthesizer.
#
# Settings that can be overridden from the environment:
#   EXPERIMENTS    generalization tasks to evaluate (default: the paper's
#                  NONE and LENGTH_GENERALIZATION)
#   TRAIN_OPTIONS  which decomposers to evaluate (default: "supervised grpo");
#                  "exedec" evaluates the ExeDec baseline (ExeDec's pretrained
#                  decomposer and synthesizer_model, sampling with compute_align)
#   SEEDS          seeds to evaluate (default: "10 20 30 40 50")
#   ORACLE_TYPE    none   - use the decomposer's predicted subgoals
#                  kbest  - pick the ground-truth subgoal if it is in the beam
#                  oracle - always use the ground-truth subgoal
#   NSA            true   - synthesizer is ExeDec's joint_model, the one SAD
#                           trains against
#                  false  - synthesizer is ExeDec's synthesizer_model
#   NUM_TEST       number of test tasks (default: 1000)
# Extra arguments are passed through to launch_end_to_end_predict.py.

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

DATA_DIR=${DATA_DIR:-./generated_data}
RESULTS_DIR=${RESULTS_DIR:-./results}

read -r -a experiments_array <<< "${EXPERIMENTS:-NONE LENGTH_GENERALIZATION}"
read -r -a train_options <<< "${TRAIN_OPTIONS:-supervised grpo}"
read -r -a seeds <<< "${SEEDS:-10 20 30 40 50}"

compute_false_positives=true
greedy=true
compute_align=false
nsa=${NSA:-true}
oracle_type=${ORACLE_TYPE:-none}

num_examples=4
deepcoder_max_list_length=5
deepcoder_max_int=50
max_program_arity=2
max_num_statements=5

eval_run=e2e_predict_1
base_data_dir=${DATA_DIR}/deepcoder_data
train_run=1
base_model_dir=${RESULTS_DIR}/exedec_train_deepcoder_run-${train_run}

num_test=${NUM_TEST:-1000}
embedding_dim=512
hidden_dim=1024

# Reimplement the length and distance computation from launch_train.py.
# It is important that these distances are exactly as used in training.
object_token_length=$((deepcoder_max_list_length + 5))
max_input_objects=$((max_program_arity + max_num_statements - 1))
max_input_length=$((max_input_objects * object_token_length))
max_output_prediction_length=$(((num_examples - 1) * object_token_length))
max_program_part_length=6
spec_decomposer_max_distance=$((max_input_length > max_output_prediction_length ? max_input_length : max_output_prediction_length))
synthesizer_max_distance=$((max_input_length > max_program_part_length ? max_input_length : max_program_part_length))
spec_decomposer_max_program_cross_embed_distance=$((max_input_length * (num_examples - 1) > max_output_prediction_length ? max_input_length * (num_examples - 1) : max_output_prediction_length))
synthesizer_max_program_cross_embed_distance=$((max_input_length * (num_examples - 1) > max_program_part_length ? max_input_length * (num_examples - 1) : max_program_length))
echo "spec_decomposer_max_distance=${spec_decomposer_max_distance}"
echo "synthesizer_max_distance=${synthesizer_max_distance}"
echo "spec_decomposer_max_program_cross_embed_distance=${spec_decomposer_max_program_cross_embed_distance}"
echo "synthesizer_max_program_cross_embed_distance=${synthesizer_max_program_cross_embed_distance}"

# Compute lengths. These don't have to be exact, only long enough.
max_num_variables=10
max_io_length=$((max_num_variables * object_token_length))
max_num_program_parts=7
max_program_length=$((max_program_part_length * max_num_program_parts))
max_spec_part_length=30

test_dataset_format=${base_data_dir}/{experiment}_data/entire_programs_test.tf_records*

synthesizer_path_format=gs://exedec/trained_models/deepcoder/synthesizer_model/checkpoints/adr=0.1,ara=False,dr=0.1,e={experiment},ed=512,hd=1024,l=0.0002,md=60,mpced=180,npb=32,s={seed},scnpr=0.0,ura=True/
if [[ "$nsa" == "true" ]]; then
  synthesizer_path_format=gs://exedec/trained_models/deepcoder/joint_model/checkpoints/adr=0.1,ara=False,dr=0.1,e={experiment},ed=512,hd=1024,l=0.0002,md=60,mpced=180,npb=32,s={seed},scnpr=0.0,ura=True/
fi

# Generate comma-separated strings to pass as an argument.
experiments=$(printf ",%s" "${experiments_array[@]}")
experiments=${experiments:1}
seed_flags=()
for seed in "${seeds[@]}"; do
  seed_flags+=("--seed=${seed}")
done

prediction_type=separate

for training_type in "${train_options[@]}"; do
  spec_decomposer_path_format=${base_model_dir}/spec_decomposer_model/${training_type}/checkpoints/adr=0.1,ara=True,dr=0.1,e={experiment},ed=${embedding_dim},hd=${hidden_dim},l=0.0001,md=60,mpced=180,npb=32,s={seed},scnpr=0.0,ura=True,dm=coupled/
  this_synthesizer_path_format=${synthesizer_path_format}
  this_nsa=${nsa}
  this_greedy=${greedy}
  this_compute_align=${compute_align}
  if [[ "${training_type}" == "exedec" ]]; then
    spec_decomposer_path_format=gs://exedec/trained_models/deepcoder/spec_decomposer_model/checkpoints/adr=0.1,ara=True,dr=0.1,e={experiment},ed=${embedding_dim},hd=${hidden_dim},l=0.0002,md=60,mpced=180,npb=32,s={seed},scnpr=0.0,ura=True/
    this_synthesizer_path_format=gs://exedec/trained_models/deepcoder/synthesizer_model/checkpoints/adr=0.1,ara=False,dr=0.1,e={experiment},ed=512,hd=1024,l=0.0002,md=60,mpced=180,npb=32,s={seed},scnpr=0.0,ura=True/
    this_nsa=false
    this_greedy=false
    this_compute_align=true
  fi
  echo "=== ${training_type}, oracle_type=${oracle_type}, nsa=${this_nsa} ==="
  save_dir=${RESULTS_DIR}/evaluation/deepcoder_${eval_run}/${training_type}_${oracle_type}_nsa${this_nsa}

  python -m spec_decomposition.launch_end_to_end_predict \
  --exp_title=end_to_end_predict-deepcoder-run-${eval_run}-${prediction_type} \
  --compute_false_positives=${compute_false_positives} \
  --oracle_type=${oracle_type} \
  --greedy_selection=${this_greedy} \
  --compute_align=${this_compute_align} \
  --save_dir=${save_dir} \
  --dataset_type=deepcoder \
  --experiments=${experiments} \
  --max_list_length=${deepcoder_max_list_length} \
  --max_int=${deepcoder_max_int} \
  --test_dataset_format=${test_dataset_format} \
  --num_test_batches=${num_test} \
  --num_examples=${num_examples} \
  --max_io_length=${max_io_length} \
  --max_program_length=${max_program_length} \
  --max_spec_part_length=${max_spec_part_length} \
  --spec_decomposer_path_format=${spec_decomposer_path_format} \
  --synthesizer_path_format=${this_synthesizer_path_format} \
  --embedding_dim=${embedding_dim} \
  --hidden_dim=${hidden_dim} \
  --spec_decomposer_num_position_buckets=32 \
  --synthesizer_num_position_buckets=32 \
  --spec_decomposer_max_distance=${spec_decomposer_max_distance} \
  --synthesizer_max_distance=${synthesizer_max_distance} \
  --spec_decomposer_max_program_cross_embed_distance=${spec_decomposer_max_program_cross_embed_distance}  \
  --synthesizer_max_program_cross_embed_distance=${synthesizer_max_program_cross_embed_distance} \
  --use_relative_attention=True \
  --beam_size=10 \
  --prediction_type=${prediction_type} \
  --detect_invalid=true \
  --use_execution=true \
  --discard_repeat_functionality=true \
  --aligned_relative_attention=true \
  --corruption_rate=0.0 \
  "${seed_flags[@]}" \
  "$@"
done
