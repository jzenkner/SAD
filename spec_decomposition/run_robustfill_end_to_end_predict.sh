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

# End-to-end evaluation of the RobustFill spec decomposers trained by
# run_robustfill_training.sh, paired with ExeDec's pretrained synthesizer.
# The environment overrides (EXPERIMENTS, TRAIN_OPTIONS, SEEDS, ORACLE_TYPE,
# NSA, NUM_TEST) are described in run_deepcoder_end_to_end_predict.sh, including
# TRAIN_OPTIONS=exedec for the ExeDec baseline.
# Extra arguments are passed through to launch_end_to_end_predict.py.

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NCCL_P2P_DISABLE=1

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

num_examples=5
embedding_dim=512
hidden_dim=1024
eval_run=e2e_predict_1

base_data_dir=${DATA_DIR}/robustfill_data
train_run=1
base_model_dir=${RESULTS_DIR}/exedec_train_robustfill_run-${train_run}

num_test=${NUM_TEST:-1000}

test_dataset_format=${base_data_dir}/{experiment}_data/entire_programs_test.tf_records*
synthesizer_path_format=gs://exedec/trained_models/robustfill/synthesizer_model/checkpoints/adr=0.1,ara=False,dr=0.1,e={experiment},ed=${embedding_dim},hd=${hidden_dim},l=0.0002,md=20,mpced=80,npb=32,s={seed},scnpr={corruption_rate},ura=True/
if [[ "$nsa" == "true" ]]; then
  synthesizer_path_format=gs://exedec/trained_models/robustfill/joint_model/checkpoints/adr=0.1,ara=False,dr=0.1,e={experiment},ed=512,hd=1024,l=0.0002,md=200,mpced=800,npb=32,s={seed},scnpr=0.0,ura=True/
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
  spec_decomposer_path_format=${base_model_dir}/spec_decomposer_model/${training_type}/checkpoints/adr=0.1,ara=True,dr=0.1,e={experiment},ed=${embedding_dim},hd=${hidden_dim},l=0.0001,md=200,mpced=800,npb=32,s={seed},scnpr=0.0,ura=True,dm=coupled/
  this_synthesizer_path_format=${synthesizer_path_format}
  this_nsa=${nsa}
  this_greedy=${greedy}
  this_compute_align=${compute_align}
  if [[ "${training_type}" == "exedec" ]]; then
    spec_decomposer_path_format=gs://exedec/trained_models/robustfill/spec_decomposer_model/checkpoints/adr=0.1,ara={aligned_relative_attention},dr=0.1,e={experiment},ed=${embedding_dim},hd=${hidden_dim},l=0.0002,md=200,mpced=800,npb=32,s={seed},scnpr=0.0,ura=True/
    this_synthesizer_path_format=gs://exedec/trained_models/robustfill/synthesizer_model/checkpoints/adr=0.1,ara=False,dr=0.1,e={experiment},ed=${embedding_dim},hd=${hidden_dim},l=0.0002,md=20,mpced=80,npb=32,s={seed},scnpr={corruption_rate},ura=True/
    this_nsa=false
    this_greedy=false
    this_compute_align=true
  fi
  echo "=== ${training_type}, oracle_type=${oracle_type}, nsa=${this_nsa} ==="
  if [[ "${prediction_type}" == "separate" || "${prediction_type}" == "tiips" ]]; then
    max_io_length=200
    max_program_length=100
    max_spec_part_length=85
    spec_decomposer_max_distance=200
    spec_decomposer_max_program_cross_embed_distance=800
    synthesizer_max_distance=20
    synthesizer_max_program_cross_embed_distance=80
    if [[ "$this_nsa" == "true" ]]; then
      synthesizer_max_distance=200
      synthesizer_max_program_cross_embed_distance=800
    fi
  elif [[ "${prediction_type}" == "baseline" ]]; then
    max_io_length=200
    max_program_length=100
    max_spec_part_length=-1  # Unused.
    spec_decomposer_max_distance=-1  # Unused.
    synthesizer_max_distance=200
    spec_decomposer_max_program_cross_embed_distance=-1  # Unused.
    synthesizer_max_program_cross_embed_distance=800
  else
    echo "Unhandled model ${prediction_type}"
    exit 1
  fi

  save_dir=${RESULTS_DIR}/evaluation/robustfill_${eval_run}/${training_type}_${oracle_type}_nsa${this_nsa}

  python -m spec_decomposition.launch_end_to_end_predict \
  --exp_title=end_to_end_predict-robustfill-run-${eval_run}-${prediction_type} \
  --compute_false_positives=${compute_false_positives} \
  --oracle_type=${oracle_type} \
  --greedy_selection=${this_greedy} \
  --compute_align=${this_compute_align} \
  --save_dir=${save_dir} \
  --dataset_type=robustfill \
  --experiments=${experiments} \
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
