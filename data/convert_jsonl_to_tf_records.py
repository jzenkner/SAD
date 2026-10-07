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

"""Converts benchmark jsonl files to TFRecord test sets.

Reads `<input_dir>/<domain>/<TASK>.jsonl` and writes
`<output_dir>/<domain>_data/<TASK>_data/entire_programs_test.tf_records-00000-of-00001`,
the layout the end-to-end evaluation scripts expect next to the generated data.

Run from the repo root, e.g.:
    python -m data.convert_jsonl_to_tf_records \
      --input_dir=./data/test_data --output_dir=./generated_data \
      --domains=lambdabeam --tasks=LAMBDABEAM_HANDWRITTEN
"""

from collections.abc import Sequence
import json
import os
from typing import Any

from absl import app
from absl import flags
import tensorflow as tf

_INPUT_DIR = flags.DEFINE_string(
    'input_dir', './data/test_data',
    'Directory containing <domain>/<TASK>.jsonl files.')
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', os.environ.get('DATA_DIR', './generated_data'),
    'Root directory for the TFRecord files.')
_DOMAINS = flags.DEFINE_list(
    'domains', [],
    'Only convert these domains (subdirectories of --input_dir). Empty: all.')
_TASKS = flags.DEFINE_list(
    'tasks', [],
    'Only convert these tasks (jsonl file names without suffix). Empty: all.')


def _bytes_feature(strs):
  """Returns a bytes_list Feature from a list of strings."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(
      value=[s if isinstance(s, bytes) else str.encode(s) for s in strs]))


def serialize_task(task: dict[str, Any]):
  """Creates a tf.Example message for a PBE task."""
  inputs = [str(x) for x in task['inputs']]  # e.g., "x0 = [1 2]", already strings
  outputs = [
    f"[ {x[1:-1].replace(',', '')} ]" if x.startswith('[') else str(x) for x in task['outputs']
  ] # e.g., "[3]" as string
  program = task['program']
  feature = {
      'inputs': _bytes_feature(inputs),
      'outputs': _bytes_feature(outputs),
      'program': _bytes_feature([program]),
  }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  out_filename = None
  for dataset in sorted(os.listdir(_INPUT_DIR.value)):
    if _DOMAINS.value and dataset not in _DOMAINS.value:
      continue
    print(f'Processing {dataset=}')

    for filename in sorted(os.listdir(os.path.join(_INPUT_DIR.value, dataset))):
      generalization_task = filename.removesuffix('.jsonl')
      if not filename.endswith('.jsonl') or (
          _TASKS.value and generalization_task not in _TASKS.value):
        continue
      print(f'  Processing {filename=}')

      with open(os.path.join(_INPUT_DIR.value, dataset, filename)) as f:
        data = [json.loads(line) for line in f.readlines()]

      this_result_dir = os.path.join(
          _OUTPUT_DIR.value, dataset + '_data', generalization_task + '_data')
      os.makedirs(this_result_dir, exist_ok=True)
      out_filename = os.path.join(
          this_result_dir, 'entire_programs_test.tf_records-00000-of-00001')

      with tf.io.TFRecordWriter(out_filename) as writer:
        for task in data:
          writer.write(serialize_task(task))
      print(f'    Wrote {len(data)} tasks to {out_filename=}')

  if out_filename is None:
    raise app.UsageError('No jsonl files matched --domains/--tasks.')


if __name__ == '__main__':
  app.run(main)
