r"""Builds per-role SFT data for the ExeDec decomposer and synthesizer LLMs.

Reads the same `entire_programs_*.tf_records` shards the from-scratch
transformers train on, replays each ground-truth program step by step with
`llm_utils.get_exe_dec_trajectory`, and writes one prompt/completion record per
(task, step, role).

Usage:
  python -m spec_decomposition.llm_finetune.build_sft_data \
    --generalization_task=NONE --split=train --max_tasks=20000
"""

import collections
import hashlib
import json
import os
import random
import re

from absl import app
from absl import flags
from absl import logging
import numpy as np

from spec_decomposition import llm_utils
from spec_decomposition.end_to_end_predict import create_deepcoder_dataset
from spec_decomposition.llm_finetune import data
from spec_decomposition.llm_finetune import prompts
from tasks.deepcoder import deepcoder_dsl

_DATA_DIR = flags.DEFINE_string(
    'data_dir', './generated_data/deepcoder_data',
    'Directory holding the <EXPERIMENT>_data subdirectories of TFRecords.')
# Flag names avoid save_dir / experiment / num_examples / max_program_length /
# seed, which spec_decomposition.end_to_end_predict already defines at import.
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', './data/llm_data/deepcoder_sft',
    'Where to write the per-role JSONL files.')
_GENERALIZATION_TASK = flags.DEFINE_string(
    'generalization_task', 'NONE', 'Generalization split to read.')
_SPLIT = flags.DEFINE_enum(
    'split', 'train', ['train', 'valid', 'test'], 'Which TFRecord split.')
_MAX_TASKS = flags.DEFINE_integer(
    'max_tasks', 20000, 'Maximum number of tasks to convert.')
_NUM_FEW_SHOT = flags.DEFINE_integer(
    'num_few_shot', 4, 'Few-shot examples per prompt.')
_FEW_SHOT_POOL_SIZE = flags.DEFINE_integer(
    'few_shot_pool_size', 200,
    'Tasks reserved exclusively as few-shot donors. These never become '
    'training records, so a task can never appear in its own prompt.')
_IO_EXAMPLES = flags.DEFINE_integer(
    'io_examples', 4, 'I/O examples per task in the TFRecords.')
_VERSION_DEEPCODER = flags.DEFINE_integer(
    'version_deepcoder', 4, 'DeepCoder Python program version (4 = Pythonic).')
_MAX_STATEMENTS = flags.DEFINE_integer(
    'max_statements', 5,
    'Skip tasks whose program has more statements than this.')
_DATA_SEED = flags.DEFINE_integer(
    'data_seed', 0, 'Seed for few-shot sampling.')

DATASET_TYPE = 'deepcoder'


# The three helpers below were previously imported from
# data/llm_data/get_data_from_trafo_data.py. They live here now: that file was
# untracked and got deleted, `data/` is a namespace package so the import only
# resolved when cwd happened to be the repo root, and importing it ran
# `flags.FLAGS([''])` and a `mkdir` at module scope. They are pure functions, so
# vendoring them costs nothing and makes this script self-contained.


def parse_input_string(input_strings):
  """Converts DeepCoder input specs into per-variable Python values.

  ['x0 = [ 4 1 2 ] | x1 = 0', 'x0 = [ 4 2 3 ] | x1 = 1']
    -> {'x0': [[4, 1, 2], [4, 2, 3]], 'x1': [0, 1]}
  """
  inputs_dict = collections.defaultdict(list)

  for s in input_strings:
    parts = [p.strip() for p in s.split('|')]
    for part in parts:
      # A list, e.g. "x0 = [ 4 1 2 ]".
      m_list = re.match(r"x(\d+)\s*=\s*\[\s*([^\]]*?)\s*\]", part)
      if m_list:
        var_idx, values = m_list.groups()
        var_name = f'x{var_idx}'
        if values.strip() == '':
          parsed_values = []
        else:
          parsed_values = [int(v) for v in values.strip().split()]
        inputs_dict[var_name].append(parsed_values)
        continue

      # A single integer, e.g. "x1 = 0".
      m_int = re.match(r"x(\d+)\s*=\s*(-?\d+)", part)
      if m_int:
        var_idx, value = m_int.groups()
        inputs_dict[f'x{var_idx}'].append(int(value))
        continue

      raise ValueError(f'Cannot parse input part: {part}')

  return dict(inputs_dict)


def parse_output_strings(output_strings):
  """Converts ['[ -8 ]', '3'] into [[-8], 3]."""
  parsed_outputs = []
  for s in output_strings:
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
      parsed_outputs.append([int(v) for v in s[1:-1].strip().split()])
    else:
      parsed_outputs.append(int(s))
  return parsed_outputs


def decode_spec(target, dataset, spec_id_token_table, bos_id, eos_id):
  """Converts a token-id tensor back into its spec string."""
  if dataset not in ('robustfill', 'deepcoder', 'lambdabeam'):
    raise ValueError('Unhandled dataset_type: {}'.format(dataset))
  target = np.array(target)
  target = target[(target != 0) & (target != bos_id)
                  & (target != eos_id)].astype(np.int32)
  separator = ' ' if dataset in ('deepcoder', 'lambdabeam') else ''
  return separator.join(
      [spec_id_token_table[t_id] for t_id in target if t_id > 0])


def _decode_records(file_pattern, id_to_token, token_to_id, num_examples):
  """Yields (inputs_dict, outputs_list, dsl_program_str) from TFRecords."""
  dataset = create_deepcoder_dataset(
      file_pattern, token_to_id, num_examples, DATASET_TYPE)
  bos_id, eos_id = token_to_id['<BOS>'], token_to_id['<EOS>']

  for data in dataset.as_numpy_iterator():
    input_strs = [decode_spec(row, DATASET_TYPE, id_to_token, bos_id, eos_id)
                  for row in data['inputs']]
    output_strs = [decode_spec(row, DATASET_TYPE, id_to_token, bos_id, eos_id)
                   for row in data['outputs']]
    # An empty final output leaves the last (held-out) test case degenerate;
    # get_data_from_trafo_data.py drops these too, so eval never sees them.
    if output_strs[-1] == '[ ]':
      continue
    try:
      program = deepcoder_dsl.Program.from_tokens(
          [id_to_token[int(p_id)] for p_id in data['target']
           if p_id > 0 and p_id != deepcoder_dsl.EOS_ID])
      inputs = parse_input_string(input_strs)
      outputs = parse_output_strings(output_strs)
    except Exception:  # pylint: disable=broad-exception-caught
      continue
    yield inputs, outputs, str(program)


def _to_element(inputs, outputs, dsl_program, version):
  """Builds a canonicalized DatasetElement, or None if it is unusable."""
  try:
    element = llm_utils.json_to_dataset_element(
        {'inputs': inputs, 'outputs': outputs, 'program': dsl_program},
        DATASET_TYPE, version)
    return llm_utils.canonicalize_deepcoder_variables(element)
  except Exception:  # pylint: disable=broad-exception-caught
    return None


def _num_statements(element):
  return len(element.python_program.splitlines()) - 2  # signature and return


def _record_for_task(task_id, element, few_shots, version):
  """Builds one packed record covering every step of one task.

  Each step used to be its own sequence, so a task was forwarded once per step
  per role -- re-encoding a ~2400-token prefix that barely changed each time.
  Instead the whole trajectory is stored once, cut into alternating segments
  tagged with the role that must predict them:

      [prefix, None] [subgoal_0, decomposer] [between, None] [code_0, synth] ...

  Because step j's evaluation context is exactly the trajectory prefix up to
  `Step j+1 computes:`, and attention is causal, one forward over the packed
  sequence gives the same logits at every completion position as the separate
  per-step forwards did. `prompts_test` asserts that prefix property, and
  `iter_steps` in data.py reconstructs the per-step view for SAD.
  """
  trajectory = llm_utils.get_exe_dec_trajectory(element, DATASET_TYPE)
  # The prompt shows every example but the last, which stays held out.
  num_examples = llm_utils.get_num_examples(element.inputs, DATASET_TYPE) - 1

  segments = []
  consumed = 0  # characters of the packed text already emitted as segments.

  for step_idx in range(len(trajectory) - 1):
    partial = prompts.partial_python_program(trajectory, step_idx)
    step_ctx = prompts.step_context(
        few_shots, element, DATASET_TYPE, version, step_idx, partial)
    subgoal = prompts.render_subgoal(
        trajectory[step_idx + 1].states, num_examples, DATASET_TYPE)
    synth_ctx = prompts.synthesizer_context(step_ctx, subgoal, step_idx)
    code = prompts.render_step_code(
        trajectory[step_idx + 1].python_program_step)

    # Every context is a prefix of the packed text, so the text between the
    # previous completion and this one is just the slice in between.
    if len(step_ctx) < consumed:
      raise ValueError(
          f'Step {step_idx} context is shorter than what is already packed; '
          'the trajectory prefixes are not nested as expected.')
    segments.append([step_ctx[consumed:], None])
    segments.append([subgoal, data.DECOMPOSER])
    segments.append([synth_ctx[len(step_ctx) + len(subgoal):], None])
    segments.append([code, data.SYNTHESIZER])
    consumed = len(synth_ctx) + len(code)

  return {
      'task_id': task_id,
      'num_steps': len(trajectory) - 1,
      'dsl_program': element.dsl_program,
      'python_program': element.python_program,
      'inputs': element.inputs,
      'outputs': element.outputs,
      'segments': segments,
  }


def main(_):
  rng = random.Random(_DATA_SEED.value)
  id_to_token, token_to_id = deepcoder_dsl.vocab_tables()
  version = _VERSION_DEEPCODER.value

  file_pattern = os.path.join(
      _DATA_DIR.value, f'{_GENERALIZATION_TASK.value}_data',
      f'entire_programs_{_SPLIT.value}.tf_records-*')
  logging.info('Reading %s', file_pattern)

  few_shot_pool = []
  tasks = []
  seen = set()
  stats = collections.Counter()

  for inputs, outputs, dsl_program in _decode_records(
      file_pattern, id_to_token, token_to_id, _IO_EXAMPLES.value):
    stats['decoded'] += 1
    key = hashlib.md5(
        json.dumps([inputs, outputs, dsl_program], sort_keys=True).encode()
    ).hexdigest()
    if key in seen:
      stats['duplicate'] += 1
      continue
    seen.add(key)

    element = _to_element(inputs, outputs, dsl_program, version)
    if element is None:
      stats['unparseable'] += 1
      continue
    if _num_statements(element) > _MAX_STATEMENTS.value:
      stats['too_long'] += 1
      continue
    # A few-shot donor must itself render, since the prompt replays its steps.
    try:
      llm_utils.get_exe_dec_trajectory(element, DATASET_TYPE)
    except Exception:  # pylint: disable=broad-exception-caught
      stats['no_trajectory'] += 1
      continue

    if len(few_shot_pool) < _FEW_SHOT_POOL_SIZE.value:
      few_shot_pool.append(element)
    else:
      tasks.append(element)
      if len(tasks) >= _MAX_TASKS.value:
        break

  if len(few_shot_pool) < _NUM_FEW_SHOT.value:
    raise ValueError(
        f'Few-shot pool has only {len(few_shot_pool)} usable tasks, need at '
        f'least {_NUM_FEW_SHOT.value}.')
  logging.info('Kept %d tasks (+%d few-shot donors). Filters: %s',
               len(tasks), len(few_shot_pool), dict(stats))

  os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
  path = os.path.join(
      _OUTPUT_DIR.value,
      f'{_GENERALIZATION_TASK.value}_{_SPLIT.value}.jsonl')
  num_written = 0

  num_steps = 0
  with open(path, 'w') as f:
    for task_id, element in enumerate(tasks):
      few_shots = rng.sample(few_shot_pool, _NUM_FEW_SHOT.value)
      try:
        record = _record_for_task(task_id, element, few_shots, version)
      except Exception:  # pylint: disable=broad-exception-caught
        # A few-shot donor or this task failed to replay; drop the whole task
        # rather than emit a half-built prompt.
        stats['render_failed'] += 1
        continue
      f.write(json.dumps(record) + '\n')
      num_written += 1
      num_steps += record['num_steps']

  logging.info('Wrote %d packed task records (%d steps) to %s',
               num_written, num_steps, path)
  print(f'tasks={len(tasks)} filters={dict(stats)} '
        f'records={num_written} steps={num_steps}')
  print(f'  {path}')


if __name__ == '__main__':
  app.run(main)
