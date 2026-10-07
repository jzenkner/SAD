"""Round-trips ground-truth SFT completions through the evaluation path.

Takes each step record's `step_code` completion, feeds it through exactly the
parse-assemble-execute pipeline `run_llm_experiment.solve_problem_exedec` uses,
and requires the final program to reproduce the task's outputs. A failure means
the SFT target is not what the evaluation harness can consume, which would make
every downstream success rate meaningless.

Usage:
  python -m spec_decomposition.llm_finetune.data_test --records=<path.jsonl>
"""

import collections
import sys

from absl import app
from absl import flags

from spec_decomposition import llm_utils
from spec_decomposition.llm_finetune import data
from spec_decomposition.llm_finetune import prompts

_RECORDS = flags.DEFINE_string(
    'records', None, 'Step-record JSONL from build_sft_data.py.',
    required=True)
_MAX_TASKS = flags.DEFINE_integer(
    'max_tasks', 100, 'Number of tasks to check.')


def main(_):
  records = data.load_records(_RECORDS.value, _MAX_TASKS.value)

  stats = collections.Counter()
  failures = []

  for record in records:
    # iter_steps rebuilds the per-step view from the packed segments; if the
    # segment boundaries were wrong, the contexts below would not line up.
    steps = list(data.iter_steps(record))
    if not steps:
      stats['no_steps'] += 1
      continue
    partial = None

    for step_record in steps:
      # The decomposer prompt must stop exactly where generation begins.
      assert step_record['context'].endswith(
          f"Step {step_record['step'] + 1} computes:\n"), (
              step_record['context'][-80:])

      # Parse the fenced completion the same way the eval harness does.
      step_code = llm_utils.cut_program_from_sample(step_record['step_code'])
      partial = data.compose_program(step_record, step_code, partial)

    outputs = llm_utils.run_program(partial, record['inputs'], 'deepcoder')
    if outputs == record['outputs']:
      stats['ok'] += 1
    else:
      stats['mismatch'] += 1
      if len(failures) < 5:
        failures.append(
            (record['task_id'], partial, outputs, record['outputs']))

  print(f'checked {len(records)} tasks: {dict(stats)}')
  for task_id, program, got, want in failures:
    print(f'\n--- task {task_id} ---\n{program}\n  got:  {got}\n  want: {want}')

  if stats['mismatch']:
    sys.exit(1)
  print('All ground-truth completions reproduce their task outputs.')


if __name__ == '__main__':
  app.run(main)
