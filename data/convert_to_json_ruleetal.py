"""Convert the "rules et al" concept-learning tasks (the adapted BIG-bench
list_functions CSV written by adapt_bigbench_solved.py) into the RULES_ET_AL.jsonl
format consumed by convert_jsonl_to_tf_records.py.

Each task's raw I/O values are adapted so the tasks are solvable by our models
pretrained on DeepCoder/LambdaBeam:
  - lists are truncated to at most MAX_LEN elements,
  - all ints are rescaled into [INT_MIN, INT_MAX] preserving relative distances,
  - programs take at most 2 input variables (already true in the source data).
The GT program (the `program` column) is then re-executed on the adapted inputs
to recompute the output, and a trial is kept only if that output is also valid
(within [INT_MIN, INT_MAX], length <= MAX_LEN, not None). Exactly NUM_EXAMPLES
valid trials are emitted per task; tasks that cannot reach that many are dropped.

Run from the repo root as a module so `tasks.*` is importable:
    python -m data.convert_to_json_ruleetal
"""

import ast
import json
import math
import os
import sys

from absl import flags
import pandas as pd

from tasks.lambdabeam.lambdabeam_dsl import Program, ProgramState, result_to_str

# The DSL reads absl flags (max_int, max_list_length) at run time; parse them so
# their defaults (50 / 5) are available. Our own [-5, 5] / MAX_LEN checks are
# applied separately in output_ok / adapt_trial.
flags.FLAGS(sys.argv[:1])

INT_MIN, INT_MAX = -5, 5
MAX_LEN = 5
NUM_EXAMPLES = 4

RULES_DIR = os.path.join(os.environ.get('DATA_DIR', './generated_data'), 'rules_et_al')
CSV_PATH = os.path.join(RULES_DIR, 'rules_bigbench_solved_adapted.csv')
# <input_dir>/<domain>/<TASK>.jsonl, as read by convert_jsonl_to_tf_records.py.
OUT_PATH = os.path.join(RULES_DIR, 'jsonl', 'lambdabeam', 'RULES_ET_AL.jsonl')


def _clamp(x, lo, hi):
  return max(lo, min(hi, x))


def rescale_list(lst, lo=INT_MIN, hi=INT_MAX):
  """Affinely maps a list of ints into [lo, hi] (within [INT_MIN, INT_MAX]).

  An affine map preserves the relative distances between elements. When all
  elements are equal there is no distance to preserve, so each is clamped into
  range. A narrower [lo, hi] leaves headroom so an operation that inflates
  values (e.g. Map (+1)) still produces an in-range output; a non-negative
  [0, hi] keeps values usable as list indices.
  """
  if not lst:
    return []
  m, M = min(lst), max(lst)
  if M == m:
    return [_clamp(m, lo, hi)] * len(lst)
  scale = (hi - lo) / (M - m)
  return [round(lo + (x - m) * scale) for x in lst]


def _all_ints(out):
  if isinstance(out, list):
    return out
  return [out]


def output_ok(out):
  """True if the recomputed output satisfies the input constraints too."""
  if out is None:
    return False
  if isinstance(out, list) and len(out) > MAX_LEN:
    return False
  return all(isinstance(x, int) and INT_MIN <= x <= INT_MAX for x in _all_ints(out))


def _has_second_input(input2):
  if input2 is None:
    return False
  if isinstance(input2, float) and math.isnan(input2):
    return False
  if isinstance(input2, str) and input2.strip() in ('', 'None', 'nan'):
    return False
  return True


def adapt_trial(program, input1, input2):
  """Adapts one trial's inputs and re-executes the GT program.

  Rescales the input list and tries decreasing amplitudes (INT_MAX down to 1),
  keeping the largest one for which the recomputed output is also in range. At
  each amplitude it first tries a symmetric target range [-amp, amp], then a
  non-negative range [0, amp]. The non-negative fallback rescues programs that
  use a list value as an index into the list (e.g. Head then Access), where
  negative rescaled values would be out-of-bounds indices. Both targets are
  affine maps, so the input's relative distances are preserved, and both keep
  input and output within [INT_MIN, INT_MAX].

  Returns (input_str, output_str) on success, or None if nothing yields a valid
  example.
  """
  raw = ast.literal_eval(input1)[:MAX_LEN]
  prog = Program.from_str(program)

  for amp in range(INT_MAX, 0, -1):
    for lo, hi in ((-amp, amp), (0, amp)):
      list0 = rescale_list(raw, lo, hi)

      if _has_second_input(input2):
        x1 = int(input2)
        # Keep the scalar a valid index/count for the (possibly truncated) list.
        # With len(list0) <= MAX_LEN this is always within [INT_MIN, INT_MAX].
        x1 = _clamp(x1, 0, max(len(list0) - 1, 0))
        values = [list0, x1]
        variables = ['x0', 'x1']
      else:
        values = [list0]
        variables = ['x0']

      state = prog.run(values)
      if state is None:
        continue
      out = state.get_output()
      if output_ok(out):
        return str(ProgramState(values, variables)), result_to_str(out)

  return None


def main():
  df = pd.read_csv(CSV_PATH)
  df = df[df['run'] == 1]

  json_data = []
  dropped = []  # (task_id, num_valid)

  # groupby(sort=False) preserves the order tasks first appear in the CSV.
  for task_id, group in df.groupby('id', sort=False):
    program = group['program'].iloc[0]

    inputs, outputs = [], []
    for _, row in group.iterrows():
      adapted = adapt_trial(program, row['input1'], row['input2'])
      if adapted is None:
        continue
      input_str, output_str = adapted
      inputs.append(input_str)
      outputs.append(output_str)
      if len(inputs) == NUM_EXAMPLES:
        break

    if len(inputs) < NUM_EXAMPLES:
      dropped.append((task_id, len(inputs)))
      continue

    json_data.append({
        'index': len(json_data),
        'inputs': inputs,
        'outputs': outputs,
        'program': program,
    })

  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  with open(OUT_PATH, 'w') as f:
    for entry in json_data:
      f.write(json.dumps(entry) + '\n')

  print(f'Wrote {len(json_data)} tasks to {OUT_PATH}')
  print(f'Dropped {len(dropped)} tasks (fewer than {NUM_EXAMPLES} valid trials):')
  for task_id, n_valid in dropped:
    program = df[df['id'] == task_id]['concept'].iloc[0]
    print(f'  {task_id}: {n_valid} valid  |  {program}')


if __name__ == '__main__':
  main()
