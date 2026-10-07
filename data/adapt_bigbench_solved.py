"""Adapt the solved BIG-bench tasks so every I/O example fits the model constraints.

Reads $DATA_DIR/rules_et_al/rules_bigbench_solved.csv (36 concepts x 32 examples, each with a verified LambdaBeam
program). For every example it:
  - truncates the input list to <= MAX_LEN elements,
  - rescales the input ints into [-5, 5] preserving relative distances (largest amplitude first,
    symmetric [-A, A] then non-negative [0, A]; mirrors data/convert_to_json_ruleetal.py),
  - re-executes the concept's LambdaBeam program on the adapted input,
  - keeps the example only if the recomputed output is also in range ([-5, 5], length <= MAX_LEN).

Writes $DATA_DIR/rules_et_al/rules_bigbench_solved_adapted.csv with the adapted input/output (and the program).
Duplicate adapted (input, output) pairs within a concept are dropped.

Run from the repo root:  python -m data.adapt_bigbench_solved
"""

import ast
import csv
import collections
import os
import sys

from absl import flags

flags.FLAGS(sys.argv[:1])

from tasks.lambdabeam.lambdabeam_dsl import Program  # noqa: E402

INT_MIN, INT_MAX = -5, 5
MAX_LEN = 5

RULES_DIR = os.path.join(os.environ.get('DATA_DIR', './generated_data'), 'rules_et_al')
IN_CSV = os.path.join(RULES_DIR, 'rules_bigbench_solved.csv')
OUT_CSV = os.path.join(RULES_DIR, 'rules_bigbench_solved_adapted.csv')


def _clamp(x, lo, hi):
  return max(lo, min(hi, x))


def rescale_list(lst, lo, hi):
  """Affinely map ints of `lst` into [lo, hi] (preserves relative distances)."""
  if not lst:
    return []
  m, M = min(lst), max(lst)
  if M == m:
    return [_clamp(m, lo, hi)] * len(lst)
  scale = (hi - lo) / (M - m)
  return [round(lo + (x - m) * scale) for x in lst]


def _ints_of(out):
  return out if isinstance(out, list) else [out]


def output_ok(out):
  if out is None:
    return False
  if isinstance(out, list) and len(out) > MAX_LEN:
    return False
  return all(isinstance(x, int) and INT_MIN <= x <= INT_MAX for x in _ints_of(out))


def run_program(program, input_list):
  """Execute `program` on a single input list, using a FRESH Program each call.

  The LambdaBeam Program object is stateful across run() calls (empty-list edge cases can
  diverge when an instance is reused), so a fresh instance per call guarantees deterministic,
  reproducible results. Returns the output or None.
  """
  state = Program.from_str(program).run([list(input_list)])
  return None if state is None else state.get_output()


def adapt_example(program, input1):
  """Return (adapted_list, output) or None if no in-range adaptation exists."""
  raw = ast.literal_eval(input1)[:MAX_LEN]
  for amp in range(INT_MAX, 0, -1):
    for lo, hi in ((-amp, amp), (0, amp)):
      list0 = rescale_list(raw, lo, hi)
      out = run_program(program, list0)
      if output_ok(out):
        return list0, out
  return None


def main():
  rows = list(csv.DictReader(open(IN_CSV)))
  by = collections.OrderedDict()
  for r in rows:
    by.setdefault(r['id'], []).append(r)

  out_rows = []
  per_concept = {}
  for cid, group in by.items():
    program = group[0]['program']
    concept = group[0]['concept']
    seen = set()
    trial = 0
    for r in group:
      adapted = adapt_example(program, r['input1'])
      if adapted is None:
        continue
      list0, out = adapted
      key = (tuple(list0), tuple(out) if isinstance(out, list) else out)
      if key in seen:
        continue
      seen.add(key)
      trial += 1
      out_rows.append({
          'id': cid, 'trial': trial, 'concept': concept,
          'input1': str(list0), 'input2': '', 'output': str(out),
          'run': 1, 'program': program,
      })
    per_concept[cid] = trial

  fieldnames = ['id', 'trial', 'concept', 'input1', 'input2', 'output', 'run', 'program']
  with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(out_rows)

  # Verify all constraints hold, and that every stored output re-executes exactly.
  bad = 0
  mismatch = 0
  for r in out_rows:
    inp = ast.literal_eval(r['input1'])
    out = ast.literal_eval(r['output'])
    for v in (inp, out):
      if isinstance(v, list) and len(v) > MAX_LEN:
        bad += 1
      if any(not (INT_MIN <= x <= INT_MAX) for x in (v if isinstance(v, list) else [v])):
        bad += 1
    if run_program(r['program'], inp) != out:
      mismatch += 1

  concepts_with_examples = sum(1 for n in per_concept.values() if n > 0)
  print(f'Wrote {len(out_rows)} adapted examples to {OUT_CSV}')
  print(f'Concepts: {len(by)} solved -> {concepts_with_examples} with >=1 in-range example')
  print(f'Per-concept adapted-example counts: min={min(per_concept.values())} '
        f'max={max(per_concept.values())}')
  empty = [c for c, n in per_concept.items() if n == 0]
  if empty:
    print(f'Concepts with 0 in-range examples (dropped): {empty}')
  low = [(c, n) for c, n in per_concept.items() if 0 < n < 4]
  if low:
    print(f'Concepts with fewer than 4 adapted examples: {low}')
  print(f'Constraint violations in output file: {bad}')
  print(f'Re-execution mismatches (program(input) != output): {mismatch}')


if __name__ == '__main__':
  main()
