"""Extract the solved I/O rows for every SOLVED concept.

A concept is SOLVED (in $DATA_DIR/rules_et_al/rules_bigbench_programs.csv) when its synthesized LambdaBeam program
reproduces at least MIN_SOLVED of the 32 I/O pairs. This script keeps, per solved concept, exactly
the rows where program(input) == output (the "solved I/O"), and attaches the program.

Output: $DATA_DIR/rules_et_al/rules_bigbench_solved.csv  (id, trial, concept, input1, input2, output, run, program)
Feed it to data/adapt_bigbench_solved.py for the list-length / int-range adaptation.

Run from the repo root:  python -m data.extract_solved_io
"""

import ast
import csv
import os
import sys

from absl import flags

flags.FLAGS(sys.argv[:1])

from tasks.lambdabeam.lambdabeam_dsl import Program  # noqa: E402

RULES_DIR = os.path.join(os.environ.get('DATA_DIR', './generated_data'), 'rules_et_al')
PROGRAMS_CSV = os.path.join(RULES_DIR, 'rules_bigbench_programs.csv')
SOURCE_CSV = './data/rules_bigbench.csv'
OUT_CSV = os.path.join(RULES_DIR, 'rules_bigbench_solved.csv')


def run_fresh(program, input_list):
  """Fresh Program per call (the LambdaBeam Program is stateful across run() calls)."""
  state = Program.from_str(program).run([list(input_list)])
  return None if state is None else state.get_output()


def main():
  program_by_id = {}
  for r in csv.DictReader(open(PROGRAMS_CSV)):
    if r['status'] == 'SOLVED' and r['program']:
      program_by_id[r['id']] = r['program']

  src = list(csv.DictReader(open(SOURCE_CSV)))
  fieldnames = list(src[0].keys()) + ['program']

  out_rows = []
  per_concept = {}
  for row in src:
    cid = row['id']
    program = program_by_id.get(cid)
    if program is None:
      continue
    inp = ast.literal_eval(row['input1'])
    out = ast.literal_eval(row['output'])
    if run_fresh(program, inp) == out:  # this example is actually solved
      row = dict(row)
      row['program'] = program
      out_rows.append(row)
      per_concept[cid] = per_concept.get(cid, 0) + 1

  with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(out_rows)

  print(f'SOLVED concepts: {len(program_by_id)}')
  print(f'Extracted {len(out_rows)} solved I/O rows to {OUT_CSV}')
  if per_concept:
    print(f'Solved I/O per concept: min={min(per_concept.values())} '
          f'max={max(per_concept.values())}')
    low = sorted(c for c, n in per_concept.items() if n < 4)
    if low:
      print(f'Concepts with < 4 solved I/O rows: {low}')
    missing = [c for c in program_by_id if c not in per_concept]
    if missing:
      print(f'SOLVED concepts with 0 matching rows under fresh execution: {missing}')


if __name__ == '__main__':
  main()
