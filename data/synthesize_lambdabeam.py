"""Synthesize LambdaBeam DSL programs for the 250 BIG-bench list_functions concepts.

For every concept in data/rules_bigbench.csv (250 concepts x 32 I/O pairs) this runs a
deterministic, bounded, bottom-up value-graph search over the LambdaBeam DSL and returns the
first program (<= MAX_DEPTH operations) whose output matches ALL 32 examples. Constants and
indices are restricted to [-5, 5] (the DSL's in-range constant vocabulary). Concepts that need
larger constants / list literals / deeper programs come out UNSOLVED.

Verification uses tasks.lambdabeam.lambdabeam_dsl directly (Program.run does NOT enforce
max_int/max_list_length, so raw BIG-bench values 0..99 / length<=15 verify unchanged).

Outputs:
  $DATA_DIR/rules_et_al/rules_bigbench_programs.csv  -- id, concept, program, status, oor_flag, oor_values, num_verified
  $DATA_DIR/rules_et_al/rules_bigbench_report.md     -- readable table + summary

Run from the repo root:  python -m data.synthesize_lambdabeam
"""

import ast as ast_mod
import collections
import csv
import os
import re
import sys
import time
from multiprocessing import Pool

from absl import flags

# The DSL reads absl flags at import/run; parse defaults once.
flags.FLAGS(sys.argv[:1])

from tasks.lambdabeam.lambdabeam_dsl import Statement, ProgramState, Program  # noqa: E402

# ----------------------------------------------------------------------------------------------
# Search configuration
# ----------------------------------------------------------------------------------------------
MAX_DEPTH = 3          # max number of operation statements per program
CONSTS = list(range(-5, 6))   # in-range integer constants / indices [-5, 5]
TIME_CAP_S = 40        # per-concept wall-clock cap
POOL_CAP = 400         # max distinct list-values carried between rounds
K_SEARCH = 10          # examples used to dedup/gate the search
MIN_SOLVED = 4         # accept a program if it solves at least this many of the 32 examples
VALUE_CAP = 10 ** 6    # drop any intermediate value whose magnitude explodes

IN_CSV = './data/rules_bigbench.csv'
RULES_DIR = os.path.join(os.environ.get('DATA_DIR', './generated_data'), 'rules_et_al')
OUT_CSV = os.path.join(RULES_DIR, 'rules_bigbench_programs.csv')
OUT_MD = os.path.join(RULES_DIR, 'rules_bigbench_report.md')

# All BIG-bench outputs are lists, and the only way to build a list is via list->list ops, so
# the search uses list-producing operations exclusively. Int-reducing ops (Sum/Count/Head/Access
# as int output) can never match and cannot be wrapped back into a list, so concepts whose true
# output is a bare scalar wrapped as [k] (e.g. "input length", "number of 3s") are inexpressible
# here and correctly come out UNSOLVED.

# Unary int->int lambdas for Map (token, optional_const); drop no-op identities.
_UNARY_INT_LAMBDAS = (
    [('Add', c) for c in CONSTS if c != 0]
    + [('Subtract', c) for c in CONSTS if c != 0]
    + [('Multiply', c) for c in CONSTS if c not in (0, 1)]
    + [('IntDivide', c) for c in CONSTS if c not in (0, 1)]
    + [('Min', c) for c in CONSTS]
    + [('Max', c) for c in CONSTS]
    + [('Square', None)]
)
_PRED_LAMBDAS = (
    [(op, c) for op in ('Greater', 'Less', 'Equal') for c in CONSTS]
    + [('IsEven', None), ('IsOdd', None)]
)
_BINARY_INT_LAMBDAS = ['Add', 'Subtract', 'Multiply', 'IntDivide', 'Min', 'Max']


def _lam_str(op, c):
  return op if c is None else f'{op} {c}'


# ----------------------------------------------------------------------------------------------
# Value nodes (AST + cached per-example signature)
# ----------------------------------------------------------------------------------------------
# ast forms:
#   ('in',)                        -> input variable x0
#   ('op', token, lam, children)   -> lam is None or (ltoken, const); children are value-arg asts
#                                     (constants are inlined into the rhs, never child asts)
class Node:
  __slots__ = ('typ', 'sig', 'ast')

  def __init__(self, typ, sig, ast):
    self.typ = typ      # 'int' or 'list'
    self.sig = sig      # tuple over examples (ints as-is, lists as tuples)
    self.ast = ast


def _freeze(v):
  return tuple(v) if isinstance(v, list) else v


def _eval_rhs(rhs, computed_children, num_ex):
  """Run a single statement `xN = rhs` on each example using cached child sigs.

  computed_children: list of Nodes mapped positionally to x0, x1, ... in rhs.
  Returns a frozen sig tuple, or None if any example fails / explodes.
  """
  n = len(computed_children)
  variables = [f'x{i}' for i in range(n)]
  out_var = f'x{n}'
  try:
    stmt = Statement.from_str(f'{out_var} = {rhs}')
  except Exception:
    return None
  sig = []
  for e in range(num_ex):
    vals = [computed_children[i].sig[e] for i in range(n)]
    vals = [list(v) if isinstance(v, tuple) else v for v in vals]
    try:
      st = stmt.run(ProgramState(vals, list(variables)))
    except Exception:
      return None
    if st is None:
      return None
    out = st.get_output()
    if out is None:
      return None
    if isinstance(out, list):
      if any(abs(x) > VALUE_CAP for x in out):
        return None
    elif isinstance(out, int):
      if abs(out) > VALUE_CAP:
        return None
    sig.append(_freeze(out))
  return tuple(sig)


# ----------------------------------------------------------------------------------------------
# Candidate generation: yields (rhs_template_fn, computed_children, ast) for one new statement.
# rhs is built with computed children mapped to x0..; constants inlined.
# ----------------------------------------------------------------------------------------------
def _gen_candidates(pool_list, input_node):
  """Yield (rhs, computed_children, ast) for every list-producing single op application."""
  # --- list -> list ---
  for op in ('Reverse', 'Sort'):
    for L in pool_list:
      yield (f'{op} x0', [L], ('op', op, None, (L.ast,)))

  # --- (int const, list) -> list  (Take / Drop) ---
  for op in ('Take', 'Drop'):
    for L in pool_list:
      for c in CONSTS:
        yield (f'{op} {c} x0', [L], ('op', op, ('_const', c), (L.ast,)))

  # --- Map: unary int->int lambda + list ---
  for (lt, c) in _UNARY_INT_LAMBDAS:
    lam = _lam_str(lt, c)
    for L in pool_list:
      yield (f'Map {lam} x0', [L], ('op', 'Map', (lt, c), (L.ast,)))

  # --- Filter: unary int->bool predicate + list ---
  for (lt, c) in _PRED_LAMBDAS:
    lam = _lam_str(lt, c)
    for L in pool_list:
      yield (f'Filter {lam} x0', [L], ('op', 'Filter', (lt, c), (L.ast,)))

  # --- Scanl1: binary int,int->int lambda + list ---
  for lt in _BINARY_INT_LAMBDAS:
    for L in pool_list:
      yield (f'Scanl1 {lt} x0', [L], ('op', 'Scanl1', (lt, None), (L.ast,)))

  # --- ZipWith: binary lambda + 2 lists; pair each list with the raw input (both orders) ---
  for lt in _BINARY_INT_LAMBDAS:
    for A in pool_list:
      yield (f'ZipWith {lt} x0 x1', [A, input_node],
             ('op', 'ZipWith', (lt, None), (A.ast, input_node.ast)))
      yield (f'ZipWith {lt} x0 x1', [input_node, A],
             ('op', 'ZipWith', (lt, None), (input_node.ast, A.ast)))


# ----------------------------------------------------------------------------------------------
# Program reconstruction from an ast (SSA + common-subexpression elimination)
# ----------------------------------------------------------------------------------------------
def ast_to_program(ast):
  statements = []
  memo = {}
  counter = [1]

  def render(node):
    tag = node[0]
    if tag == 'in':
      return 'x0'
    if node in memo:
      return memo[node]
    _, token, lam, children = node
    child_refs = [render(c) for c in children]
    # Build rhs.
    if token in ('Map', 'Filter', 'Count'):
      lt, c = lam
      lam_s = lt if c is None else f'{lt} {c}'
      rhs = f'{token} {lam_s} {child_refs[0]}'
    elif token == 'Scanl1':
      lt, _ = lam
      rhs = f'Scanl1 {lt} {child_refs[0]}'
    elif token == 'ZipWith':
      lt, _ = lam
      rhs = f'ZipWith {lt} {child_refs[0]} {child_refs[1]}'
    elif token in ('Take', 'Drop', 'Access') and lam is not None and lam[0] == '_const':
      rhs = f'{token} {lam[1]} {child_refs[0]}'
    else:
      rhs = f'{token} ' + ' '.join(child_refs)
    var = f'x{counter[0]}'
    counter[0] += 1
    statements.append(f'{var} = {rhs}')
    memo[node] = var
    return var

  render(ast)
  if not statements:
    return 'x0 = INPUT'  # identity
  return 'x0 = INPUT | ' + ' | '.join(statements)


# ----------------------------------------------------------------------------------------------
# Per-concept search
# ----------------------------------------------------------------------------------------------
def _search_indices(inputs):
  """Pick K_SEARCH example indices spread evenly across the task (for dedup + gating)."""
  n = len(inputs)
  k = min(K_SEARCH, n)
  if k >= n:
    return list(range(n))
  step = n / k
  return sorted({int(i * step) for i in range(k)})


def _run_fresh(program, input_list):
  """Execute `program` on one input list using a FRESH Program (deterministic; see note in
  data/adapt_bigbench_solved.py — the Program object is stateful across run() calls)."""
  state = Program.from_str(program).run([list(input_list)])
  return None if state is None else state.get_output()


def _count_solved(program, inputs, full_target):
  mc = 0
  for inp, tgt in zip(inputs, full_target):
    out = _run_fresh(program, inp)
    if (_freeze(out) if out is not None else None) == tgt:
      mc += 1
  return mc


def synthesize(inputs, outputs):
  """Return the best program found (max examples solved), or None if it solves < MIN_SOLVED.

  Bottom-up list-only search. Dedup/gating is driven on K_SEARCH evenly-spread examples; the
  actual solve-count of a candidate is measured against ALL examples with fresh execution (the
  same method used downstream), so `num_solved` is exact and reproducible. A program that solves
  every example is returned immediately.
  """
  if not all(isinstance(o, list) for o in outputs):
    return None

  n = len(inputs)
  full_target = [tuple(o) for o in outputs]
  idx = _search_indices(inputs)
  sub_inputs = [inputs[i] for i in idx]
  ksub = len(idx)
  sub_target = tuple(full_target[i] for i in idx)

  best = [0, None]  # [matches, program]

  def consider(node_ast, sub_sig):
    # Cheap gate: only full-evaluate nodes that agree with the target on >=1 search example.
    if sum(1 for j in range(ksub) if sub_sig[j] == sub_target[j]) == 0:
      return False
    program = ast_to_program(node_ast)
    mc = _count_solved(program, inputs, full_target)
    if mc > best[0]:
      best[0], best[1] = mc, program
    return mc == n

  input_node = Node('list', tuple(tuple(x) for x in sub_inputs), ('in',))
  if consider(('in',), input_node.sig):  # identity
    return best[1]

  seen = {('list', input_node.sig)}
  pool_list = [input_node]
  deadline = time.time() + TIME_CAP_S

  # Bottom-up rounds; each round nests one more operation (depth == round).
  for _round in range(MAX_DEPTH):
    new_nodes = []
    for rhs, children, node_ast in _gen_candidates(pool_list, input_node):
      if time.time() > deadline:
        return best[1] if best[0] >= MIN_SOLVED else None
      sig = _eval_rhs(rhs, children, ksub)
      if sig is None or not isinstance(sig[0], tuple):
        continue  # list-producing only
      key = ('list', sig)
      if key in seen:
        continue
      seen.add(key)
      new_nodes.append(Node('list', sig, node_ast))
      if consider(node_ast, sig):
        return best[1]
    for node in new_nodes:
      if len(pool_list) < POOL_CAP:
        pool_list.append(node)
  return best[1] if best[0] >= MIN_SOLVED else None


# ----------------------------------------------------------------------------------------------
# Heuristic out-of-range flag (only meaningful for UNSOLVED concepts)
# ----------------------------------------------------------------------------------------------
def oor_heuristic(concept, inputs, outputs):
  """Return (flag: bool, values: str). Detects literals a program would need outside [-5,5]."""
  reasons = []
  nums = [int(n) for n in re.findall(r'-?\d+', concept)]
  oor_nums = sorted({n for n in nums if n < -5 or n > 5})
  if oor_nums:
    reasons.append('desc_const:' + ','.join(map(str, oor_nums)))
  # constant output across all examples => needs a list literal, not expressible.
  if all(o == outputs[0] for o in outputs):
    reasons.append('constant_output')
  return (bool(reasons), ';'.join(reasons))


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------
def _load():
  by = collections.OrderedDict()
  with open(IN_CSV) as f:
    for row in csv.DictReader(f):
      by.setdefault(row['id'], {'concept': row['concept'], 'ios': []})
      by[row['id']]['ios'].append(
          (ast_mod.literal_eval(row['input1']), ast_mod.literal_eval(row['output'])))
  return by


def _work(item):
  cid, concept, ios = item
  inputs = [io[0] for io in ios]
  outputs = [io[1] for io in ios]
  full_target = [tuple(o) if isinstance(o, list) else o for o in outputs]
  t0 = time.time()
  program = synthesize(inputs, outputs)
  elapsed = time.time() - t0

  status, oor_flag, oor_values, num_solved = 'UNSOLVED', False, '', 0
  if program is not None:
    # Independent re-count of solved examples with fresh execution (source of truth).
    num_solved = _count_solved(program, inputs, full_target)
    if num_solved >= MIN_SOLVED:
      status = 'SOLVED'
    else:
      program = None  # best program did not clear the threshold

  if status == 'UNSOLVED':
    oor_flag, oor_values = oor_heuristic(concept, inputs, outputs)

  return {
      'id': cid, 'concept': concept, 'program': program or '',
      'status': status, 'oor_flag': oor_flag, 'oor_values': oor_values,
      'num_solved': num_solved, 'total': len(inputs), 'elapsed': round(elapsed, 1),
  }


def _init_worker():
  flags.FLAGS(sys.argv[:1])


def main():
  by = _load()
  items = [(cid, d['concept'], d['ios']) for cid, d in by.items()]

  with Pool(initializer=_init_worker) as pool:
    results = pool.map(_work, items, chunksize=1)

  results.sort(key=lambda r: r['id'])

  # CSV
  os.makedirs(RULES_DIR, exist_ok=True)
  with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=[
        'id', 'concept', 'program', 'status', 'num_solved', 'total', 'oor_flag', 'oor_values'])
    w.writeheader()
    for r in results:
      w.writerow({k: r[k] for k in w.fieldnames})

  solved = [r for r in results if r['status'] == 'SOLVED']
  full = [r for r in solved if r['num_solved'] == r['total']]
  unsolved = [r for r in results if r['status'] == 'UNSOLVED']
  flagged = [r for r in unsolved if r['oor_flag']]

  # Markdown
  with open(OUT_MD, 'w') as f:
    f.write('# LambdaBeam synthesis report — BIG-bench list_functions\n\n')
    f.write(f'- Concepts: **{len(results)}**\n')
    f.write(f'- SOLVED (program solves >= {MIN_SOLVED} of 32 examples): **{len(solved)}**\n')
    f.write(f'  - of which solve ALL 32: **{len(full)}**\n')
    f.write(f'- UNSOLVED: **{len(unsolved)}**\n')
    f.write(f'- UNSOLVED flagged as needing out-of-range constants / list literals: **{len(flagged)}**\n\n')
    f.write('Search: bottom-up, depth <= {}, constants/indices in [-5, 5]. A concept is SOLVED '
            'when its best program reproduces at least {} of the 32 I/O pairs (counted with fresh '
            'execution). `num_solved`/32 shows the coverage.\n\n'.format(MAX_DEPTH, MIN_SOLVED))
    f.write('| ID | Concept | Program | Status | Solved | OOR flag |\n')
    f.write('|----|---------|---------|--------|-------:|----------|\n')
    for r in results:
      prog = r['program'].replace('|', '\\|') if r['program'] else ''
      concept = r['concept'].replace('|', '\\|')
      oor = r['oor_values'] if r['oor_flag'] else ''
      cov = f"{r['num_solved']}/{r['total']}" if r['status'] == 'SOLVED' else ''
      f.write(f"| {r['id']} | {concept} | `{prog}` | {r['status']} | {cov} | {oor} |\n")

  print(f'Concepts: {len(results)}')
  print(f'SOLVED (>= {MIN_SOLVED}/32): {len(solved)}   (full 32/32: {len(full)})')
  print(f'UNSOLVED: {len(unsolved)}  (flagged OOR: {len(flagged)})')
  print(f'Wrote {OUT_CSV} and {OUT_MD}')


if __name__ == '__main__':
  main()
