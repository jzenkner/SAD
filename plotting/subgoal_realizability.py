"""Utilities for analyzing predicted subgoal realizability in DeepCoder logs."""

from __future__ import annotations

import collections
import itertools
import json
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import pandas as pd
from absl import flags


_THIS_FILE = pathlib.Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from tasks.deepcoder import deepcoder_dsl as dsl
from tasks.lambdabeam import lambdabeam_dsl
from tasks import operation_base
from tasks import value as value_module


def _ensure_absl_flags_parsed() -> None:
  if not flags.FLAGS.is_parsed():
    flags.FLAGS(['subgoal_realizability'])


_ensure_absl_flags_parsed()


@dataclass(frozen=True)
class Witness:
  step_index: int
  statement: str


@dataclass(frozen=True)
class AtomicProgramTemplate:
  """A cached one-step DeepCoder program template.

  `arg_refs` contains either:
    * an integer index into the current state's variables, or
    * a lambda token string such as '(+1)'.
  """
  operation_token: str
  arg_refs: tuple[Any, ...]


_ATOMIC_TEMPLATE_CACHE: dict[tuple[str, ...], list[AtomicProgramTemplate]] = {}
_LAMBDABEAM_CONSTANTS = tuple(
    range(lambdabeam_dsl.lambdabeam_min_const(),
          lambdabeam_dsl.lambdabeam_max_const() + 1))


def _get_dsl_module(dataset_type: str):
  if dataset_type == 'deepcoder':
    return dsl
  if dataset_type == 'lambdabeam':
    return lambdabeam_dsl
  raise ValueError(f'Unsupported dataset_type: {dataset_type}')


def _fresh_variable(existing_variables: list[str]) -> str:
  """Returns a fresh xN variable name without relying on the 10-variable limit."""
  max_index = -1
  for variable in existing_variables:
    match = re.fullmatch(r'x(\d+)', variable)
    if match:
      max_index = max(max_index, int(match.group(1)))
  return f'x{max_index + 1}'


def parse_result_string(result_str: str) -> dsl.ResultType:
  """Parses strings like '[ -8 -8 ]' or '3' into Python values."""
  parsed = dsl.str_to_result(result_str.strip())
  if not dsl.validate_result(parsed):
    raise ValueError(f'Could not parse valid DeepCoder result from: {result_str!r}')
  return parsed


def parse_state_string(state_str: str) -> dsl.ProgramState:
  """Parses one example state like 'x0 = [ 1 2 ] | x1 = 3'."""
  return dsl.ProgramState.from_str(state_str)


def _parse_result_string_for_dataset(dataset_type: str, result_str: str):
  dsl_module = _get_dsl_module(dataset_type)
  parsed = dsl_module.str_to_result(result_str.strip())
  if not dsl_module.validate_result(parsed):
    raise ValueError(
        f'Could not parse valid {dataset_type} result from: {result_str!r}')
  return parsed


def _parse_state_string_for_dataset(dataset_type: str, state_str: str):
  return _get_dsl_module(dataset_type).ProgramState.from_str(state_str)


def normalize_predicted_subgoal(prediction: Any) -> Optional[list[str]]:
  """Normalizes a logged subgoal prediction to per-example output strings."""
  if prediction == '[finished]':
    return None
  if isinstance(prediction, list):
    return prediction
  if isinstance(prediction, str):
    pieces = [piece.strip() for piece in prediction.split('|')]
    return [piece for piece in pieces if piece]
  raise TypeError(f'Unhandled prediction type: {type(prediction)}')


def _lambda_choices(lambda_type: dsl.LambdaType) -> list[dsl.Lambda]:
  return [
      lam for lam in dsl.LAMBDAS
      if (lam.inputs_type, lam.output_type) == lambda_type
  ]


def _state_signature(state: dsl.ProgramState) -> tuple[str, ...]:
  return tuple(type(value).__name__ for value in state.state)


def _validate_state_shapes(current_states: list[dsl.ProgramState]) -> None:
  if not current_states:
    raise ValueError('Need at least one current state.')
  reference_variables = current_states[0].variables
  reference_signature = _state_signature(current_states[0])
  for state in current_states[1:]:
    if state.variables != reference_variables:
      raise ValueError(
          'All examples must have the same variable layout. '
          f'Expected {reference_variables}, got {state.variables}.')
    if _state_signature(state) != reference_signature:
      raise ValueError(
          'All examples must have the same variable types. '
          f'Expected {reference_signature}, got {_state_signature(state)}.')


def _state_shapes_are_consistent(current_states: list[Any]) -> tuple[bool, Optional[str]]:
  try:
    _validate_state_shapes(current_states)
    return True, None
  except ValueError as exc:
    return False, str(exc)


def _atomic_program_templates_for_signature(
    signature: tuple[str, ...],
) -> list[AtomicProgramTemplate]:
  cached = _ATOMIC_TEMPLATE_CACHE.get(('deepcoder',) + signature)
  if cached is not None:
    return cached

  index_dict = collections.defaultdict(list)
  for index, type_name in enumerate(signature):
    value_type = int if type_name == 'int' else list
    index_dict[value_type].append(index)

  templates = []
  for op in dsl.OPERATIONS:
    arg_choices = []
    valid = True
    for arg_type in op.inputs_type:
      if isinstance(arg_type, tuple):
        choices = [lam.token for lam in _lambda_choices(arg_type)]
      else:
        choices = index_dict[arg_type]
      if not choices:
        valid = False
        break
      arg_choices.append(choices)
    if not valid:
      continue
    for arg_refs in itertools.product(*arg_choices):
      templates.append(AtomicProgramTemplate(
          operation_token=op.token,
          arg_refs=tuple(arg_refs),
      ))

  _ATOMIC_TEMPLATE_CACHE[('deepcoder',) + signature] = templates
  return templates


def _lambdabeam_lambda_arg_choices(
    lambda_op: operation_base.OperationBase,
    num_bound_variables: int,
    state_signature: tuple[str, ...],
) -> list[tuple[Any, ...]]:
  """Enumerates lambda argument patterns for LambdaBeam higher-order ops."""
  if lambda_op.arity < num_bound_variables:
    return []

  int_var_refs = [
      f'x{idx}' for idx, type_name in enumerate(state_signature)
      if type_name == 'int'
  ]

  free_vars = [
      value_module.FreeVariable(f'v{i + 1}')
      for i in range(num_bound_variables)
  ]

  choices = []
  for bound_positions in itertools.combinations(range(lambda_op.arity),
                                                num_bound_variables):
    remaining_choices = []
    free_var_iter = iter(free_vars)
    for arg_index, arg_type in enumerate(lambda_op.inputs_type):
      if arg_index in bound_positions:
        remaining_choices.append([next(free_var_iter)])
      elif arg_type is int:
        constant_values = [
            value_module.ConstantValue(constant)
            for constant in _LAMBDABEAM_CONSTANTS
        ]
        int_variable_values = [
            value_module.ConstantValue(var_name) for var_name in int_var_refs
        ]
        remaining_choices.append(constant_values + int_variable_values)
      else:
        remaining_choices.append([])
    if any(not c for c in remaining_choices):
      continue
    choices.extend(itertools.product(*remaining_choices))
  return choices


def _lambdabeam_higher_order_arg_choices(
    op: operation_base.OperationBase,
    index_dict: dict[type, list[int]],
    state_signature: tuple[str, ...],
) -> list[list[Any]]:
  arg_choices = []
  for arg_index, arg_type in enumerate(op.inputs_type):
    if isinstance(arg_type, tuple):
      num_bound_variables = op.num_bound_variables[arg_index]
      lambda_values = []
      for lambda_op in lambdabeam_dsl.TOKEN_TO_LAMBDA.values():
        if (lambda_op.inputs_type, lambda_op.output_type) != arg_type:
          continue
        for lambda_args in _lambdabeam_lambda_arg_choices(
            lambda_op, num_bound_variables, state_signature):
          free_variables = [
              value_module.FreeVariable(f'v{i + 1}')
              for i in range(num_bound_variables)
          ]
          lambda_value = lambda_op.apply(
              list(lambda_args), free_variables=free_variables)
          if lambda_value is not None:
            lambda_values.append(lambda_value)
      arg_choices.append(lambda_values)
    elif arg_type is int:
      constants = [f'const:{constant}' for constant in _LAMBDABEAM_CONSTANTS]
      arg_choices.append(index_dict[int] + constants)
    else:
      arg_choices.append(index_dict[arg_type])
  return arg_choices


def _lambdabeam_atomic_program_templates_for_signature(
    signature: tuple[str, ...],
) -> list[AtomicProgramTemplate]:
  cached = _ATOMIC_TEMPLATE_CACHE.get(('lambdabeam',) + signature)
  if cached is not None:
    return cached

  index_dict = collections.defaultdict(list)
  for index, type_name in enumerate(signature):
    if type_name == 'int':
      index_dict[int].append(index)
    elif type_name == 'list':
      index_dict[list].append(index)

  templates = []
  for op in lambdabeam_dsl.get_operations():
    if op.output_type not in (int, list):
      continue
    arg_choices = _lambdabeam_higher_order_arg_choices(
        op, index_dict, signature)
    if any(not choices for choices in arg_choices):
      continue
    for arg_refs in itertools.product(*arg_choices):
      templates.append(AtomicProgramTemplate(
          operation_token=op.token,
          arg_refs=tuple(arg_refs),
      ))

  _ATOMIC_TEMPLATE_CACHE[('lambdabeam',) + signature] = templates
  return templates


def _instantiate_program_from_template(
    template: AtomicProgramTemplate,
    state: dsl.ProgramState,
) -> dsl.Program:
  output_variable = _fresh_variable(state.variables)
  args = []
  for arg_ref in template.arg_refs:
    if isinstance(arg_ref, int):
      args.append(state.variables[arg_ref])
    else:
      args.append(dsl.TOKEN_TO_LAMBDA[arg_ref])
  statement = dsl.Statement(
      variable=output_variable,
      operation=dsl.TOKEN_TO_OPERATION[template.operation_token],
      args=args,
  )
  return dsl.Program(input_variables=state.variables, statements=[statement])


def _instantiate_lambdabeam_program_from_template(
    template: AtomicProgramTemplate,
    state: lambdabeam_dsl.ProgramState,
) -> lambdabeam_dsl.Program:
  output_variable = _fresh_variable(state.variables)
  args = []
  for arg_ref in template.arg_refs:
    if isinstance(arg_ref, int):
      args.append(state.variables[arg_ref])
    elif isinstance(arg_ref, str) and arg_ref.startswith('const:'):
      args.append(value_module.ConstantValue(int(arg_ref.split(':', 1)[1])))
    else:
      args.append(arg_ref)
  statement = lambdabeam_dsl.Statement(
      variable=output_variable,
      operation=lambdabeam_dsl.TOKEN_TO_OPERATION[template.operation_token],
      args=args,
  )
  return lambdabeam_dsl.Program(
      input_variables=state.variables, statements=[statement])


def _run_atomic_program_on_states(
    template: AtomicProgramTemplate,
    current_states: list[dsl.ProgramState],
) -> Optional[list[dsl.ResultType]]:
  """Executes one atomic program template on all current states."""
  program = _instantiate_program_from_template(template, current_states[0])
  outputs = []
  for state in current_states:
    result_state = program.run(state.state)
    if result_state is None:
      return None
    outputs.append(result_state.get_output())
  return outputs


def _run_atomic_program_on_states_for_dataset(
    dataset_type: str,
    template: AtomicProgramTemplate,
    current_states: list[Any],
) -> Optional[list[Any]]:
  if dataset_type == 'deepcoder':
    return _run_atomic_program_on_states(template, current_states)

  program = _instantiate_lambdabeam_program_from_template(
      template, current_states[0])
  outputs = []
  for state in current_states:
    result_state = program.run(state.state)
    if result_state is None:
      return None
    outputs.append(result_state.get_output())
  return outputs


def _instantiate_program_for_dataset(
    dataset_type: str,
    template: AtomicProgramTemplate,
    state: Any,
):
  if dataset_type == 'deepcoder':
    return _instantiate_program_from_template(template, state)
  return _instantiate_lambdabeam_program_from_template(template, state)


def _atomic_program_templates_for_dataset(
    dataset_type: str,
    signature: tuple[str, ...],
) -> list[AtomicProgramTemplate]:
  if dataset_type == 'deepcoder':
    return _atomic_program_templates_for_signature(signature)
  if dataset_type == 'lambdabeam':
    return _lambdabeam_atomic_program_templates_for_signature(signature)
  raise ValueError(f'Unsupported dataset_type: {dataset_type}')


def enumerate_single_step_witnesses(
    current_states: list[dsl.ProgramState],
    target_outputs: list[dsl.ResultType],
    max_witnesses: Optional[int] = 10,
) -> list[Witness]:
  """Returns atomic DSL programs whose execution matches the target."""
  if len(current_states) != len(target_outputs):
    raise ValueError(
        f'Need one target per example, got {len(current_states)} states and '
        f'{len(target_outputs)} targets.')
  _validate_state_shapes(current_states)

  templates = _atomic_program_templates_for_signature(
      _state_signature(current_states[0]))
  witnesses = []
  for template in templates:
    outputs = _run_atomic_program_on_states(template, current_states)
    if outputs == target_outputs:
      program = _instantiate_program_from_template(template, current_states[0])
      witnesses.append(Witness(
          step_index=-1,
          statement=str(program.statements[0]),
      ))
      if max_witnesses is not None and len(witnesses) >= max_witnesses:
        return witnesses
  return witnesses


def append_subgoal_as_new_variable(
    current_states: list[dsl.ProgramState],
    predicted_outputs: list[dsl.ResultType],
) -> list[dsl.ProgramState]:
  """Appends the predicted subgoal outputs as a new variable for the next step."""
  if len(current_states) != len(predicted_outputs):
    raise ValueError(
        f'Need one predicted output per example, got {len(current_states)} '
        f'states and {len(predicted_outputs)} outputs.')
  new_variable = _fresh_variable(current_states[0].variables)
  next_states = []
  for state, output in zip(current_states, predicted_outputs):
    state_copy = state.copy()
    state_copy.add_result(output, new_variable)
    next_states.append(state_copy)
  return next_states


def analyze_task_subgoals(
    task: dict[str, Any],
    use_synth_success_proxy: bool = False,
    max_witnesses: Optional[int] = 3,
) -> list[dict[str, Any]]:
  """Analyzes every predicted subgoal in one task log entry.

  Note:
    The JSON log contains exact per-step subgoal predictions, so realizability
    and syntactic drift are exact. It does not contain exact per-step
    synthesizer success for the *predicted* subgoal. When
    `use_synth_success_proxy=True`, we derive a simple proxy from the aggregate
    `synth_subgoal_success` count by treating the first `k` steps as successes.
    This is useful for rough plots, but it is not an exact procedural-mismatch
    measurement.
  """
  current_states = [parse_state_string(example) for example in task['inputs']]
  synth_success_budget = int(task.get('synth_subgoal_success', 0) or 0)
  rows = []

  for step_index, raw_prediction in enumerate(task.get('subgoals', [])):
    normalized_prediction = normalize_predicted_subgoal(raw_prediction)
    if normalized_prediction is None:
      rows.append({
          'test_example_index': task['test_example_index'],
          'step_index': step_index,
          'predicted_subgoal': raw_prediction,
          'is_finished_marker': True,
          'realizable': False,
          'num_witnesses': 0,
          'witnesses': [],
          'classification': 'finished',
          'classification_proxy': 'finished',
      })
      break

    target_outputs = [parse_result_string(output) for output in normalized_prediction]
    witnesses = enumerate_single_step_witnesses(
        current_states,
        target_outputs,
        max_witnesses=max_witnesses,
    )
    realizable = bool(witnesses)

    if not realizable:
      classification = 'class_3_syntactic_drift'
      classification_proxy = classification
    else:
      classification = 'realizable_unresolved'
      if use_synth_success_proxy:
        synth_success_proxy = step_index < synth_success_budget
        classification_proxy = (
            'class_1_realizable_synth_success_proxy'
            if synth_success_proxy else
            'class_2_realizable_synth_failure_proxy'
        )
      else:
        classification_proxy = 'realizable'

    rows.append({
        'test_example_index': task['test_example_index'],
        'step_index': step_index,
        'predicted_subgoal': normalized_prediction,
        'is_finished_marker': False,
        'realizable': realizable,
        'num_witnesses': len(witnesses),
        'witnesses': [w.statement for w in witnesses],
        'classification': classification,
        'classification_proxy': classification_proxy,
    })

    current_states = append_subgoal_as_new_variable(current_states, target_outputs)

  return rows


def analyze_log_file(
    log_file_path: str,
    approach: Optional[str] = None,
    experiment: Optional[str] = None,
    seed: Optional[int] = None,
    use_synth_success_proxy: bool = False,
    max_witnesses: Optional[int] = 3,
) -> pd.DataFrame:
  """Analyzes one results JSON file and returns one row per predicted subgoal."""
  with open(log_file_path, 'r') as f:
    logs = json.load(f)

  rows = []
  for task in logs:
    task_rows = analyze_task_subgoals(
        task,
        use_synth_success_proxy=use_synth_success_proxy,
        max_witnesses=max_witnesses,
    )
    for row in task_rows:
      row['approach'] = approach
      row['experiment'] = experiment
      row['seed'] = seed
      rows.append(row)
  return pd.DataFrame(rows)


def analyze_failed_tasks_atomic(
    log_file_path: str,
    approach: Optional[str] = None,
    experiment: Optional[str] = None,
    seed: Optional[int] = None,
    max_witnesses: Optional[int] = 3,
) -> pd.DataFrame:
  """Analyzes only unsolved tasks using exhaustive atomic-program execution.

  For each failed task and each predicted subgoal:
    * if at least one atomic DSL program realizes the predicted subgoal from the
      current state, the subgoal is counted as realizable-but-synthesizer-fails
    * otherwise it is counted as syntactic drift
  """
  with open(log_file_path, 'r') as f:
    logs = json.load(f)

  rows = []
  for task in logs:
    if task.get('success'):
      continue

    current_states = [parse_state_string(example) for example in task['inputs']]
    _validate_state_shapes(current_states)
    templates = _atomic_program_templates_for_signature(
        _state_signature(current_states[0]))

    for step_index, raw_prediction in enumerate(task.get('subgoals', [])):
      normalized_prediction = normalize_predicted_subgoal(raw_prediction)
      if normalized_prediction is None:
        break

      target_outputs = [parse_result_string(output) for output in normalized_prediction]
      witnesses = []
      for template in templates:
        outputs = _run_atomic_program_on_states(template, current_states)
        if outputs == target_outputs:
          program = _instantiate_program_from_template(template, current_states[0])
          witnesses.append(str(program.statements[0]))
          if max_witnesses is not None and len(witnesses) >= max_witnesses:
            break

      realizable = bool(witnesses)
      classification = (
          'class_2_realizable_but_synth_fails'
          if realizable else
          'class_3_syntactic_drift'
      )
      rows.append({
          'approach': approach,
          'experiment': experiment,
          'seed': seed,
          'test_example_index': task['test_example_index'],
          'step_index': step_index,
          'current_inputs': [str(state) for state in current_states],
          'predicted_subgoal': normalized_prediction,
          'realizable': realizable,
          'classification': classification,
          'num_matching_programs_found': len(witnesses),
          'witnesses': witnesses,
          'num_atomic_program_templates': len(templates),
      })

      current_states = append_subgoal_as_new_variable(current_states, target_outputs)
      _validate_state_shapes(current_states)
      templates = _atomic_program_templates_for_signature(
          _state_signature(current_states[0]))

  return pd.DataFrame(rows)


def _infer_dataset_type_from_path(log_path: str) -> str:
  match = re.search(r'dataset_type=([^,\/]+)', log_path)
  if match:
    return match.group(1)
  if 'deepcoder' in log_path:
    return 'deepcoder'
  if 'lambdabeam' in log_path:
    return 'lambdabeam'
  raise ValueError(f'Could not infer dataset_type from path: {log_path}')


def _extract_metadata_from_result_path(result_path: str) -> dict[str, Any]:
  metadata = {
      'path': result_path,
      'dataset_type': _infer_dataset_type_from_path(result_path),
      'approach': pathlib.Path(result_path).parts[-6]
      if len(pathlib.Path(result_path).parts) >= 6 else None,
      'experiment': None,
      'seed': None,
  }
  exp_match = re.search(r'experiment=([^,\/]+)', result_path)
  seed_match = re.search(r'seed=(\d+)', result_path)
  if exp_match:
    metadata['experiment'] = exp_match.group(1)
  if seed_match:
    metadata['seed'] = int(seed_match.group(1))
  return metadata


def discover_result_files(log_dir: str) -> list[str]:
  """Recursively finds result JSON files under a log directory."""
  root = pathlib.Path(log_dir)
  if root.is_file():
    return [str(root)]
  return sorted(str(path) for path in root.rglob('results-*.json'))


def analyze_failed_tasks_atomic_file(
    log_file_path: str,
    approach: Optional[str] = None,
    experiment: Optional[str] = None,
    seed: Optional[int] = None,
    dataset_type: Optional[str] = None,
    max_witnesses: Optional[int] = 3,
) -> pd.DataFrame:
  """Analyzes one results file for failed tasks only."""
  if dataset_type is None:
    dataset_type = _infer_dataset_type_from_path(log_file_path)

  with open(log_file_path, 'r') as f:
    logs = json.load(f)

  rows = []
  for task in logs:
    if task.get('success'):
      continue

    current_states = [
        _parse_state_string_for_dataset(dataset_type, example)
        for example in task['inputs']
    ]
    for step_index, raw_prediction in enumerate(task.get('subgoals', [])):
      normalized_prediction = normalize_predicted_subgoal(raw_prediction)
      if normalized_prediction is None:
        break

      states_are_consistent, consistency_error = _state_shapes_are_consistent(
          current_states)
      target_outputs = [
          _parse_result_string_for_dataset(dataset_type, output)
          for output in normalized_prediction
      ]
      if states_are_consistent:
        templates = _atomic_program_templates_for_dataset(
            dataset_type, _state_signature(current_states[0]))
        witnesses = []
        for template in templates:
          outputs = _run_atomic_program_on_states_for_dataset(
              dataset_type, template, current_states)
          if outputs == target_outputs:
            program = _instantiate_program_for_dataset(
                dataset_type, template, current_states[0])
            witnesses.append(str(program.statements[0]))
            if max_witnesses is not None and len(witnesses) >= max_witnesses:
              break
      else:
        templates = []
        witnesses = []

      realizable = states_are_consistent and bool(witnesses)
      classification = (
          'class_2_realizable_but_synth_fails'
          if realizable else
          'class_3_syntactic_drift'
      )
      rows.append({
          'dataset_type': dataset_type,
          'approach': approach,
          'experiment': experiment,
          'seed': seed,
          'log_file_path': log_file_path,
          'test_example_index': task['test_example_index'],
          'step_index': step_index,
          'current_inputs': [str(state) for state in current_states],
          'predicted_subgoal': normalized_prediction,
          'realizable': realizable,
          'classification': classification,
          'consistency_error': consistency_error,
          'num_matching_programs_found': len(witnesses),
          'witnesses': witnesses,
          'num_atomic_program_templates': len(templates),
      })

      current_states = append_subgoal_as_new_variable(current_states, target_outputs)

  return pd.DataFrame(rows)


def analyze_failed_tasks_atomic_directory(
    log_dir: str,
    approach: Optional[str] = None,
    dataset_type: Optional[str] = None,
    experiment: Optional[str] = None,
    max_witnesses: Optional[int] = 3,
) -> pd.DataFrame:
  """Recursively analyzes all result files under a directory."""
  result_files = discover_result_files(log_dir)
  frames = []
  for result_file in result_files:
    metadata = _extract_metadata_from_result_path(result_file)
    file_dataset_type = dataset_type or metadata['dataset_type']
    file_experiment = experiment or metadata['experiment']
    frames.append(analyze_failed_tasks_atomic_file(
        result_file,
        approach=approach or metadata['approach'],
        experiment=file_experiment,
        seed=metadata['seed'],
        dataset_type=file_dataset_type,
        max_witnesses=max_witnesses,
    ))
  if not frames:
    return pd.DataFrame()
  return pd.concat(frames, ignore_index=True)


def summarize_across_seeds(
    analysis_df: pd.DataFrame,
    classification_column: str = 'classification',
) -> pd.DataFrame:
  """Aggregates results across seeds with mean/std over per-seed proportions."""
  df = analysis_df.copy()
  df = df[~df[classification_column].isin(['finished'])]
  base_group_cols = [
      col for col in ['dataset_type', 'approach', 'experiment']
      if col in df.columns
  ]
  per_seed_group_cols = base_group_cols + ['seed']

  counts = (
      df.groupby(per_seed_group_cols + [classification_column], dropna=False)
      .size()
      .rename('count')
      .reset_index()
  )
  totals = (
      df.groupby(per_seed_group_cols, dropna=False)
      .size()
      .rename('total')
      .reset_index()
  )
  per_seed = counts.merge(totals, on=per_seed_group_cols, how='left')
  per_seed['proportion'] = per_seed['count'] / per_seed['total']

  summary = (
      per_seed.groupby(base_group_cols + [classification_column], dropna=False)
      .agg(
          mean_count=('count', 'mean'),
          std_count=('count', 'std'),
          mean_total=('total', 'mean'),
          std_total=('total', 'std'),
          mean_proportion=('proportion', 'mean'),
          std_proportion=('proportion', 'std'),
          num_seeds=('seed', 'nunique'),
      )
      .reset_index()
  )

  summary['std_count'] = summary['std_count'].fillna(0.0)
  summary['std_total'] = summary['std_total'].fillna(0.0)
  summary['std_proportion'] = summary['std_proportion'].fillna(0.0)

  # Backward-compatible aliases for downstream notebook code.
  summary['count'] = summary['mean_count']
  summary['total'] = summary['mean_total']
  summary['proportion'] = summary['mean_proportion']
  return summary


def summarize_per_seed(
    analysis_df: pd.DataFrame,
    classification_column: str = 'classification',
) -> pd.DataFrame:
  """Returns per-seed proportions before cross-seed aggregation."""
  df = analysis_df.copy()
  df = df[~df[classification_column].isin(['finished'])]
  group_cols = [
      col for col in ['dataset_type', 'approach', 'experiment', 'seed']
      if col in df.columns
  ]
  counts = (
      df.groupby(group_cols + [classification_column], dropna=False)
      .size()
      .rename('count')
      .reset_index()
  )
  totals = (
      df.groupby(group_cols, dropna=False)
      .size()
      .rename('total')
      .reset_index()
  )
  per_seed = counts.merge(totals, on=group_cols, how='left')
  per_seed['proportion'] = per_seed['count'] / per_seed['total']
  return per_seed


def analyze_approach_directories(
    approach_to_log_dir: dict[str, str],
    dataset_type: Optional[str] = None,
    experiment: Optional[str] = None,
    max_witnesses: Optional[int] = 3,
) -> pd.DataFrame:
  """Analyzes multiple approach directories and concatenates the results."""
  frames = []
  for approach, log_dir in approach_to_log_dir.items():
    frames.append(analyze_failed_tasks_atomic_directory(
        log_dir=log_dir,
        approach=approach,
        dataset_type=dataset_type,
        experiment=experiment,
        max_witnesses=max_witnesses,
    ))
  if not frames:
    return pd.DataFrame()
  return pd.concat(frames, ignore_index=True)


def summarize_by_approach(
    approach_to_log_dir: dict[str, str],
    dataset_type: Optional[str] = None,
    experiment: Optional[str] = None,
    max_witnesses: Optional[int] = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Convenience wrapper returning per-subgoal rows and aggregated summary."""
  analysis_df = analyze_approach_directories(
      approach_to_log_dir=approach_to_log_dir,
      dataset_type=dataset_type,
      experiment=experiment,
      max_witnesses=max_witnesses,
  )
  summary_df = summarize_across_seeds(analysis_df)
  return analysis_df, summary_df


def summarize_analysis(
    analysis_df: pd.DataFrame,
    classification_column: str = 'classification_proxy',
) -> pd.DataFrame:
  """Aggregates subgoal analysis into counts and proportions."""
  df = analysis_df.copy()
  df = df[~df[classification_column].isin(['finished'])]
  group_cols = [col for col in ['approach', 'experiment', 'seed'] if col in df.columns]
  if not group_cols:
    group_cols = [classification_column]

  counts = (
      df.groupby(group_cols + [classification_column], dropna=False)
      .size()
      .rename('count')
      .reset_index()
  )
  totals = (
      df.groupby(group_cols, dropna=False)
      .size()
      .rename('total')
      .reset_index()
  )
  summary = counts.merge(totals, on=group_cols, how='left')
  summary['proportion'] = summary['count'] / summary['total']
  return summary


def print_summary(
    summary_df: pd.DataFrame,
    classification_column: str = 'classification_proxy',
) -> None:
  """Pretty-prints a summary table."""
  for keys, group in summary_df.groupby(
      [col for col in ['approach', 'experiment', 'seed'] if col in summary_df.columns],
      dropna=False,
  ):
    if not isinstance(keys, tuple):
      keys = (keys,)
    label_parts = []
    for name, value in zip(
        [col for col in ['approach', 'experiment', 'seed'] if col in summary_df.columns],
        keys,
    ):
      label_parts.append(f'{name}={value}')
    if label_parts:
      print(', '.join(label_parts))
    for _, row in group.sort_values(classification_column).iterrows():
      print(
          f"  {row[classification_column]}: "
          f"{int(row['count'])} / {int(row['total'])} "
          f"({100 * row['proportion']:.2f}%)"
      )
    if label_parts:
      print()


def analyze_many_log_files(
    specs: Iterable[dict[str, Any]],
    use_synth_success_proxy: bool = False,
    max_witnesses: Optional[int] = 3,
) -> pd.DataFrame:
  """Convenience wrapper for analyzing multiple result files at once.

  Each spec dict should contain:
    * path: path to the JSON log
  and may additionally contain:
    * approach
    * experiment
    * seed
  """
  frames = []
  for spec in specs:
    frames.append(analyze_log_file(
        log_file_path=spec['path'],
        approach=spec.get('approach'),
        experiment=spec.get('experiment'),
        seed=spec.get('seed'),
        use_synth_success_proxy=use_synth_success_proxy,
        max_witnesses=max_witnesses,
    ))
  if not frames:
    return pd.DataFrame()
  return pd.concat(frames, ignore_index=True)
