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

r"""Measure compositional generalization ability of LLMs.

From the repository root:

python spec_decomposition/run_llm_experiment.py \
    --model=favorite_llm --prompt_format=exedec
"""

import json
import multiprocessing.dummy
import os
import sys
import timeit
from typing import Any, Callable
import ollama

from absl import app
from absl import flags
import tqdm

# NOTE: TensorFlow is deliberately not imported here. With --llm_backend=hf this
# process runs PyTorch on the GPU, and TensorFlow in the same process makes
# torch's CUDA allocator segfault.

# pylint: disable=g-import-not-at-top

from spec_decomposition import cached_llm_access
from spec_decomposition import llm_utils

_MODEL = flags.DEFINE_string(
    'model', '',
    'Which model to use.')
_PROMPT_FORMAT = flags.DEFINE_enum(
    'prompt_format',
    'baseline',
    ['baseline', 'exedec', 'exedec_ablation', 'tiips', 'transductively'],
    'Format of the prompt to use.',
)
_TARGET_TASK = flags.DEFINE_enum(
    'task',
    None,
    [
        'NONE',
        'LENGTH_GENERALIZATION',
        'COMPOSE_DIFFERENT_CONCEPTS',
        'SWITCH_CONCEPT_ORDER',
        'COMPOSE_NEW_OP',
        'ADD_OP_FUNCTIONALITY',
    ],
    'If specified, only run one generalization task.',
)

_NUM_SAMPLES = flags.DEFINE_integer(
    'num_samples', 1,
    'Number of samples to draw for one problem.')
_TEMPERATURE = flags.DEFINE_float(
    'temperature', 0.0, 'Temperature for the LLM.'
)
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers', 72, 'Number of workers.',
)

_LLM_CACHE_DIR = flags.DEFINE_string(
    'llm_cache_dir', './llm_cache',
    'Directory for storing the LLM cache.')

# The ICLR'24 paper used version_robustfill=1, version_deepcoder=1, and
# version_deepcoder=4 (DeepCoder-Pythonic in the paper).
_VERSION_DEEPCODER = flags.DEFINE_integer(
    'version_deepcoder', 4, 'Version of Python programs and prompts to use.'
)
_VERSION_ROBUSTFILL = flags.DEFINE_integer(
    'version_robustfill', 1, 'Version of Python programs and prompts to use.'
)

_ABLATION = flags.DEFINE_bool(
  'ablation', False, 'Whether to use the ablation style prompting of ExeDec.'
)

_LLM_BACKEND = flags.DEFINE_enum(
    'llm_backend', 'ollama', ['ollama', 'hf'],
    'Where to run the LLM. "ollama" queries a local Ollama daemon (the '
    'original behaviour); "hf" loads a base model plus LoRA adapters in this '
    'process, which is how fine-tuned decomposers/synthesizers are served.'
)
_BASE_MODEL = flags.DEFINE_string(
    'base_model', 'meta-llama/Llama-3.1-8B',
    'HF model id or path for --llm_backend=hf.'
)
_DECOMPOSER_ADAPTER = flags.DEFINE_string(
    'decomposer_adapter', None,
    'LoRA adapter for the decomposer role. Unset means the untuned base model.'
)
_SYNTHESIZER_ADAPTER = flags.DEFINE_string(
    'synthesizer_adapter', None,
    'LoRA adapter for the synthesizer role. Unset means the untuned base model.'
)
_TWO_STAGE_EXEDEC = flags.DEFINE_bool(
    'two_stage_exedec', False,
    'Split each ExeDec step into two calls: the decomposer proposes the '
    'subgoal, then the synthesizer writes the code conditioned on it. Required '
    'for separately fine-tuned adapters. When False a single call emits both, '
    'as in the original setup.'
)
_MAX_DEC_STEPS = flags.DEFINE_integer(
    'max_dec_steps', 5,
    'Maximum ExeDec decomposition steps. The DeepCoder eval sets contain '
    'programs of up to 4 statements, so the historical value of 3 left some '
    'tasks unsolvable by construction.'
)
_DATASET_TYPES = flags.DEFINE_list(
    'dataset_types', ['robustfill', 'deepcoder'],
    'Which datasets to evaluate. Restrict this to deepcoder when using '
    'adapters that were only fine-tuned on DeepCoder.'
)
_NUM_TEST_PROBLEMS = flags.DEFINE_integer(
    'num_test_problems', None,
    'If set, evaluate only the first N problems of each split. For smoke '
    'testing; leave unset for real numbers.'
)

DATA_FORMAT = 'data/llm_data/{dataset_type}/{generalization_task}.jsonl'
RESULTS_FORMAT = os.path.join(
    os.path.expanduser('./results/LLM_experiments'),
    '{prompt_format}_{model}_{num_samples}-samples_{temperature}-temperature_deepcoder-v{version_deepcoder}_robustfill-v{version_robustfill}.json')
DatasetElement = llm_utils.DatasetElement
ExeDecTrajectory = llm_utils.ExeDecTrajectory


def _max_num_dec_steps() -> int:
  return _MAX_DEC_STEPS.value


def _sample_length(dataset_type: str) -> int:
  """Returns the maximum number of decode steps for a given dataset type."""
  if _PROMPT_FORMAT.value in ['baseline', 'transductively']:
    if dataset_type == 'deepcoder':
      return 150
    elif dataset_type == 'robustfill':
      return 400
    else:
      raise ValueError(f'Unhandled dataset type: {dataset_type}')
  elif _PROMPT_FORMAT.value == 'exedec_ablation':
    if dataset_type == 'deepcoder':
      return 80
    elif dataset_type == 'robustfill':
      return 200
    else:
      raise ValueError(f'Unhandled dataset type: {dataset_type}')
  elif _PROMPT_FORMAT.value == 'exedec':
    if dataset_type == 'deepcoder':
      return 200
    elif dataset_type == 'robustfill':
      return 400
    else:
      raise ValueError(f'Unhandled dataset type: {dataset_type}')
  else:
    raise ValueError(f'Unhandled prompt format: {_PROMPT_FORMAT.value}')


def query_llm(
    prompt: str,
    n: int,
    temperature: float,
    model: str,
    num_output_tokens: int
) -> list[str]:
  """Queries an LLM with the given prompt, drawing n samples via Ollama.
 
  Notes:
    - Make sure the Ollama daemon is running (defaults to http://127.0.0.1:11434).
    - Ensure the model is available locally (e.g., `ollama pull <model>`).
    - Set OLLAMA_HOST env var if your server isn't on localhost.
  """
  # Map our arguments to Ollama's options.
  options = {"temperature": float(temperature)}
  if num_output_tokens and num_output_tokens > 0:
    # Ollama uses `num_predict` for the max number of generated tokens.
    options["num_predict"] = int(num_output_tokens)
 
  outputs: list[str] = []
  for _ in range(int(n)):
    res = ollama.generate(
      model=model,
      prompt=prompt,
      options=options,
      # Keep the model in memory briefly so repeated calls are faster.
      keep_alive="5m"
    )
    # `response` contains the generated text.
    outputs.append(res["response"])
 
  return outputs


def _query_fn() -> Callable[..., list[str]]:
  """The backend `cached_llm_access.query_llm` should call."""
  if _LLM_BACKEND.value == 'hf':
    from spec_decomposition.llm_finetune import hf_backend
    return hf_backend.query_llm
  return query_llm


def _sample_step(prompt: str, num_output_tokens: int, role: str = None) -> str:
  """Draws one continuation for one ExeDec step.

  `role` selects the LoRA adapter under the `hf` backend and is ignored by
  Ollama, which serves a single model.
  """
  kwargs = {'role': role} if _LLM_BACKEND.value == 'hf' else {}
  return cached_llm_access.query_llm(
      _query_fn(),
      prompt,
      n=1,  # For step-by-step, we generate one solution at a time.
      temperature=_TEMPERATURE.value,
      model=_MODEL.value,
      num_output_tokens=num_output_tokens,
      **kwargs,
  )[0]


def solve_problem_baseline(
    problem_index: int,
    few_shot_examples: list[DatasetElement],
    test_problem: DatasetElement,
    dataset_type: str,
    num_output_tokens: int,
    verbose: bool = False,
    ablation_style: bool = False,
) -> dict[str, Any]:
  """Solve a problem with baseline prompt."""
  del ablation_style
  start_time = timeit.default_timer()
  if dataset_type == 'robustfill':
    version = _VERSION_ROBUSTFILL.value
  elif dataset_type == 'deepcoder':
    version = _VERSION_DEEPCODER.value
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')

  prompt = llm_utils.few_shot_prompt(
      few_shot_examples,
      test_problem,
      dataset_type=dataset_type,
      version=version,
      transductive=False
  )
  samples = cached_llm_access.query_llm(
      query_llm,
      prompt,
      n=_NUM_SAMPLES.value,
      temperature=_TEMPERATURE.value,
      model=_MODEL.value,
      num_output_tokens=num_output_tokens,
  )
  success = False
  for sample in samples:
    sample = llm_utils.cut_program_from_sample(sample)
    try:
      outputs = llm_utils.run_program(
          sample, test_problem.inputs, dataset_type=dataset_type
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      outputs = None

    if outputs == test_problem.outputs:
      success = True
      break
  elapsed_time = timeit.default_timer() - start_time
  result = {
      'index': problem_index,
      'test_problem': test_problem,
      'samples': samples,
      'success': success,
      'elapsed_time': elapsed_time,
  }

  print(
      f'  Test problem #{problem_index}: '
      f'{"SUCCESS" if result["success"] else "fail"}',
      f'\n\tPrediction: {sample}',
      f'\n\tGround Truth: {test_problem.dsl_program}',
      flush=True,
  )
  return result


def solve_problem_transductively(
    problem_index: int,
    few_shot_examples: list[DatasetElement],
    test_problem: DatasetElement,
    dataset_type: str,
    num_output_tokens: int,
    verbose: bool = False,
    ablation_style: bool = False,
) -> dict[str, Any]:
  """Solve a problem with transductive prompt."""
  del ablation_style
  start_time = timeit.default_timer()
  if dataset_type == 'robustfill':
    version = _VERSION_ROBUSTFILL.value
  elif dataset_type == 'deepcoder':
    version = _VERSION_DEEPCODER.value
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')

  prompt = llm_utils.few_shot_prompt(
      few_shot_examples,
      test_problem,
      dataset_type=dataset_type,
      version=version,
      transductive=True
  )

  samples = cached_llm_access.query_llm(
      query_llm,
      prompt,
      n=_NUM_SAMPLES.value,
      temperature=_TEMPERATURE.value,
      model=_MODEL.value,
      num_output_tokens=num_output_tokens,
  )
  success = False
  print(test_problem.dsl_program)

  for sample in samples:
    sample = llm_utils.cut_program_from_sample(sample, transductive=True)
    if sample == str(test_problem.outputs[-1]):
      success = True
      break
  elapsed_time = timeit.default_timer() - start_time
  result = {
      'index': problem_index,
      'test_problem': test_problem,
      'samples': samples,
      'success': success,
      'elapsed_time': elapsed_time,
  }
  print(
      f'  Test problem #{problem_index}: '
      f'{"SUCCESS" if result["success"] else "fail"}',
      '\n',
      f'{sample if result["success"] else f"Prediction: {sample}, Ground Truth: {test_problem.outputs[-1]}"}',
      flush=True,
  )
  return result


def solve_problem_exedec(
    problem_index: int,
    few_shot_examples: list[DatasetElement],
    test_problem: DatasetElement,
    dataset_type: str,
    num_output_tokens: int,
    verbose: bool = False,
    ablation_style: bool = False,
) -> dict[str, Any]:
  """Solve a problem with ExeDec prompt."""
  start_time = timeit.default_timer()

  samples = []
  trajectories = []
  success = False
  if dataset_type == 'robustfill':
    version = _VERSION_ROBUSTFILL.value
  elif dataset_type == 'deepcoder':
    version = _VERSION_DEEPCODER.value
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')

  sample = None
  for _ in range(_NUM_SAMPLES.value):
    test_problem_wo_solution = DatasetElement(
        test_problem.inputs, test_problem.outputs, None, None
    )
    trajectory = []
    for i in range(_max_num_dec_steps()):
      try:
        prompt = llm_utils.few_shot_exe_dec_prompt(
            few_shot_examples,
            test_problem_wo_solution,
            dataset_type=dataset_type,
            version=version,
            ablation_style=ablation_style,
        )

      except Exception as e:  # pylint: disable=broad-exception-caught
        # Throws error if the previous step does not match target string in
        # RobustFill or any other runtime error during program execution.
        # print(e)
        break

      if i > 0:
        prompt = prompt.rsplit('Putting the steps together', 1)[0]
        if ablation_style:
          prompt = prompt + f'Step {i + 1} code:\n'
        else:
          prompt = prompt + f'Step {i + 1} computes:\n'
      if verbose:
        print('===prompt')
        print(prompt.rsplit('[BEGIN PROBLEM]', 1)[-1])

      subgoal_sample = None
      if _TWO_STAGE_EXEDEC.value:
        if ablation_style:
          raise ValueError(
              '--two_stage_exedec is incompatible with ablation-style prompts, '
              'which put the code before the subgoal.')
        # The decomposer proposes the subgoal, then the synthesizer writes the
        # code for it. The two prompts are cuts of the same string the
        # single-call path uses, so the model sees an identical format.
        from spec_decomposition.llm_finetune import prompts as ft_prompts
        subgoal_sample = _sample_step(
            prompt, num_output_tokens, role='decomposer')
        subgoal = ft_prompts.clean_subgoal_sample(subgoal_sample)
        synth_prompt = ft_prompts.synthesizer_context(prompt, subgoal, i)
        sample = _sample_step(
            synth_prompt, num_output_tokens, role='synthesizer')
        if verbose:
          print('===subgoal:')
          print(subgoal)
      else:
        sample = _sample_step(prompt, num_output_tokens)

      program_step = llm_utils.cut_program_from_sample(sample)
      # Record the full prediction containing the LLM-predicted subgoals.
      # The actual outputs are not recorded but can be recomputed later.
      trajectory.append({
          'subgoal_sample': subgoal_sample,
          'sample': sample,
          'program_step': program_step,
          'prompt': prompt,
      })
      if verbose:
        print('===program_step:')
        print(program_step)
      # We construct a executable program for the steps generated so far.
      program_prefix = test_problem_wo_solution.python_program
      if dataset_type == 'deepcoder':
        if program_prefix is None:
          # If this is the first step:
          # Borrow function signature from standard solution.
          program_prefix = test_problem.python_program.splitlines()[0] + '\n'
        else:
          program_prefix = program_prefix.rsplit('  return', 1)[0]
        new_var = program_step.split('=', 1)[0].strip()
        program_suffix = f'  return {new_var}'
      elif dataset_type == 'robustfill':
        if program_prefix is None:
          program_prefix = 'def program(x):\n  parts = [\n'
        else:
          program_prefix = program_prefix.rsplit('  ]', 1)[0]
        program_suffix = "  ]\n  return ''.join(parts)"
      else:
        raise ValueError(f'Unhandled dataset type: {dataset_type}')
      indent_spaces = '  ' if dataset_type == 'deepcoder' else '    '
      suffix_ = ',\n' if dataset_type == 'robustfill' else '\n'
      compose_program = (
          program_prefix
          + f'{indent_spaces}{program_step.strip()}{suffix_}'
          + program_suffix
      )
      # Update test problem with the new program
      test_problem_wo_solution = DatasetElement(
          test_problem.inputs, test_problem.outputs, None, compose_program
      )
      if verbose:
        print('===compose_program:')
        print(compose_program)
      try:
        outputs = llm_utils.run_program(
            compose_program, test_problem.inputs, dataset_type=dataset_type
        )
      except Exception:  # pylint: disable=broad-exception-caught
        outputs = None

      if outputs == test_problem.outputs:
        success = True
        break  # Stop at the first successful sample

    if verbose:
      print(test_problem_wo_solution.python_program)
    samples.append(test_problem_wo_solution.python_program)
    trajectories.append(trajectory)
    if success:
      break

  elapsed_time = timeit.default_timer() - start_time
  result = {
      'index': problem_index,
      'test_problem': test_problem,
      'samples': samples,
      'trajectories': trajectories,  # For ExeDec only
      'success': success,
      'elapsed_time': elapsed_time,
  }
  print(
      f'  Test problem #{problem_index}: '
      f'{"SUCCESS" if result["success"] else "fail"}',
      f'\n\tPrediction: {sample}',
      f'\n\tGround Truth: {test_problem.dsl_program}',
      flush=True,
  )
  return result


def solve_problem_tiips(
    problem_index: int,
    few_shot_examples: list[DatasetElement],
    test_problem: DatasetElement,
    dataset_type: str,
    num_output_tokens: int,
    verbose: bool = False,
    ablation_style: bool = False,
) -> dict[str, Any]:
  """Solve a problem with tiips prompt."""
  start_time = timeit.default_timer()

  samples = []
  trajectories = []
  success = False
  if dataset_type == 'robustfill':
    version = _VERSION_ROBUSTFILL.value
  elif dataset_type == 'deepcoder':
    version = _VERSION_DEEPCODER.value
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')

  o = 0
  for _ in range(_NUM_SAMPLES.value):
    test_problem_wo_solution = DatasetElement(
        test_problem.inputs, test_problem.outputs, None, None
    )
    trajectory = []
    for o in range(_max_num_dec_steps()):
      trajectory = []
      for i in range(_max_num_dec_steps()):
        try:
          if o > i:
            ablation_style = False
          else:
            ablation_style = True

          prompt = llm_utils.few_shot_exe_dec_prompt(
              few_shot_examples,
              test_problem_wo_solution,
              dataset_type=dataset_type,
              version=version,
              ablation_style=ablation_style,
          )
        except Exception:  # pylint: disable=broad-exception-caught
          # Throws error if the previous step does not match target string in
          # RobustFill or any other runtime error during program execution.
          # print(e)
          break
        if i > 0:
          prompt = prompt.rsplit('Putting the steps together', 1)[0]
          if ablation_style:
            prompt = prompt + f'Step {i + 1} code:\n'
          else:
            prompt = prompt + f'Step {i + 1} computes:\n'
        if verbose:
          print('===prompt')
          print(prompt.rsplit('[BEGIN PROBLEM]', 1)[-1])

        sample = cached_llm_access.query_llm(
            query_llm,
            prompt,
            n=1,  # For step-by-step, we generate one solution at a time
            temperature=_TEMPERATURE.value,
            model=_MODEL.value,
            num_output_tokens=num_output_tokens,
        )[0]
        program_step = llm_utils.cut_program_from_sample(sample)
        # Record the full prediction containing the LLM-predicted subgoals.
        # The actual outputs are not recorded but can be recomputed later.
        trajectory.append({
            'sample': sample,
            'program_step': program_step,
            'prompt': prompt,
        })
        if verbose:
          print('===program_step:')
          print(program_step)
        # We construct a executable program for the steps generated so far.
        program_prefix = test_problem_wo_solution.python_program
        if dataset_type == 'deepcoder':
          if program_prefix is None:
            # If this is the first step:
            # Borrow function signature from standard solution.
            program_prefix = test_problem.python_program.splitlines()[0] + '\n'
          else:
            program_prefix = program_prefix.rsplit('  return', 1)[0]
          new_var = program_step.split('=', 1)[0].strip()
          program_suffix = f'  return {new_var}'
        elif dataset_type == 'robustfill':
          if program_prefix is None:
            program_prefix = 'def program(x):\n  parts = [\n'
          else:
            program_prefix = program_prefix.rsplit('  ]', 1)[0]
          program_suffix = "  ]\n  return ''.join(parts)"
        else:
          raise ValueError(f'Unhandled dataset type: {dataset_type}')
        indent_spaces = '  ' if dataset_type == 'deepcoder' else '    '
        suffix_ = ',\n' if dataset_type == 'robustfill' else '\n'
        compose_program = (
            program_prefix
            + f'{indent_spaces}{program_step.strip()}{suffix_}'
            + program_suffix
        )
        # Update test problem with the new program
        test_problem_wo_solution = DatasetElement(
            test_problem.inputs, test_problem.outputs, None, compose_program
        )
        if verbose:
          print('===compose_program:')
          print(compose_program)
        try:
          outputs = llm_utils.run_program(
              compose_program, test_problem.inputs, dataset_type=dataset_type
          )
        except Exception:  # pylint: disable=broad-exception-caught
          outputs = None
        if outputs == test_problem.outputs:
          success = True
          break  # Stop at the first successful sample
      if success:
        break

    if verbose:
      print(test_problem_wo_solution.python_program)
    samples.append(test_problem_wo_solution.python_program)
    trajectories.append(trajectory)
    if success:
      break

  elapsed_time = timeit.default_timer() - start_time
  result = {
      'index': problem_index,
      'test_problem': test_problem,
      'samples': samples,
      'trajectories': trajectories,  # For ExeDec only
      'success': success,
      'elapsed_time': elapsed_time,
  }
  print(
      f'  Test problem #{problem_index}: '
      f'{"SUCCESS" if result["success"] else "fail"}',
      f'\n\tPredictio using {0} guidance steps:\n{sample}',
      f'\n\tGround Truth: {test_problem.dsl_program}',
      flush=True,
  )
  return result

def solver_parallel_run(func: Callable, inputs, num_workers: int):  # pylint: disable=g-bare-generic
  """Run solvers in parallel with multithreading."""
  # Run the solvers and show metrics.
  total_cnt, pass_cnt = 0, 0
  with multiprocessing.dummy.Pool(processes=num_workers) as p:
    num_inputs = len(inputs)
    results: list[Any] = [{}] * num_inputs
    with tqdm.tqdm(total=num_inputs) as pbar:
      for res in p.starmap(func, inputs):
        # Update cummulated metrics.
        total_cnt += 1
        pass_cnt += int(res['success'])
        pass_rate = pass_cnt / total_cnt
        pbar.update()
        pbar.set_postfix({
            'pass_cnt': pass_cnt,
            'pass_rate': pass_rate,
            'total': total_cnt,
        })
        results[res['index']] = res
  return results


def run_experiment(
    dataset_type: str,
    generalization_task: str,
    verbose: bool = False,
    parallel: bool = True,
) -> list[dict[str, Any]]:
  """Runs the experiment for a generalization task."""
  print(f'Running experiment for {dataset_type} {generalization_task}...',
        flush=True)
  if dataset_type == 'robustfill':
    version = _VERSION_ROBUSTFILL.value
  elif dataset_type == 'deepcoder':
    version = _VERSION_DEEPCODER.value
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')
  dataset = llm_utils.load_jsonl_dataset(
      dataset_type=dataset_type,
      generalization_task=generalization_task,
      data_format=DATA_FORMAT,
      version=version,
  )
  if _NUM_TEST_PROBLEMS.value is not None:
    dataset = dataset[:_NUM_TEST_PROBLEMS.value]
  num_test = len(dataset)
  print(f'Loaded {num_test} test problems', flush=True)

  results = []
  num_output_tokens =  10000 # if _PROMPT_FORMAT.value == 'tiips' else _sample_length(dataset_type)
  prompt_format = _PROMPT_FORMAT.value
  ablation_style = prompt_format == 'exedec_ablation'

  solver_inputs = []
  for problem_index in range(num_test):

    test_problem, few_shot_examples = dataset[problem_index]
    solver_inputs.append([
        problem_index,
        few_shot_examples,
        test_problem,
        dataset_type,
        num_output_tokens,
        verbose,
        ablation_style,
    ])
  if not parallel:
    for solver_input in solver_inputs:
      if prompt_format == 'baseline':
        problem_result = solve_problem_baseline(*solver_input)
      elif prompt_format == 'exedec':
        problem_result = solve_problem_exedec(*solver_input)
      elif prompt_format == 'tiips':
        problem_result = solve_problem_tiips(*solver_input)
      elif prompt_format == 'exedec_ablation':
        problem_result = solve_problem_exedec(*solver_input)
      elif prompt_format == 'transductively':
        problem_result = solve_problem_transductively(*solver_input)
      else:
        raise ValueError(f'Unhandled prompt format: {prompt_format}')

      results.append(problem_result)
  else:
    num_workers = _NUM_WORKERS.value
    if _LLM_BACKEND.value == 'hf' and num_workers != 1:
      # The hf backend is one CUDA model in this process; concurrent threads
      # would interleave adapter switches and corrupt each other's generations.
      print('Forcing --num_workers=1 for the hf backend.', flush=True)
      num_workers = 1
    if prompt_format == 'baseline':
      results = solver_parallel_run(
          solve_problem_baseline, solver_inputs, num_workers
      )
    elif prompt_format == 'exedec':
      results = solver_parallel_run(
          solve_problem_exedec, solver_inputs, num_workers
      )
    elif prompt_format == 'tiips':
      results = solver_parallel_run(
          solve_problem_tiips, solver_inputs, num_workers
      )
    elif prompt_format == 'exedec_ablation':
      results = solver_parallel_run(
          solve_problem_exedec, solver_inputs, num_workers
      )
    elif prompt_format == 'transductively':
         results = solver_parallel_run(
          solve_problem_transductively, solver_inputs, num_workers
      )
    else:
      raise ValueError(f'Unhandled prompt format: {prompt_format}')

  num_success = sum(r['success'] for r in results)
  print(f'  Solved {num_success} / {len(results)} problems', flush=True)
  return results


def run_entire_experiment() -> dict[str, dict[str, list[dict[str, Any]]]]:
  """Runs the experiment for all datasets and generalization tasks."""
  # Perform experiment.
  all_results = {}
  for dataset_type in _DATASET_TYPES.value:
    all_results[dataset_type] = {}
    for generalization_task in [
        'NONE',
        'LENGTH_GENERALIZATION',
        'COMPOSE_DIFFERENT_CONCEPTS',
        'SWITCH_CONCEPT_ORDER',
        'COMPOSE_NEW_OP',
        'ADD_OP_FUNCTIONALITY',
    ]:
      if _TARGET_TASK.value is not None:
        if generalization_task != _TARGET_TASK.value:
          continue
      results = run_experiment(dataset_type, generalization_task, verbose=False, parallel=True)
      all_results[dataset_type][generalization_task] = results

  # Write actual results files.

  results_path = RESULTS_FORMAT.format(
      prompt_format=_PROMPT_FORMAT.value,
      model=_MODEL.value.split('/')[-1],
      num_samples=_NUM_SAMPLES.value,
      temperature=_TEMPERATURE.value,
      version_deepcoder=_VERSION_DEEPCODER.value,
      version_robustfill=_VERSION_ROBUSTFILL.value,
  )
  print(f'Writing results to {results_path}...', flush=True)
  os.makedirs(os.path.dirname(results_path), exist_ok=True)
  with open(results_path, 'w') as f:
    json.dump(all_results, f)
  print('Experiment done!')

  return all_results


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if _TEMPERATURE.value == 0.0:
    assert _NUM_SAMPLES.value == 1

  if _TWO_STAGE_EXEDEC.value and _PROMPT_FORMAT.value != 'exedec':
    raise app.UsageError(
        '--two_stage_exedec only applies to --prompt_format=exedec.')

  if _LLM_BACKEND.value == 'hf':
    from spec_decomposition.llm_finetune import hf_backend
    hf_backend.init(
        base_model=_BASE_MODEL.value,
        decomposer_adapter=_DECOMPOSER_ADAPTER.value,
        synthesizer_adapter=_SYNTHESIZER_ADAPTER.value,
    )

  cache_dir = os.path.expanduser(_LLM_CACHE_DIR.value)
  model_name = _MODEL.value.split('/')[-1]
  # The cache keys only on the prompt and temperature, so two adapter
  # checkpoints producing the same prompt would collide. Callers must give each
  # checkpoint its own --llm_cache_dir; see run_llm_finetuned_eval.sh.
  cached_llm_access.init_cache(cache_dir, model_name)

  run_entire_experiment()


if __name__ == '__main__':
  app.run(main)
