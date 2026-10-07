"""Prompt/completion strings for the two ExeDec LLM roles.

The stock ExeDec prompt (`llm_utils.few_shot_exe_dec_prompt`) asks a single model
to emit both halves of a step:

    Step 1 computes:
      Case 1. x1 = [25, 9, 16]
      Case 2. x1 = [4]

    Step 1 code:
    ```python
    x1 = [x ** 2 for x in x0]
    ```

Here the two halves become two separately-trainable roles. The *decomposer*
completes the `Case i.` block after `Step j computes:`; the *synthesizer* is then
given that block plus `Step j code:` and completes the fenced program step. Both
prompts are cuts of the very same string the stock prompt builder produces, so a
model trained on them sees exactly the format the evaluation harness uses.

Everything here except `step_context` is plain string assembly. `step_context`
imports `llm_utils` lazily on purpose: that module pulls in TensorFlow, which the
PyTorch trainers have no use for and should not pay to load.
"""

from typing import Optional

# Where `get_exe_dec_prompt_prefix` stops rendering completed steps.
_PUTTING_TOGETHER = 'Putting the steps together'

# Text that follows a decomposer completion / a synthesizer completion. Used as
# generation stop strings for models that do not emit EOS (e.g. the untuned
# base model), and to trim samples before splicing them into the next prompt.
SUBGOAL_STOP = '\nStep'
CODE_STOP = '\n```'


def partial_python_program(trajectory, num_steps: int) -> Optional[str]:
  """The executable program formed by the first `num_steps` steps.

  Mirrors the composition in `llm_utils.get_exe_dec_trajectory` and in
  `run_llm_experiment.solve_problem_exedec`: the signature line, the step lines
  verbatim (they already carry their indent), then a `return` of the variable
  bound by the last step.

  Returns None for `num_steps == 0`, which is what `get_exe_dec_prompt_prefix`
  expects in order to truncate the prompt at `Step 1 computes:`.
  """
  if num_steps == 0:
    return None
  if num_steps >= len(trajectory):
    raise ValueError(
        f'num_steps={num_steps} exceeds trajectory of length {len(trajectory)}')
  last_step = trajectory[num_steps].python_program_step
  new_var = last_step.strip().split('=', 1)[0].strip()
  return '\n'.join(
      [step.python_program_step for step in trajectory[:num_steps + 1]]
      + [f'  return {new_var}']
  )


def step_context(few_shot_examples,
                 test_problem,
                 dataset_type: str,
                 version: int,
                 step_idx: int,
                 partial_program: Optional[str]) -> str:
  """The decomposer prompt: everything up to and including `Step j computes:`.

  `step_idx` is 0-based; `partial_program` must be the program formed by the
  first `step_idx` steps (None when `step_idx == 0`).

  For `step_idx == 0` the stock prefix already ends at `Step 1 computes:`
  because `python_program` is None. For later steps it renders the completed
  steps from *executed* state and then runs on into the final program, so we cut
  it back the same way `solve_problem_exedec` does.
  """
  from spec_decomposition import llm_utils  # pylint: disable=g-import-not-at-top

  partial_element = llm_utils.DatasetElement(
      test_problem.inputs, test_problem.outputs, None, partial_program)
  prompt = llm_utils.few_shot_exe_dec_prompt(
      few_shot_examples,
      partial_element,
      dataset_type=dataset_type,
      version=version,
      ablation_style=False,
  )
  if step_idx > 0:
    prompt = prompt.rsplit(_PUTTING_TOGETHER, 1)[0]
    prompt = prompt + f'Step {step_idx + 1} computes:\n'
  return prompt


def render_subgoal(states, num_examples: int, dataset_type: str) -> str:
  """The `  Case i. ...` block that follows `Step j computes:`.

  Copied from `llm_utils.get_exe_dec_prompt_prefix` so the two stay byte-identical.
  `states` is a `StepData.states` (a dict for DeepCoder, a list for RobustFill).
  """
  subgoal = ''
  if dataset_type == 'deepcoder':
    for i in range(num_examples):
      subgoal += f'  Case {i + 1}. '
      sep = ''
      for name in states:
        subgoal += f'{sep}{name} = {states[name][i]}'
        sep = ', '
      subgoal += '\n'
  else:
    raise ValueError(
        f'Only deepcoder is supported for role-split fine-tuning, got: '
        f'{dataset_type}')
  return subgoal


def render_step_code(python_program_step: str) -> str:
  """The fenced code block that follows `Step j code:`."""
  return f'```python\n{python_program_step.strip()}\n```\n'


def synthesizer_context(step_ctx: str,
                        subgoal_block: str,
                        step_idx: int) -> str:
  """The synthesizer prompt: decomposer prompt + its subgoal + `Step j code:`."""
  return step_ctx + subgoal_block + '\n' + f'Step {step_idx + 1} code:\n'


def clean_subgoal_sample(sample: str) -> str:
  """Trims a raw decomposer sample down to the `Case i.` block.

  Keeps the trailing newline the stock prompt has, so the result can be spliced
  straight into `synthesizer_context`.
  """
  for stop in (SUBGOAL_STOP, '\nPutting the steps together'):
    if stop in sample:
      sample = sample.partition(stop)[0]
  sample = sample.strip('\n')
  return sample + '\n' if sample else ''
