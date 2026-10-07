"""Checks that the two role prompts reassemble into the stock ExeDec prompt.

If this fails, a model fine-tuned on these prompts is being trained on a format
the evaluation harness does not produce, and every downstream number is void.
"""

from absl.testing import absltest

from spec_decomposition import llm_utils
from spec_decomposition.llm_finetune import prompts

DatasetElement = llm_utils.DatasetElement

# From llm_utils_test.py's paper examples: `x1 = x0 ** 2 (elementwise); sorted`.
_PROBLEM = DatasetElement(
    inputs={'x0': [[5, 3, -4], [-2], [3, -7, 1, 4]]},
    outputs=[[9, 16, 25], [4], [1, 9, 16, 49]],
    dsl_program=None,
    python_program=(
        'def program(x0):\n'
        '  x1 = [x ** 2 for x in x0]\n'
        '  x2 = sorted(x1)\n'
        '  return x2'
    ),
)

_FEW_SHOTS = [_PROBLEM]


class PromptsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.trajectory = llm_utils.get_exe_dec_trajectory(_PROBLEM, 'deepcoder')
    # inputs carry a held-out final example, which the prompt never shows.
    self.num_examples = (
        llm_utils.get_num_examples(_PROBLEM.inputs, 'deepcoder') - 1)

  def test_partial_program_reaches_full_program(self):
    """The last partial program must be the ground-truth program itself."""
    last = prompts.partial_python_program(
        self.trajectory, len(self.trajectory) - 1)
    self.assertEqual(last, _PROBLEM.python_program)

  def test_roles_reassemble_into_stock_prompt(self):
    """decomposer prompt + subgoal + `code:` + code == the stock ExeDec prompt."""
    stock = llm_utils.few_shot_exe_dec_prompt(
        _FEW_SHOTS, _PROBLEM, dataset_type='deepcoder', version=4)

    rebuilt = None
    for step_idx in range(len(self.trajectory) - 1):
      partial = prompts.partial_python_program(self.trajectory, step_idx)
      ctx = prompts.step_context(
          _FEW_SHOTS, _PROBLEM, 'deepcoder', 4, step_idx, partial)
      subgoal = prompts.render_subgoal(
          self.trajectory[step_idx + 1].states, self.num_examples, 'deepcoder')
      synth_ctx = prompts.synthesizer_context(ctx, subgoal, step_idx)
      code = prompts.render_step_code(
          self.trajectory[step_idx + 1].python_program_step)
      rebuilt = synth_ctx + code + '\n'

      # Every reassembled step must be a prefix of the stock prompt, i.e. the
      # role split introduces no stray or missing whitespace.
      self.assertTrue(
          stock.startswith(rebuilt),
          msg=(f'step {step_idx}: reassembled prompt diverges from the stock '
               f'prompt.\n--- rebuilt tail ---\n{rebuilt[-400:]!r}\n'
               f'--- stock at same offset ---\n{stock[:len(rebuilt)][-400:]!r}'))

    # After the final step, all that remains is the `Putting the steps
    # together` wrap-up that the stock prompt appends for the test problem.
    self.assertEqual(stock[len(rebuilt):].lstrip('\n').split('\n')[0],
                     'Putting the steps together, the problem is solved with '
                     'the program:')

  def test_step_context_ends_at_computes_header(self):
    for step_idx in range(len(self.trajectory) - 1):
      partial = prompts.partial_python_program(self.trajectory, step_idx)
      ctx = prompts.step_context(
          _FEW_SHOTS, _PROBLEM, 'deepcoder', 4, step_idx, partial)
      self.assertTrue(
          ctx.endswith(f'Step {step_idx + 1} computes:\n'),
          msg=f'step {step_idx} context ends with: {ctx[-80:]!r}')

  def test_clean_subgoal_sample(self):
    subgoal = prompts.render_subgoal(
        self.trajectory[1].states, self.num_examples, 'deepcoder')
    # A model continuing past the subgoal must be trimmed back to it exactly.
    noisy = subgoal + '\nStep 1 code:\n```python\nx1 = 0\n```\n'
    self.assertEqual(prompts.clean_subgoal_sample(noisy), subgoal)
    self.assertEqual(prompts.clean_subgoal_sample(subgoal), subgoal)


if __name__ == '__main__':
  absltest.main()
