r"""Checks that the SAD reward actually discriminates good subgoals from bad.

The reward driving SAD training is the frozen synthesizer's summed CE of the
ground-truth step code given a proposed subgoal. If a corrupted subgoal scores
no worse than the true one, the reward carries no signal and SAD training cannot
help, however well the optimizer behaves. Run this before trusting any SAD run.

Corruptions mirror `_corrupt` in tasks/deepcoder/dataset/write_data.py:
copy_output, perturb and new_random, plus a shuffle across cases.

  python -m spec_decomposition.llm_finetune.reward_check \
    --synthesizer_adapter=./results/llm_sft/synthesizer_NONE/adapter
"""

import ast
import collections
import random
import re
import sys

from absl import app
from absl import flags

from spec_decomposition.llm_finetune import torch_utils

torch_utils.assert_torch_imported_first()

import torch  # pylint: disable=g-import-not-at-top,g-bad-import-order
from peft import PeftModel  # pylint: disable=g-import-not-at-top

from spec_decomposition.llm_finetune import data  # pylint: disable=g-import-not-at-top,g-bad-import-order
from spec_decomposition.llm_finetune import prompts  # pylint: disable=g-import-not-at-top

_BASE_MODEL = flags.DEFINE_string(
    'base_model', 'meta-llama/Llama-3.1-8B', 'Frozen base model.')
_SYNTHESIZER_ADAPTER = flags.DEFINE_string(
    'synthesizer_adapter', None,
    'Synthesizer adapter to score with. Unset uses the untuned base model.')
_DATA_DIR = flags.DEFINE_string(
    'data_dir', './data/llm_data/deepcoder_sft', 'Step-record directory.')
_GENERALIZATION_TASK = flags.DEFINE_string(
    'generalization_task', 'NONE', 'Which split to read.')
_SPLIT = flags.DEFINE_enum('split', 'valid', ['train', 'valid'], 'Split.')
_NUM_RECORDS = flags.DEFINE_integer('num_records', 100, 'Records to score.')
_MAX_SEQ_LEN = flags.DEFINE_integer('max_seq_len', 4096, 'Token budget.')
_CHECK_SEED = flags.DEFINE_integer('check_seed', 0, 'Corruption seed.')

_CASE_RE = re.compile(r'^  Case (\d+)\. (.+?) = (.+)$')

MIN_INT, MAX_INT = -50, 50
MAX_LIST_LENGTH = 5


def parse_subgoal(subgoal):
  """Parses a rendered subgoal block into (var_name, [value per case])."""
  var_name, values = None, []
  for line in subgoal.strip('\n').split('\n'):
    match = _CASE_RE.match(line)
    if not match:
      raise ValueError(f'Unparseable subgoal line: {line!r}')
    var_name = match.group(2)
    values.append(ast.literal_eval(match.group(3)))
  return var_name, values


def render(var_name, values):
  return ''.join(f'  Case {i + 1}. {var_name} = {v}\n'
                 for i, v in enumerate(values))


def _random_value(rng, like_list):
  if like_list:
    return [rng.randint(MIN_INT, MAX_INT)
            for _ in range(rng.randint(0, MAX_LIST_LENGTH))]
  return rng.randint(MIN_INT, MAX_INT)


def corrupt(rng, values, outputs, technique):
  """Returns a corrupted copy of `values`, or None if nothing changed."""
  if technique == 'copy_output':
    new_values = list(outputs[:len(values)])
  elif technique == 'shuffle':
    new_values = list(values)
    rng.shuffle(new_values)
  elif technique == 'new_random':
    like_list = isinstance(values[0], list)
    new_values = [_random_value(rng, like_list) for _ in values]
  elif technique == 'perturb':
    new_values = []
    for value in values:
      if isinstance(value, list):
        item = list(value)
        for _ in range(rng.randint(1, 2)):
          kind = rng.choice(['insert', 'delete', 'replace'] if item
                            else ['insert'])
          if kind == 'insert':
            item.insert(rng.randint(0, len(item)),
                        rng.randint(MIN_INT, MAX_INT))
          elif kind == 'delete':
            del item[rng.randrange(len(item))]
          else:
            item[rng.randrange(len(item))] = rng.randint(MIN_INT, MAX_INT)
        new_values.append(item)
      else:
        new_values.append(rng.randint(MIN_INT, MAX_INT))
  else:
    raise ValueError(f'Unknown technique: {technique}')
  return None if new_values == values else new_values


def main(_):
  rng = random.Random(_CHECK_SEED.value)
  model, tokenizer = torch_utils.load_base_model(_BASE_MODEL.value)
  if _SYNTHESIZER_ADAPTER.value:
    model = PeftModel.from_pretrained(model, _SYNTHESIZER_ADAPTER.value)
  model.eval().cuda()
  model.config.use_cache = False

  records = data.load_step_records(
      f'{_DATA_DIR.value}/{_GENERALIZATION_TASK.value}_{_SPLIT.value}.jsonl',
      _NUM_RECORDS.value)

  techniques = ['copy_output', 'shuffle', 'perturb', 'new_random']
  losses = collections.defaultdict(list)
  wins = collections.Counter()
  totals = collections.Counter()

  def score(context, subgoal, step, code, lookahead):
    synth_ctx = prompts.synthesizer_context(context, subgoal, step)
    input_ids, labels, _ = torch_utils.encode_example(
        tokenizer, synth_ctx, code, _MAX_SEQ_LEN.value, append_eos=False,
        lookahead=lookahead)
    batch = torch_utils.collate(
        [{'input_ids': input_ids, 'labels': labels}], tokenizer.pad_token_id)
    batch = {k: v.cuda() for k, v in batch.items()}
    with torch.no_grad():
      logits = model(input_ids=batch['input_ids'],
                     attention_mask=batch['attention_mask']).logits
      sum_log_probs, _ = torch_utils.token_stats(logits, batch['labels'])
    return -sum_log_probs.item()  # summed CE, the SAD reward's negation

  for record in records:
    try:
      var_name, values = parse_subgoal(record['subgoal'])
    except ValueError:
      continue
    lookahead = record['step_code_lookahead']
    gt_loss = score(record['context'], record['subgoal'], record['step'],
                    record['step_code'], lookahead)
    losses['ground_truth'].append(gt_loss)

    for technique in techniques:
      corrupted = corrupt(rng, values, record['outputs'], technique)
      if corrupted is None:
        continue
      bad_loss = score(record['context'], render(var_name, corrupted),
                       record['step'], record['step_code'], lookahead)
      losses[technique].append(bad_loss)
      totals[technique] += 1
      wins[technique] += int(gt_loss < bad_loss)

  gt_mean = sum(losses['ground_truth']) / len(losses['ground_truth'])
  print(f'\nScored {len(losses["ground_truth"])} records '
        f'(adapter: {_SYNTHESIZER_ADAPTER.value or "none, untuned base"})')
  print(f'{"subgoal":14s} {"mean summed CE":>15s} {"vs ground truth":>16s} '
        f'{"gt better":>12s}')
  print(f'{"ground_truth":14s} {gt_mean:15.2f} {"-":>16s} {"-":>12s}')

  failures = []
  for technique in techniques:
    if not totals[technique]:
      continue
    mean = sum(losses[technique]) / len(losses[technique])
    rate = wins[technique] / totals[technique]
    print(f'{technique:14s} {mean:15.2f} {mean - gt_mean:+16.2f} '
          f'{rate:11.1%}')
    if mean <= gt_mean:
      failures.append(technique)

  if failures:
    print(f'\nFAIL: corrupted subgoals scored no worse than the truth for '
          f'{failures}. The SAD reward has no signal here; fix the synthesizer '
          f'before running sad_train.py.')
    sys.exit(1)
  print('\nOK: every corruption makes the ground-truth code less likely.')


if __name__ == '__main__':
  app.run(main)
