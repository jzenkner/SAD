r"""Reconciles the SAD reward's absolute scale with SFT's `eval/loss`.

Two numbers did not line up after the first real SFT run, and both matter because
the SAD reward is a summed CE whose magnitude is otherwise uninterpretable:

1. `reward_check` scored the ground-truth step code at 7.46 summed nats over ~21
   tokens = 0.36 nats/token, against an SFT `eval/loss` of 0.027 nats/token on
   the same valid split with (previously verified) identical contexts -- a 13x
   gap.
2. The 600-record *smoke* synthesizer scored 3.20 where the far better-trained
   `checkpoint-1000` scored 7.46. More training, worse number. The two runs used
   different `--num_records` (100 vs 200), so this may be nothing but a different
   record subset.

This scores the *same* records under both adapters and reports per-record and
per-position numbers, which distinguishes the possibilities: a per-token mean
equal to SFT's `eval/loss` means the summed/mean confusion explains everything; a
gap concentrated in the first content token means SFT's `eval/loss` is simply
diluted by the free fence and assignment tokens; a gap spread evenly means the
reconstructed context differs from the packed one.

  python -m spec_decomposition.llm_finetune.reward_scale_check
"""

import collections
import statistics

from absl import app
from absl import flags

from spec_decomposition.llm_finetune import torch_utils

torch_utils.assert_torch_imported_first()

import torch  # pylint: disable=g-import-not-at-top,g-bad-import-order
from peft import PeftModel  # pylint: disable=g-import-not-at-top,g-bad-import-order

from spec_decomposition.llm_finetune import data  # pylint: disable=g-import-not-at-top,g-bad-import-order
from spec_decomposition.llm_finetune import prompts  # pylint: disable=g-import-not-at-top,g-bad-import-order

_BASE_MODEL = flags.DEFINE_string(
    'base_model', 'meta-llama/Llama-3.1-8B', 'Frozen base.')
_ADAPTERS = flags.DEFINE_list(
    'adapters',
    ['results/llm_smoke/synthesizer_NONE/adapter',
     'results/llm_sft/synthesizer_NONE/checkpoint-1000'],
    'Synthesizer adapters to compare, on identical records.')
_DATA_DIR = flags.DEFINE_string(
    'data_dir', './data/llm_data/deepcoder_sft', 'Packed record directory.')
_GENERALIZATION_TASK = flags.DEFINE_string(
    'generalization_task', 'NONE', 'Split the data was built from.')
_SPLIT = flags.DEFINE_enum('split', 'valid', ['train', 'valid'], 'Split.')
_NUM_RECORDS = flags.DEFINE_integer(
    'num_records', 200, 'Step records to score. reward_check used 100 for the '
    'smoke adapter and 200 for checkpoint-1000, which is the confound to rule '
    'out first.')
_MAX_SEQ_LEN = flags.DEFINE_integer('max_seq_len', 4096, 'Token budget.')


def score_record(model, tokenizer, record):
  """Per-token NLL of one record's ground-truth step code, in order."""
  synth_ctx = prompts.synthesizer_context(
      record['context'], record['subgoal'], record['step'])
  input_ids, labels, _ = torch_utils.encode_example(
      tokenizer, synth_ctx, record['step_code'], _MAX_SEQ_LEN.value,
      append_eos=False, lookahead=record['step_code_lookahead'])
  ids = torch.tensor([input_ids], device='cuda')
  mask = torch.ones_like(ids)
  with torch.no_grad():
    logits = model(input_ids=ids, attention_mask=mask).logits[0].float()
  log_probs = torch.log_softmax(logits, dim=-1)

  nlls, tokens = [], []
  for pos, label in enumerate(labels):
    if label == -100 or pos == 0:
      continue
    nlls.append(-log_probs[pos - 1, label].item())
    tokens.append(tokenizer.decode([label]))
  return nlls, tokens


def main(_):
  model, tokenizer = torch_utils.load_base_model(_BASE_MODEL.value)
  model.eval().cuda()
  model.config.use_cache = False

  records = data.load_step_records(
      f'{_DATA_DIR.value}/{_GENERALIZATION_TASK.value}_{_SPLIT.value}.jsonl',
      _NUM_RECORDS.value)
  print(f'Scoring {len(records)} step records from {_SPLIT.value}.\n')

  peft_model = None
  per_adapter = {}
  for adapter in _ADAPTERS.value:
    # Swap the adapter in place so the base weights load exactly once.
    if peft_model is None:
      peft_model = PeftModel.from_pretrained(model, adapter, adapter_name='a')
    else:
      peft_model.load_adapter(adapter, adapter_name=adapter)
      peft_model.set_adapter(adapter)
    peft_model.eval()

    sums, means, by_position = [], [], collections.defaultdict(list)
    first_content = []
    for record in records:
      nlls, tokens = score_record(peft_model, tokenizer, record)
      if not nlls:
        continue
      sums.append(sum(nlls))
      means.append(sum(nlls) / len(nlls))
      for i, nll in enumerate(nlls):
        by_position[i].append(nll)
      # "```python\n" is 3 tokens, then the variable name; the first token that
      # actually encodes a choice is the one after "xN = ".
      first_content.append(max(nlls))
    per_adapter[adapter] = (sums, means, by_position, first_content)

    print(f'=== {adapter} ===')
    print(f'  mean summed CE        {statistics.mean(sums):8.3f}   '
          f'(median {statistics.median(sums):.3f})')
    print(f'  mean per-token CE     {statistics.mean(means):8.4f}   '
          f'<- compare with SFT eval/loss')
    print(f'  mean max-token CE     {statistics.mean(first_content):8.3f}   '
          f'({100 * statistics.mean(first_content) / statistics.mean(sums):.0f}%'
          f' of the total)')
    print('  per-position mean NLL:')
    for i in sorted(by_position)[:12]:
      vals = by_position[i]
      print(f'    tok {i:2d}  n={len(vals):4d}  mean={statistics.mean(vals):7.4f}')
    print()

  if len(_ADAPTERS.value) == 2:
    a, b = _ADAPTERS.value
    sums_a, sums_b = per_adapter[a][0], per_adapter[b][0]
    n = min(len(sums_a), len(sums_b))
    better = sum(1 for i in range(n) if sums_b[i] < sums_a[i])
    print(f'Paired over {n} identical records:')
    print(f'  {b}\n    better on {better}/{n} = {100 * better / n:.1f}%')
    deltas = [sums_b[i] - sums_a[i] for i in range(n)]
    deltas.sort()
    print(f'  delta (second minus first): mean {statistics.mean(deltas):+.3f}, '
          f'median {deltas[n // 2]:+.3f}, '
          f'p10 {deltas[int(0.1 * n)]:+.3f}, p90 {deltas[int(0.9 * n)]:+.3f}')
    print('\nA mean driven by a few large deltas with a median near zero means '
          'the 3.20-vs-7.46 gap is a heavy tail, not a regression.')


if __name__ == '__main__':
  app.run(main)
