r"""Proves the two SFT optimizations change the loss by nothing.

1. Packing: one forward over a task's whole trajectory must give the same
   per-token loss at every completion position as the separate per-step forwards
   it replaces. This holds because step j's evaluation context is exactly the
   trajectory prefix up to `Step j+1 computes:` and attention is causal, so the
   logits at a completion position see an identical token prefix either way.

2. Selective LM head: applying `lm_head` only at labelled positions must give
   the same loss as the stock `LlamaForCausalLM` path, which applies it
   everywhere and then masks.

Needs a GPU. Runs against the base model by default, but **also run it with a
trained adapter** -- that is what this test originally missed. With no adapter
the model's distribution is flat enough that a wrong final token costs almost
nothing, so the boundary-token mismatch between packed and per-step tokenization
looked like a rounding detail. With a real SFT adapter the same token costs ~7
nats, 97% of a step's total loss.

  python -m spec_decomposition.llm_finetune.packing_equivalence_test
  python -m spec_decomposition.llm_finetune.packing_equivalence_test \
    --decomposer_adapter=results/llm_sft/decomposer_NONE/checkpoint-1000 \
    --synthesizer_adapter=results/llm_sft/synthesizer_NONE/checkpoint-1000
"""

from absl import app
from absl import flags

from spec_decomposition.llm_finetune import torch_utils

torch_utils.assert_torch_imported_first()

import torch  # pylint: disable=g-import-not-at-top,g-bad-import-order

from spec_decomposition.llm_finetune import data  # pylint: disable=g-import-not-at-top,g-bad-import-order

_BASE_MODEL = flags.DEFINE_string(
    'base_model', 'meta-llama/Llama-3.1-8B', 'Model to test with.')
_RECORDS = flags.DEFINE_string(
    'records', './data/llm_data/deepcoder_sft/NONE_valid.jsonl',
    'Packed task records.')
_DECOMPOSER_ADAPTER = flags.DEFINE_string(
    'decomposer_adapter', None,
    'LoRA to load for the decomposer role. Unset uses the untuned base, which '
    'is far less sensitive to tokenization differences -- run this with a real '
    'adapter before trusting the result.')
_SYNTHESIZER_ADAPTER = flags.DEFINE_string(
    'synthesizer_adapter', None, 'LoRA to load for the synthesizer role.')
_NUM_TASKS = flags.DEFINE_integer('num_tasks', 4, 'Tasks to check.')
_MAX_SEQ_LEN = flags.DEFINE_integer('max_seq_len', 4096, 'Token budget.')
_TOLERANCE = flags.DEFINE_float(
    'tolerance', 1e-2,
    'Allowed mean |delta log p| over shared positions. Non-zero only because '
    'the model runs in bf16, where reduction order depends on sequence length '
    'and the packed sequence is longer. Confirmed numerical, not logical: in '
    'float32 the same comparison drops from max 0.207 / mean 0.0036 to max '
    '0.000031 / mean 0.000001.')


def _token_logps(model, input_ids, labels):
  """{position: log p(token at that position)} for every labelled position."""
  ids = torch.tensor([input_ids], device='cuda')
  mask = torch.ones_like(ids)
  with torch.no_grad():
    logits = model(input_ids=ids, attention_mask=mask).logits[0].float()
  log_probs = torch.log_softmax(logits, dim=-1)
  out = {}
  for pos, label in enumerate(labels):
    if label == -100 or pos == 0:
      continue
    out[pos] = log_probs[pos - 1, label].item()
  return out


def _role_adapter(role):
  return (_DECOMPOSER_ADAPTER.value if role == data.DECOMPOSER
          else _SYNTHESIZER_ADAPTER.value)


def main(_):
  model, tokenizer = torch_utils.load_base_model(_BASE_MODEL.value)

  adapters = {r: _role_adapter(r) for r in data.ROLES if _role_adapter(r)}
  if adapters:
    from peft import PeftModel  # pylint: disable=g-import-not-at-top
    for name, path in adapters.items():
      if isinstance(model, PeftModel):
        model.load_adapter(path, adapter_name=name)
      else:
        model = PeftModel.from_pretrained(model, path, adapter_name=name)
    print(f'Loaded adapters: {", ".join(sorted(adapters))}')
  else:
    print('No adapter loaded. The untuned base is insensitive to the final-'
          'token difference this test exists to catch -- rerun with '
          '--decomposer_adapter/--synthesizer_adapter before trusting a PASS.')

  model.eval().cuda()
  model.config.use_cache = False

  records = data.load_records(_RECORDS.value, _NUM_TASKS.value)
  failures = []

  # Packing must not change the token-level supervision. This used to exclude
  # each completion's final token, because standalone a subgoal ends "]\n" while
  # in context BPE merges it with the following newline into "]\n\n". That
  # exclusion hid a real defect: SAD and reward_check scored the standalone form
  # the model was never trained on. `encode_example` now takes the `lookahead`
  # that `iter_steps` records, so both paths produce the same final token and
  # **every** labelled position must agree, boundary included.
  print(f'{"task":>5s} {"role":13s} {"aligned":>8s} {"shared":>7s} '
        f'{"max |dlogp|":>12s} {"mean |dlogp|":>13s} {"bnd":>4s}')
  for record in records:
    steps = list(data.iter_steps(record))
    for role in data.ROLES:
      if adapters:
        model.set_adapter(role if role in adapters else next(iter(adapters)))
      packed_ids, packed_labels, _ = data.packed_example(
          record, role, tokenizer, _MAX_SEQ_LEN.value)
      packed_logps = _token_logps(model, packed_ids, packed_labels)

      aligned = True
      max_delta = 0.0
      total_delta = 0.0
      shared = 0
      boundary = 0
      for step_record in steps:
        prompt, completion = data.role_example(step_record, role)
        lookahead = step_record['subgoal_lookahead' if role == data.DECOMPOSER
                                else 'step_code_lookahead']
        ids, labels, _ = torch_utils.encode_example(
            tokenizer, prompt, completion, _MAX_SEQ_LEN.value,
            append_eos=False, lookahead=lookahead)
        step_logps = _token_logps(model, ids, labels)
        positions = sorted(step_logps)
        if not set(positions) <= set(packed_logps):
          aligned = False
          continue
        for pos in positions:
          delta = abs(step_logps[pos] - packed_logps[pos])
          max_delta = max(max_delta, delta)
          total_delta += delta
          shared += 1
        # Must now be zero: a differing final token means `lookahead` is wrong.
        if packed_ids[positions[-1]] != ids[positions[-1]]:
          boundary += 1

      mean_delta = total_delta / max(shared, 1)
      ok = aligned and mean_delta <= _TOLERANCE.value and boundary == 0
      flag = '' if ok else '  <-- MISMATCH'
      print(f'{record["task_id"]:5d} {role:13s} {str(aligned):>8s} {shared:7d} '
            f'{max_delta:12.6f} {mean_delta:13.6f} {boundary:4d}{flag}')
      if not ok:
        failures.append((record['task_id'], role, aligned, mean_delta,
                         boundary))

  # (3) Selective LM head vs the stock full-logits loss, same batch.
  print()
  record = records[0]
  ids, labels, _ = data.packed_example(
      record, data.DECOMPOSER, tokenizer, _MAX_SEQ_LEN.value)
  t_ids = torch.tensor([ids], device='cuda')
  t_lab = torch.tensor([labels], device='cuda')
  t_mask = torch.ones_like(t_ids)
  with torch.no_grad():
    stock = model(input_ids=t_ids, attention_mask=t_mask, labels=t_lab).loss
    selective = torch_utils.selective_cross_entropy(
        model, t_ids, t_mask, t_lab)
  head_diff = abs(stock.item() - selective.item())
  print(f'stock causal-LM loss     : {stock.item():.6f}')
  print(f'selective-LM-head loss   : {selective.item():.6f}')
  print(f'abs diff                 : {head_diff:.8f}')
  if head_diff > 1e-4:
    failures.append(('lm_head', 'selective', stock.item(), selective.item()))

  print()
  if failures:
    print(f'FAIL: {len(failures)} mismatch(es); packing/head is NOT equivalent.')
    raise SystemExit(1)
  print('PASS: every step\'s labelled positions land at the same indices in the '
        'packed sequence and score identically -- including each completion\'s '
        'final token, which now tokenizes the same in both paths; and the '
        'selective LM head reproduces the stock loss exactly.')


if __name__ == '__main__':
  app.run(main)
