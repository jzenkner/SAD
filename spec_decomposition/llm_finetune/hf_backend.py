"""Local HF-transformers backend for `run_llm_experiment`.

Serves the fine-tuned decomposer and synthesizer LoRA adapters from a single
in-process copy of the base model, exposing the same
`(prompt, n, temperature, **kwargs) -> list[str]` contract that
`cached_llm_access.query_llm` expects from the Ollama backend.
"""

import contextlib

from absl import logging

from spec_decomposition.llm_finetune import torch_utils

torch_utils.assert_torch_imported_first()

import torch  # pylint: disable=g-import-not-at-top,g-bad-import-order
from transformers import StoppingCriteria  # pylint: disable=g-import-not-at-top
from transformers import StoppingCriteriaList  # pylint: disable=g-import-not-at-top

from spec_decomposition.llm_finetune import prompts  # pylint: disable=g-import-not-at-top,g-bad-import-order

_STATE = {}

# Text that ends a role's useful output. An SFT-tuned adapter emits EOS on its
# own, but the untuned base model does not, so without these it would generate
# until the token budget runs out on every single step.
_STOP_TEXT = {
    'decomposer': [prompts.SUBGOAL_STOP, 'Putting the steps together'],
    'synthesizer': [prompts.CODE_STOP],
}


class _StopOnText(StoppingCriteria):
  """Halts once every sequence in the batch has produced a stop marker."""

  def __init__(self, tokenizer, prompt_len, stop_strings):
    self.tokenizer = tokenizer
    self.prompt_len = prompt_len
    self.stop_strings = stop_strings

  def __call__(self, input_ids, scores, **kwargs):
    for row in input_ids:
      text = self.tokenizer.decode(
          row[self.prompt_len:], skip_special_tokens=True)
      if not any(stop in text for stop in self.stop_strings):
        return False
    return True


def init(base_model, decomposer_adapter=None, synthesizer_adapter=None,
         max_seq_len=4096, max_new_tokens=256):
  """Loads the base model once, plus whichever adapters were given.

  With no adapters this serves the untuned base model, which is the ablation
  needed to separate the effect of fine-tuning from the effect of splitting one
  generation into two. With one, the other role is served from the base weights
  as well -- that is how a SAD run trained against an untuned reward model gets
  scored against the synthesizer it actually optimised for.
  """
  if _STATE:
    return

  model, tokenizer = torch_utils.load_base_model(base_model)
  adapters = set()
  if decomposer_adapter or synthesizer_adapter:
    from peft import PeftModel  # pylint: disable=g-import-not-at-top
    first_role, first_path = (
        ('decomposer', decomposer_adapter) if decomposer_adapter
        else ('synthesizer', synthesizer_adapter))
    model = PeftModel.from_pretrained(
        model, first_path, adapter_name=first_role)
    adapters.add(first_role)
    if decomposer_adapter and synthesizer_adapter:
      model.load_adapter(synthesizer_adapter, adapter_name='synthesizer')
      adapters.add('synthesizer')

  model.eval().cuda()
  model.config.use_cache = True
  _STATE.update(model=model, tokenizer=tokenizer, adapters=adapters,
                max_seq_len=max_seq_len, max_new_tokens=max_new_tokens,
                logged_base_roles=set())
  logging.info('HF backend ready: base=%s adapters=%s',
               base_model, sorted(adapters) or 'none (untuned base)')


@torch.no_grad()
def query_llm(prompt, n, temperature, model=None, num_output_tokens=None,
              role=None):
  """Draws `n` samples for `prompt`, optionally under a role's adapter.

  `model` is accepted and ignored so the signature matches the Ollama backend,
  which takes the model name there; here the model is fixed by `init`.
  """
  del model
  if not _STATE:
    raise ValueError('Call hf_backend.init() first.')
  hf_model, tokenizer = _STATE['model'], _STATE['tokenizer']

  # A role with no adapter of its own must run on the base weights. set_adapter
  # is sticky, so after the decomposer's turn the decomposer LoRA is still
  # active and would silently serve the synthesizer too; disable_adapter is what
  # actually takes it back out. This is the eval-time counterpart of the frozen
  # zero-init LoRA sad_train gives a role that was left untuned.
  base_only = contextlib.nullcontext()
  if role and role in _STATE['adapters']:
    hf_model.set_adapter(role)
  elif role and _STATE['adapters']:
    if role not in _STATE['logged_base_roles']:
      logging.info('No adapter loaded for role %s; serving it from the untuned '
                   'base model.', role)
      _STATE['logged_base_roles'].add(role)
    base_only = hf_model.disable_adapter()

  tokenizer.padding_side = 'left'
  inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                     max_length=_STATE['max_seq_len']).to('cuda')

  # run_experiment passes num_output_tokens=10000, which is a cap for Ollama but
  # would be a real generation budget here. One ExeDec step is a handful of
  # lines, so clamp it; the stop strings below usually end generation earlier.
  budget = min(int(num_output_tokens or _STATE['max_new_tokens']),
               _STATE['max_new_tokens'])
  stop_strings = _STOP_TEXT.get(role)
  stopping = StoppingCriteriaList([
      _StopOnText(tokenizer, inputs['input_ids'].shape[1], stop_strings)
  ]) if stop_strings else None

  do_sample = temperature is not None and temperature > 0
  with base_only:
    outputs = hf_model.generate(
        **inputs,
        do_sample=do_sample,
        temperature=float(temperature) if do_sample else None,
        top_p=0.95 if do_sample else None,
        num_return_sequences=int(n),
        max_new_tokens=budget,
        stopping_criteria=stopping,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
  new_tokens = outputs[:, inputs['input_ids'].shape[1]:]
  return [tokenizer.decode(row, skip_special_tokens=True)
          for row in new_tokens]
