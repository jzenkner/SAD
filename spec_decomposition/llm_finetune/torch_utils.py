"""Shared setup for the PyTorch side of the LLM fine-tuning code.

This package runs PyTorch inside a repo whose other half is TensorFlow/JAX, and
the two need keeping apart. Importing this module is what does it, so import it
before `transformers`.
"""

import os

# transformers auto-detects the installed TensorFlow and imports its TF
# integration, which needs `tf-keras` because TF 2.17 ships Keras 3 — an import
# error, not a warning. Declaring the torch backend makes transformers skip TF
# entirely. Set at import time so it cannot lose a race with any import order.
os.environ['USE_TORCH'] = '1'
os.environ['USE_TF'] = '0'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def assert_torch_imported_first():
  """Fails loudly if TensorFlow was imported before PyTorch in this process.

  TensorFlow and PyTorch coexist here only in one order. Measured on an H100
  node with torch 2.4.1+cu121 and TF 2.17:

    import tensorflow; import torch  -> segfault on the first CUDA allocation
    import torch; import tensorflow  -> works

  `torch.cuda.is_available()` returns True in the broken case, so the failure
  reads as a random crash rather than a library conflict. Hiding the GPU from TF
  with `tf.config.set_visible_devices([], 'GPU')` does not help -- only the order
  matters.

  TF is easy to pull in by accident: `llm_utils` used to import it eagerly (now
  lazy, inside `create_dataset`), and `torch.utils.tensorboard` still does, as
  does `tensorboard`'s own `EventFileWriter` when instantiated. So every entry
  point in this package calls this *before* importing torch, which pins the
  order; anything importing TF later is then harmless.
  """
  import sys  # pylint: disable=g-import-not-at-top
  if 'tensorflow' in sys.modules:
    raise RuntimeError(
        'TensorFlow was imported before PyTorch in a process that uses the GPU; '
        'the first CUDA allocation would segfault. Find the eager `import '
        'tensorflow` in the import graph and make it lazy.')


def load_base_model(model_path, dtype='bfloat16'):
  """Loads the frozen base LLM and its tokenizer."""
  import torch  # pylint: disable=g-import-not-at-top
  from transformers import AutoModelForCausalLM  # pylint: disable=g-import-not-at-top
  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

  tokenizer = AutoTokenizer.from_pretrained(model_path)
  if tokenizer.pad_token is None:
    # Llama has no pad token; padding is masked out of the loss anyway.
    tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = 'right'

  model = AutoModelForCausalLM.from_pretrained(
      model_path,
      torch_dtype=getattr(torch, dtype),
      # flash-attn is not installed in this environment.
      attn_implementation='sdpa',
  )
  model.config.pad_token_id = tokenizer.pad_token_id
  return model, tokenizer


LORA_TARGET_MODULES = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'gate_proj', 'up_proj', 'down_proj',
]


def encode_example(tokenizer, prompt, completion, max_seq_len,
                   append_eos=True, lookahead=''):
  """Tokenizes one (prompt, completion) pair with the prompt masked out.

  Over-long examples lose tokens from the *front* of the prompt, so the
  completion and the most recent few-shot examples always survive. Returns
  (input_ids, labels, num_truncated).

  `append_eos` should be True for training targets, so the model learns to stop,
  and False when the completion is only being *scored* (as in the SAD reward),
  where an EOS the model was never asked to produce would just add noise.

  `lookahead` is the text that follows the completion in the real sequence, and
  supplying it is what makes this path agree with `data.packed_example`. BPE
  merges across the completion's trailing newline: measured over 152 spans per
  role, the *final* token differs from the standalone tokenization on 100% of
  decomposer spans ("]\\n\\n" packed vs "]\\n" alone) and 67% of synthesizer
  spans. SFT trains on the packed form, so scoring the standalone form asks a
  confident model for a token it has never seen there -- worth ~7 nats, which is
  97% of a trained synthesizer's total loss on a step. The lookahead is used
  only to steer the merge; tokens lying entirely inside it are dropped, so
  nothing beyond the completion is ever fed to the model or scored.
  """
  # Tokenize as one string and recover labels from character offsets, the same
  # rule `data.packed_example` uses, so the two paths agree token for token.
  text = prompt + completion + lookahead
  start_char, end_char = len(prompt), len(prompt) + len(completion)
  encoded = tokenizer(text, add_special_tokens=False,
                      return_offsets_mapping=True)

  prompt_ids, completion_ids = [], []
  for token_id, (start, end) in zip(encoded['input_ids'],
                                    encoded['offset_mapping']):
    if start >= end_char:
      break  # entirely inside the lookahead
    # Overlap, not containment -- see `data.packed_example`. The boundary token
    # straddles the completion and the lookahead, and belongs to the completion
    # because it is a token the model must actually generate.
    if start < end_char and end > start_char:
      completion_ids.append(token_id)
    else:
      prompt_ids.append(token_id)
  if append_eos:
    completion_ids = completion_ids + [tokenizer.eos_token_id]

  budget = max_seq_len - 1 - len(completion_ids)  # 1 for BOS
  if budget <= 0:
    raise ValueError(
        f'Completion of {len(completion_ids)} tokens does not fit in '
        f'max_seq_len={max_seq_len}.')
  num_truncated = max(0, len(prompt_ids) - budget)
  if num_truncated:
    prompt_ids = prompt_ids[num_truncated:]

  input_ids = [tokenizer.bos_token_id] + prompt_ids + completion_ids
  labels = [-100] * (1 + len(prompt_ids)) + completion_ids
  return input_ids, labels, num_truncated


def collate(batch, pad_token_id):
  """Right-pads a batch of {input_ids, labels} dicts."""
  import torch  # pylint: disable=g-import-not-at-top

  max_len = max(len(item['input_ids']) for item in batch)
  input_ids, labels, attention_mask = [], [], []
  for item in batch:
    pad = max_len - len(item['input_ids'])
    input_ids.append(item['input_ids'] + [pad_token_id] * pad)
    labels.append(item['labels'] + [-100] * pad)
    attention_mask.append([1] * len(item['input_ids']) + [0] * pad)
  return {
      'input_ids': torch.tensor(input_ids, dtype=torch.long),
      'labels': torch.tensor(labels, dtype=torch.long),
      'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
  }


def selective_lm_logits(model, input_ids, attention_mask, labels):
  """Runs the LM head only where a loss is actually taken.

  `LlamaForCausalLM.forward` applies the head at every position and then does
  `logits = logits.float()`, allocating `batch x seq x 128256 x 4` bytes -- the
  single 37.7 GB allocation that OOMs at batch 32. Only 4-6% of positions carry
  a label here, so the head is run on the gathered positions instead, cutting
  that tensor by ~20x and skipping the corresponding matmul.

  Returns (selected_logits [n, vocab] float32, selected_labels [n],
  row_index [n]) for the *shifted* positions, i.e. position t predicts label
  t+1, matching the convention in `token_stats`.
  """
  import torch  # pylint: disable=g-import-not-at-top

  # Reach through PEFT to the LlamaForCausalLM, whose `.model` is the body and
  # `.lm_head` the output projection. LoRA lives on the body's submodules, so
  # gradients still flow to the adapter.
  causal_lm = model.get_base_model() if hasattr(model, 'get_base_model') else model
  body, lm_head = causal_lm.model, causal_lm.lm_head

  hidden = body(input_ids=input_ids, attention_mask=attention_mask)[0]

  shift_hidden = hidden[:, :-1, :]
  shift_labels = labels[:, 1:]
  mask = shift_labels != -100
  if not bool(mask.any()):
    empty = shift_hidden.new_zeros((0, lm_head.out_features), dtype=torch.float32)
    return empty, shift_labels.new_zeros((0,)), shift_labels.new_zeros((0,))

  rows = mask.nonzero(as_tuple=True)[0]
  selected_logits = lm_head(shift_hidden[mask]).float()
  return selected_logits, shift_labels[mask], rows


def selective_cross_entropy(model, input_ids, attention_mask, labels):
  """Token-mean CE over the labelled positions, via `selective_lm_logits`."""
  import torch  # pylint: disable=g-import-not-at-top

  logits, targets, _ = selective_lm_logits(
      model, input_ids, attention_mask, labels)
  if logits.shape[0] == 0:
    return logits.sum()  # zero, but keeps the graph connected
  return torch.nn.functional.cross_entropy(logits, targets)


def token_stats(logits, labels):
  """Per-example summed log-prob of the labelled tokens, and their entropy.

  Returns (sum_log_probs [batch], mean_entropy scalar). The sum (rather than a
  token mean) matches `train.py`'s `sum_log_probs_per_ex` and its
  `compute_weighted_cross_entropy(..., per_example=True)`, which also sums over
  the sequence.

  Only the labelled positions are gathered before the softmax. Prompts here run
  to ~2500 tokens while completions are ~20-35, so taking log_softmax over the
  whole sequence in fp32 would allocate gigabytes over a 128k vocabulary for
  positions whose values are then masked away.
  """
  import torch  # pylint: disable=g-import-not-at-top

  logits = logits[:, :-1, :]
  labels = labels[:, 1:]
  mask = labels != -100

  batch_size = labels.shape[0]
  sum_log_probs = torch.zeros(
      batch_size, device=logits.device, dtype=torch.float32)
  if not bool(mask.any()):
    return sum_log_probs, sum_log_probs.new_zeros(())

  rows = mask.nonzero(as_tuple=True)[0]
  selected_logits = logits[mask].float()          # [num_labelled, vocab]
  selected_labels = labels[mask]                  # [num_labelled]

  log_probs = torch.log_softmax(selected_logits, dim=-1)
  token_log_probs = log_probs.gather(
      -1, selected_labels.unsqueeze(-1)).squeeze(-1)
  sum_log_probs = sum_log_probs.index_add(0, rows, token_log_probs)

  mean_entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
  return sum_log_probs, mean_entropy
