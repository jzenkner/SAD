r"""LoRA supervised fine-tuning of one ExeDec role.

Trains a single LoRA adapter on top of a frozen Llama base, on the step records
written by `build_sft_data.py`. Loss is taken over the completion only.

  python -m spec_decomposition.llm_finetune.sft_train \
    --role=synthesizer --generalization_task=NONE

Run once per role; the two resulting adapters share the same base model and are
loaded side by side at evaluation and during SAD training.
"""

import functools
import os

from absl import app
from absl import flags
from absl import logging

from spec_decomposition.llm_finetune import torch_utils

torch_utils.assert_torch_imported_first()

import torch  # pylint: disable=g-import-not-at-top,g-bad-import-order
from peft import LoraConfig  # pylint: disable=g-import-not-at-top
from peft import get_peft_model  # pylint: disable=g-import-not-at-top
from torch.utils.data import Dataset  # pylint: disable=g-import-not-at-top
from transformers import Trainer  # pylint: disable=g-import-not-at-top
from transformers import TrainingArguments  # pylint: disable=g-import-not-at-top

from spec_decomposition.llm_finetune import data  # pylint: disable=g-import-not-at-top,g-bad-import-order

_ROLE = flags.DEFINE_enum(
    'role', None, list(data.ROLES), 'Which role to train.', required=True)
_BASE_MODEL = flags.DEFINE_string(
    'base_model', 'meta-llama/Llama-3.1-8B',
    'HF model id or local path of the frozen base model.')
_DATA_DIR = flags.DEFINE_string(
    'data_dir', './data/llm_data/deepcoder_sft',
    'Directory of step-record JSONL files.')
_GENERALIZATION_TASK = flags.DEFINE_string(
    'generalization_task', 'NONE', 'Which split the data was built from.')
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', './results/llm_sft',
    'Adapters are saved to <output_dir>/<role>_<generalization_task>.')
_MAX_SEQ_LEN = flags.DEFINE_integer('max_seq_len', 4096, 'Token budget.')
_LORA_R = flags.DEFINE_integer('lora_r', 32, 'LoRA rank.')
_LORA_ALPHA = flags.DEFINE_integer('lora_alpha', 64, 'LoRA alpha.')
_LORA_DROPOUT = flags.DEFINE_float('lora_dropout', 0.05, 'LoRA dropout.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 1e-4, 'Peak LR.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 1, 'Per-device batch size.')
_GRAD_ACCUM = flags.DEFINE_integer(
    'grad_accum', 16, 'Gradient accumulation steps.')
_NUM_EPOCHS = flags.DEFINE_float('num_epochs', 2.0, 'Training epochs.')
_WARMUP_STEPS = flags.DEFINE_integer('warmup_steps', 100, 'LR warmup steps.')
_MAX_TRAIN_RECORDS = flags.DEFINE_integer(
    'max_train_records', None, 'Cap on training records (for smoke tests).')
_MAX_EVAL_RECORDS = flags.DEFINE_integer(
    'max_eval_records', 500, 'Cap on eval records.')
_LOG_STEPS = flags.DEFINE_integer('log_steps', 10, 'Logging interval.')
_EVAL_STEPS = flags.DEFINE_integer('eval_steps', 500, 'Eval interval.')
_SAVE_STEPS = flags.DEFINE_integer('save_steps', 500, 'Checkpoint interval.')
_SAVE_TOTAL_LIMIT = flags.DEFINE_integer(
    'save_total_limit', 2,
    'How many checkpoints to keep. Raise it for long runs: with the default 2, '
    'a run of thousands of steps keeps only the last two, so if eval loss turns '
    'upward halfway through, the good checkpoint is already deleted and there '
    'is no resume path.')

# Lives in torch_utils so sad_train can start a fresh LoRA matching this one
# without importing this module, which would collide on ~8 shared absl flags.
LORA_TARGET_MODULES = torch_utils.LORA_TARGET_MODULES


class RoleDataset(Dataset):
  """Tokenized (prompt, completion) pairs for one role."""

  def __init__(self, records, role, tokenizer, max_seq_len):
    self.examples = []
    num_truncated = 0
    for record in records:
      # One sequence per task, not per step: step j's context is a prefix of the
      # packed trajectory, so a single causal forward covers every step at once.
      input_ids, labels, truncated = data.packed_example(
          record, role, tokenizer, max_seq_len)
      num_truncated += bool(truncated)
      self.examples.append({'input_ids': input_ids, 'labels': labels})

    lengths = [len(e['input_ids']) for e in self.examples]
    loss_tokens = sum(sum(1 for x in e['labels'] if x != -100)
                      for e in self.examples)
    total_tokens = sum(lengths)
    logging.info(
        '%s: %d packed tasks, token length min/mean/max = %d/%.0f/%d, '
        '%d truncated (%.1f%%), %d of %d tokens carry loss (%.1f%%)',
        role, len(self.examples), min(lengths),
        total_tokens / len(self.examples), max(lengths),
        num_truncated, 100 * num_truncated / len(self.examples),
        loss_tokens, total_tokens, 100 * loss_tokens / total_tokens)
    if num_truncated > 0.05 * len(self.examples):
      logging.warning(
          'Over 5%% of trajectories were truncated. Raise --max_seq_len, '
          'otherwise training and evaluation see different prompts.')

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, index):
    return self.examples[index]


class SelectiveHeadTrainer(Trainer):
  """Trainer whose loss only materializes logits where there is a label.

  Equivalent to the stock causal-LM loss (verified in
  `packing_equivalence_test.py`), but without allocating the full
  `batch x seq x vocab` float32 logits tensor.
  """

  def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    loss = torch_utils.selective_cross_entropy(
        model, inputs['input_ids'], inputs['attention_mask'],
        inputs['labels'])
    # Trainer.prediction_step does `outputs[1:]` unless outputs is a dict, so an
    # empty dict is the safe "no logits to return" answer. We never need the
    # logits at eval time -- only eval_loss.
    return (loss, {}) if return_outputs else loss


def main(_):
  role = _ROLE.value
  run_dir = os.path.join(
      _OUTPUT_DIR.value, f'{role}_{_GENERALIZATION_TASK.value}')
  os.makedirs(run_dir, exist_ok=True)

  model, tokenizer = torch_utils.load_base_model(_BASE_MODEL.value)

  train_path = os.path.join(
      _DATA_DIR.value, f'{_GENERALIZATION_TASK.value}_train.jsonl')
  eval_path = os.path.join(
      _DATA_DIR.value, f'{_GENERALIZATION_TASK.value}_valid.jsonl')

  train_dataset = RoleDataset(
      data.load_records(train_path, _MAX_TRAIN_RECORDS.value),
      role, tokenizer, _MAX_SEQ_LEN.value)
  eval_dataset = RoleDataset(
      data.load_records(eval_path, _MAX_EVAL_RECORDS.value),
      role, tokenizer, _MAX_SEQ_LEN.value)

  model = get_peft_model(model, LoraConfig(
      r=_LORA_R.value,
      lora_alpha=_LORA_ALPHA.value,
      lora_dropout=_LORA_DROPOUT.value,
      target_modules=LORA_TARGET_MODULES,
      bias='none',
      task_type='CAUSAL_LM',
  ))
  model.print_trainable_parameters()
  # Required for gradient checkpointing to reach the LoRA parameters, since the
  # frozen embedding output would otherwise carry no grad.
  model.enable_input_require_grads()

  trainer = SelectiveHeadTrainer(
      model=model,
      args=TrainingArguments(
          output_dir=run_dir,
          per_device_train_batch_size=_BATCH_SIZE.value,
          per_device_eval_batch_size=_BATCH_SIZE.value,
          gradient_accumulation_steps=_GRAD_ACCUM.value,
          num_train_epochs=_NUM_EPOCHS.value,
          learning_rate=_LEARNING_RATE.value,
          lr_scheduler_type='cosine',
          warmup_steps=_WARMUP_STEPS.value,
          bf16=True,
          gradient_checkpointing=True,
          gradient_checkpointing_kwargs={'use_reentrant': False},
          logging_steps=_LOG_STEPS.value,
          eval_strategy='steps',
          eval_steps=_EVAL_STEPS.value,
          save_strategy='steps',
          save_steps=_SAVE_STEPS.value,
          save_total_limit=_SAVE_TOTAL_LIMIT.value,
          report_to=['tensorboard'],
          logging_dir=os.path.join(run_dir, 'tb'),
          remove_unused_columns=False,
          # The selective-head loss returns no logits, and only eval_loss is
          # wanted anyway; this keeps Trainer from trying to gather them.
          prediction_loss_only=True,
          seed=0,
      ),
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      data_collator=functools.partial(
          torch_utils.collate, pad_token_id=tokenizer.pad_token_id),
  )

  trainer.train()

  # Peak VRAM is the knob for choosing --batch_size: the aim is to fill the card
  # (~80% of capacity), since anything below ~50% is throughput left on the
  # table. Reported here so a short run can size the real one.
  peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
  total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
  logging.info('Peak VRAM %.1f GB of %.1f GB (%.0f%%) at batch_size=%d',
               peak_gb, total_gb, 100 * peak_gb / total_gb, _BATCH_SIZE.value)
  print(f'PEAK_VRAM {peak_gb:.1f} GB / {total_gb:.1f} GB '
        f'({100 * peak_gb / total_gb:.0f}%) batch_size={_BATCH_SIZE.value}')

  adapter_dir = os.path.join(run_dir, 'adapter')
  model.save_pretrained(adapter_dir)
  tokenizer.save_pretrained(adapter_dir)
  logging.info('Saved %s adapter to %s', role, adapter_dir)
  print(f'Saved {role} adapter to {adapter_dir}')


if __name__ == '__main__':
  torch.backends.cuda.matmul.allow_tf32 = True
  app.run(main)
