r"""SAD training of the decomposer LLM against a frozen synthesizer LLM.

A port of `spec_decomposition/train.py:427-488` to a LoRA-adapted Llama. The
decomposer samples a subgoal; the reward is how likely the *frozen synthesizer*
then finds the ground-truth step code:

    synth_loss  = summed CE of gt_code | context, sampled_subgoal
    greedy_loss = summed CE of gt_code | context, greedy_subgoal
    advantage   = greedy_loss - synth_loss           (self-critical baseline)
    advantage  -= advantage.mean()                   (batch centering)
    rl_loss     = -mean(advantage.detach() * sum_log_probs(sampled))
    total_loss  = rl_loss - ent_coef * entropy + ce_weight * supervised_ce

This is self-critical REINFORCE, not group-relative GRPO: one sample per prompt,
a greedy baseline, no importance ratio and no clipping. The `--rl_loss=grpo`
flag name is kept because `train.py` uses it for the same algorithm.

  python -m spec_decomposition.llm_finetune.sad_train \
    --decomposer_adapter=./results/llm_sft/decomposer_NONE/adapter \
    --synthesizer_adapter=./results/llm_sft/synthesizer_NONE/adapter
"""

import collections
import os
import random
import time

from absl import app
from absl import flags
from absl import logging

from spec_decomposition.llm_finetune import torch_utils

torch_utils.assert_torch_imported_first()

import torch  # pylint: disable=g-import-not-at-top,g-bad-import-order
from peft import PeftModel  # pylint: disable=g-import-not-at-top
from peft import get_peft_model  # pylint: disable=g-import-not-at-top
# Safe only because torch is already imported above; see
# torch_utils.assert_torch_imported_first for why the order matters.
from torch.utils.tensorboard import SummaryWriter  # pylint: disable=g-import-not-at-top

from spec_decomposition.llm_finetune import data  # pylint: disable=g-import-not-at-top,g-bad-import-order
from spec_decomposition.llm_finetune import prompts  # pylint: disable=g-import-not-at-top

_BASE_MODEL = flags.DEFINE_string(
    'base_model', 'meta-llama/Llama-3.1-8B', 'Frozen base model.')
_LORA_R = flags.DEFINE_integer('lora_r', 32, 'LoRA rank, when starting fresh.')
_LORA_ALPHA = flags.DEFINE_integer('lora_alpha', 64, 'LoRA alpha.')
_LORA_DROPOUT = flags.DEFINE_float('lora_dropout', 0.05, 'LoRA dropout.')
_DECOMPOSER_ADAPTER = flags.DEFINE_string(
    'decomposer_adapter', None,
    'SFT decomposer adapter to start from. Unset starts a fresh LoRA on the '
    'untuned base -- the "SAD without SFT" ablation.')
_SYNTHESIZER_ADAPTER = flags.DEFINE_string(
    'synthesizer_adapter', None,
    'Frozen SFT synthesizer, the reward model. Unset uses the untuned base.')
_DATA_DIR = flags.DEFINE_string(
    'data_dir', './data/llm_data/deepcoder_sft', 'Step-record directory.')
_GENERALIZATION_TASK = flags.DEFINE_string(
    'generalization_task', 'NONE', 'Which split the data was built from.')
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', './results/llm_sad', 'Where to write the adapter.')
_EXP_TITLE = flags.DEFINE_string(
    'exp_title', None, 'Run name; defaults to <rl_loss>_<task>.')

_RL_LOSS = flags.DEFINE_enum(
    'rl_loss', 'grpo', ['supervised', 'grpo'],
    'Which SAD training option to use. Matches train.py, where "grpo" is the '
    'self-critical REINFORCE loss and "supervised" is CE only.')
_TEMPERATURE = flags.DEFINE_float(
    'temperature', 1.0, 'Sampling temperature (train.py hardcodes temp = 1).')
_ENT_COEF = flags.DEFINE_float(
    'ent_coef', 0.001, 'Entropy bonus, hardcoded to 0.001 in train.py.')
_CE_WEIGHT = flags.DEFINE_float(
    'ce_weight', 1.0, 'Weight on the supervised CE term, which train.py always '
    'adds to the RL loss.')
_NORMALIZE_ADVANTAGE = flags.DEFINE_bool(
    'normalize_advantage', False,
    'Divide the centered advantage by its std. Off by default because the '
    'corresponding line is commented out in train.py:469.')
_KL_BETA = flags.DEFINE_float(
    'kl_beta', 0.0,
    'KL penalty against the frozen starting policy. Not present in train.py; '
    'available because an 8B policy can drift much faster than the small '
    'from-scratch transformer. 0 disables it and loads no reference adapter.')

_LEARNING_RATE = flags.DEFINE_float('learning_rate', 5e-6, 'Peak LR.')
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 4,
    'Contexts per step. Advantages are centered within this batch, so it must '
    'be greater than 1.')
_GRAD_ACCUM = flags.DEFINE_integer('grad_accum', 4, 'Accumulation steps.')
_NUM_TRAIN_STEPS = flags.DEFINE_integer('num_train_steps', 2000, 'Steps.')
_MAX_SEQ_LEN = flags.DEFINE_integer('max_seq_len', 4096, 'Token budget.')
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    'max_new_tokens', 200, 'Generation budget for a subgoal.')
_MAX_TRAIN_RECORDS = flags.DEFINE_integer(
    'max_train_records', None, 'Cap on step records (for smoke tests).')
_LOG_STEPS = flags.DEFINE_integer('log_steps', 5, 'Logging interval.')
_PROBE_STEPS = flags.DEFINE_integer(
    'probe_steps', 25,
    'Run the fixed held-out probe every this many steps. 0 disables it.')
_PROBE_RECORDS = flags.DEFINE_integer(
    'probe_records', 64,
    'Validation step records in the probe. The same ones every time -- that is '
    'the point, since per-step training metrics are dominated by which tasks '
    'the random batch happened to draw.')
_SAVE_STEPS = flags.DEFINE_integer('save_steps', 250, 'Checkpoint interval.')
_TRAIN_SEED = flags.DEFINE_integer('train_seed', 0, 'Shuffling/sampling seed.')

DECOMPOSER = 'decomposer'
SYNTHESIZER = 'synthesizer'
REFERENCE = 'reference'


def adapter_dir(run_dir):
  """Where PEFT puts the decomposer adapter under `run_dir`."""
  return os.path.join(run_dir, DECOMPOSER)


def _set_mode(model, grad):
  """Train mode for the backward passes, eval mode for everything else.

  Not cosmetic: transformers only honours gradient checkpointing when
  `self.training` is True (`LlamaModel.forward` guards on
  `self.gradient_checkpointing and self.training`), so leaving the model in eval
  mode silently disables checkpointing and OOMs a 94 GB H100. Eval mode is
  equally necessary for the no-grad passes, since LoRA dropout would otherwise
  make the greedy baseline nondeterministic.
  """
  model.train(mode=grad)


def fresh_lora_config():
  """A LoRA matching sft_train's, for adapters started from the base model."""
  from peft import LoraConfig  # pylint: disable=g-import-not-at-top

  return LoraConfig(
      r=_LORA_R.value,
      lora_alpha=_LORA_ALPHA.value,
      lora_dropout=_LORA_DROPOUT.value,
      target_modules=torch_utils.LORA_TARGET_MODULES,
      bias='none',
      task_type='CAUSAL_LM',
  )


def load_model():
  """Loads the base model with the decomposer, synthesizer and ref adapters.

  Either adapter path may be omitted, which runs that role off the untuned base.
  A freshly initialised LoRA has B = 0 and therefore contributes exactly zero, so
  a new adapter *is* the base model at step 0: for the decomposer that is the
  right starting point to learn from, and for the synthesizer -- which is frozen
  -- it stays a no-op for the whole run, making the reward the off-the-shelf
  model's likelihood. This is the ablation for "does SAD work without SFT first".
  """
  model, tokenizer = torch_utils.load_base_model(_BASE_MODEL.value)

  if _DECOMPOSER_ADAPTER.value:
    model = PeftModel.from_pretrained(
        model, _DECOMPOSER_ADAPTER.value, adapter_name=DECOMPOSER,
        is_trainable=True)
  else:
    logging.info('No --decomposer_adapter: starting the policy from a fresh '
                 'LoRA on the untuned base.')
    model = get_peft_model(model, fresh_lora_config(),
                           adapter_name=DECOMPOSER)

  if _SYNTHESIZER_ADAPTER.value:
    model.load_adapter(_SYNTHESIZER_ADAPTER.value, adapter_name=SYNTHESIZER)
  else:
    logging.info('No --synthesizer_adapter: the reward model is the untuned '
                 'base, via a frozen zero-init LoRA.')
    model.add_adapter(SYNTHESIZER, fresh_lora_config())

  if _KL_BETA.value > 0:
    if not _DECOMPOSER_ADAPTER.value:
      raise ValueError('--kl_beta needs a --decomposer_adapter to be the '
                       'reference policy.')
    model.load_adapter(_DECOMPOSER_ADAPTER.value, adapter_name=REFERENCE)

  # Only the decomposer learns. add_adapter/get_peft_model can leave a new
  # adapter trainable, and a trainable synthesizer would stop being a frozen
  # reward model -- the reward would chase the policy instead of scoring it.
  for name, param in model.named_parameters():
    if f'.{SYNTHESIZER}.' in name or f'.{REFERENCE}.' in name:
      param.requires_grad_(False)
    elif f'.{DECOMPOSER}.' in name:
      param.requires_grad_(True)

  model.set_adapter(DECOMPOSER)
  model.config.use_cache = False
  model.enable_input_require_grads()
  # Without this a grad-enabled forward over ~2500-token contexts OOMs even on
  # a 94 GB H100. Generation runs under no_grad, where checkpointing is inert.
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
      'use_reentrant': False})
  return model.cuda(), tokenizer


@torch.no_grad()
def generate_subgoals(model, tokenizer, contexts, do_sample, temperature,
                      max_new_tokens, adapter=DECOMPOSER, clean=None):
  """Batched generation. Returns (token ids per example, cleaned texts).

  The returned ids are the exact tokens that were sampled, truncated after the
  first EOS. Scoring the policy on these ids rather than on a re-tokenization of
  the decoded text keeps the policy gradient attached to what was really drawn.

  `adapter`/`clean` default to the decomposer and its subgoal cleaner; the probe
  passes the synthesizer and `llm_utils.cut_program_from_sample` to generate
  code the same way, so it shares the batching and EOS handling.
  """
  if clean is None:
    clean = prompts.clean_subgoal_sample
  model.set_adapter(adapter)
  # Rollouts must be taken in eval mode: with LoRA dropout active the "greedy"
  # baseline would not be deterministic, which is the whole point of SCST.
  model.eval()
  tokenizer.padding_side = 'left'
  try:
    batch = tokenizer(contexts, return_tensors='pt', padding=True,
                      truncation=True, max_length=_MAX_SEQ_LEN.value).to('cuda')
    generated = model.generate(
        **batch,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=0.95 if do_sample else None,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
  finally:
    tokenizer.padding_side = 'right'

  new_tokens = generated[:, batch['input_ids'].shape[1]:]
  ids_per_example, texts = [], []
  for row in new_tokens:
    ids = row.tolist()
    # Keep everything up to and including the first EOS. Padding uses the same
    # id as EOS for Llama, so a finished row looks like [...tokens, eos, eos,
    # ...]; cutting at the first occurrence drops the padding and keeps the one
    # real EOS, which the policy gradient needs in order to shape when to stop.
    # Rows with no EOS ran to max_new_tokens and carry no padding.
    if tokenizer.eos_token_id in ids:
      ids = ids[:ids.index(tokenizer.eos_token_id) + 1]
    ids_per_example.append(ids)
    texts.append(clean(tokenizer.decode(ids, skip_special_tokens=True)))
  return ids_per_example, texts


def gt_prefix_program(record):
  """Partial program covering the ground-truth steps *before* this one.

  `data.compose_program` accumulates, so a step-j record needs steps 0..j-1
  replayed or its statement references undefined variables. Teacher-forcing the
  prefix with ground-truth code is what the step's `context` already assumes.
  """
  body = record['python_program'].splitlines()[1:]
  partial = None
  for statement in body[:record['step']]:
    partial = data.compose_program(record, statement.strip(), partial)
  return partial


def run_probe(model, tokenizer, probe_records):
  """Greedy quality on a *fixed* held-out slice. Returns a metrics dict.

  Why this exists: `mean_reward` is built from the temperature-1.0 *sample*, but
  `run_llm_experiment` evaluates at `--temperature=0.0`, so eval success is
  decided by the greedy subgoal. And every training step draws a fresh random
  batch whose between-task reward variance (`adv_std ~ 7-8`) gives a per-step
  standard error near 0.7 -- comparable to the total improvement a whole run
  might produce, so a real trend would be invisible. Scoring the same records
  every time removes that noise entirely.

  Three numbers, cheapest first:
    greedy_loss  summed CE of the true step code given the greedy subgoal
    exact_match  greedy subgoal string equals the ground-truth subgoal
    state_match  greedy subgoal -> greedy code -> execute -> same state as the
                 ground-truth step. The closest in-training analogue of eval
                 success, since it goes through the same parse-assemble-execute
                 path (`llm_utils.cut_program_from_sample` + `run_program`).
  """
  from spec_decomposition import llm_utils  # pylint: disable=g-import-not-at-top

  model.eval()
  total_loss, exact, state_ok, scored = 0.0, 0, 0, 0

  for start in range(0, len(probe_records), _BATCH_SIZE.value):
    batch = probe_records[start:start + _BATCH_SIZE.value]
    contexts = [r['context'] for r in batch]

    with torch.no_grad():
      _, greedy_subgoals = generate_subgoals(
          model, tokenizer, contexts, do_sample=False, temperature=None,
          max_new_tokens=_MAX_NEW_TOKENS.value)

      synth_ctx = [prompts.synthesizer_context(c, g, r['step'])
                   for c, g, r in zip(contexts, greedy_subgoals, batch)]
      greedy_lp, _ = score_completions(
          model, tokenizer, SYNTHESIZER, synth_ctx,
          [r['step_code'] for r in batch], append_eos=False, grad=False,
          lookaheads=[r['step_code_lookahead'] for r in batch])
      total_loss += (-greedy_lp).sum().item()

      _, greedy_codes = generate_subgoals(
          model, tokenizer, synth_ctx, do_sample=False, temperature=None,
          max_new_tokens=_MAX_NEW_TOKENS.value, adapter=SYNTHESIZER,
          clean=llm_utils.cut_program_from_sample)

    for record, subgoal, code in zip(batch, greedy_subgoals, greedy_codes):
      scored += 1
      exact += int(subgoal == record['subgoal'])
      # Compare executed states rather than parsing the rendered subgoal: the
      # subgoal shows only the prompt's I/O cases while `run_program` returns
      # all of them, and running both sides through the same executor sidesteps
      # that mismatch. Prior steps are teacher-forced with ground-truth code,
      # matching how the contexts were built.
      try:
        prefix = gt_prefix_program(record)
        gt_code = llm_utils.cut_program_from_sample(record['step_code'])
        want = llm_utils.run_program(
            data.compose_program(record, gt_code, prefix),
            record['inputs'], 'deepcoder')
        got = llm_utils.run_program(
            data.compose_program(record, code, prefix),
            record['inputs'], 'deepcoder')
      except Exception:  # pylint: disable=broad-except
        continue  # a non-executing prediction is simply not a match
      state_ok += int(got is not None and got == want)

  return {
      'probe/greedy_loss': total_loss / max(scored, 1),
      'probe/exact_match': exact / max(scored, 1),
      'probe/state_match': state_ok / max(scored, 1),
  }


def score_completions(model, tokenizer, adapter, contexts, completions,
                      append_eos, grad, lookaheads=None):
  """Summed log-prob of each completion under `adapter`, plus mean entropy.

  `lookaheads` is the text following each completion in the packed sequence, and
  supplying it is what makes these scores comparable to SFT's -- see
  `torch_utils.encode_example`. Omitting it costs ~7 nats on the completion's
  final token alone.
  """
  if lookaheads is None:
    lookaheads = [''] * len(contexts)
  examples = []
  for context, completion, lookahead in zip(contexts, completions, lookaheads):
    input_ids, labels, _ = torch_utils.encode_example(
        tokenizer, context, completion, _MAX_SEQ_LEN.value,
        append_eos=append_eos, lookahead=lookahead)
    examples.append({'input_ids': input_ids, 'labels': labels})
  batch = torch_utils.collate(examples, tokenizer.pad_token_id)
  batch = {k: v.to('cuda') for k, v in batch.items()}

  model.set_adapter(adapter)
  _set_mode(model, grad)
  with torch.set_grad_enabled(grad):
    logits = model(input_ids=batch['input_ids'],
                   attention_mask=batch['attention_mask']).logits
    return torch_utils.token_stats(logits, batch['labels'])


def score_sampled_ids(model, tokenizer, adapter, contexts, sampled_ids, grad):
  """Summed log-prob of already-sampled token ids under `adapter`."""
  examples = []
  for context, ids in zip(contexts, sampled_ids):
    context_ids = tokenizer(context, add_special_tokens=False)['input_ids']
    budget = _MAX_SEQ_LEN.value - 1 - len(ids)
    context_ids = context_ids[max(0, len(context_ids) - budget):]
    examples.append({
        'input_ids': [tokenizer.bos_token_id] + context_ids + ids,
        'labels': [-100] * (1 + len(context_ids)) + list(ids),
    })
  batch = torch_utils.collate(examples, tokenizer.pad_token_id)
  batch = {k: v.to('cuda') for k, v in batch.items()}

  model.set_adapter(adapter)
  _set_mode(model, grad)
  with torch.set_grad_enabled(grad):
    logits = model(input_ids=batch['input_ids'],
                   attention_mask=batch['attention_mask']).logits
    return torch_utils.token_stats(logits, batch['labels'])


def main(_):
  if _BATCH_SIZE.value < 2:
    raise ValueError(
        'Advantages are centered within the batch, so --batch_size must be at '
        'least 2; got {}.'.format(_BATCH_SIZE.value))

  exp_title = _EXP_TITLE.value or (
      f'{_RL_LOSS.value}_{_GENERALIZATION_TASK.value}')
  run_dir = os.path.join(_OUTPUT_DIR.value, exp_title)
  os.makedirs(run_dir, exist_ok=True)
  writer = SummaryWriter(os.path.join(run_dir, 'tb'))

  torch.manual_seed(_TRAIN_SEED.value)
  rng = random.Random(_TRAIN_SEED.value)

  model, tokenizer = load_model()
  trainable = [p for p in model.parameters() if p.requires_grad]
  logging.info('Trainable tensors: %d, parameters: %d',
               len(trainable), sum(p.numel() for p in trainable))
  optimizer = torch.optim.AdamW(trainable, lr=_LEARNING_RATE.value)

  records = data.load_step_records(
      os.path.join(_DATA_DIR.value,
                   f'{_GENERALIZATION_TASK.value}_train.jsonl'),
      _MAX_TRAIN_RECORDS.value)
  rng.shuffle(records)
  logging.info('Loaded %d step records.', len(records))

  probe_records = []
  if _PROBE_STEPS.value:
    probe_records = data.load_step_records(
        os.path.join(_DATA_DIR.value,
                     f'{_GENERALIZATION_TASK.value}_valid.jsonl'),
        _PROBE_RECORDS.value)
    logging.info('Probe: %d held-out step records every %d steps.',
                 len(probe_records), _PROBE_STEPS.value)

  cursor = 0
  running = collections.defaultdict(float)
  start_time = time.time()

  def maybe_probe(step):
    """Logs the fixed held-out metrics; step 0 gives the pre-training baseline."""
    if not probe_records:
      return
    stats = run_probe(model, tokenizer, probe_records)
    for key, value in stats.items():
      writer.add_scalar(key, value, step)
    writer.flush()
    logging.info(
        'probe @%d  greedy_loss=%.4f exact_match=%.3f state_match=%.3f',
        step, stats['probe/greedy_loss'], stats['probe/exact_match'],
        stats['probe/state_match'])

  # Baseline before any update: this is the SFT starting point every later probe
  # is compared against, and the number the grpo/supervised arms must beat.
  maybe_probe(0)

  for step in range(1, _NUM_TRAIN_STEPS.value + 1):
    optimizer.zero_grad(set_to_none=True)

    # Phase A: roll out and score the *whole* accumulation window before any
    # backward. The baseline is a batch mean, so its quality is set by how many
    # samples it averages -- and centering per micro-batch would make that
    # `batch_size` (4-8) no matter how large `grad_accum` is. With adv_std ~ 7 a
    # 4-sample baseline carries a standard error of 3.5, half the signal it is
    # meant to subtract. Everything here is no_grad (two generations and two
    # frozen-synthesizer forwards), so the only state carried into phase B is
    # token ids and floats: activation memory is unchanged, and the centering
    # window becomes the full effective batch. train.py has no accumulation and
    # centers over its entire batch, so this is also the more faithful form.
    window = []
    for _ in range(_GRAD_ACCUM.value):
      if cursor + _BATCH_SIZE.value > len(records):
        rng.shuffle(records)
        cursor = 0
      batch = records[cursor:cursor + _BATCH_SIZE.value]
      cursor += _BATCH_SIZE.value

      contexts = [r['context'] for r in batch]
      gt_codes = [r['step_code'] for r in batch]
      # What follows each completion in the packed sequence, so scoring uses the
      # tokenization SFT trained rather than the standalone one.
      code_la = [r['step_code_lookahead'] for r in batch]

      # 1. Roll out one sampled and one greedy subgoal per context.
      sampled_ids, sampled_texts = generate_subgoals(
          model, tokenizer, contexts, do_sample=True,
          temperature=_TEMPERATURE.value,
          max_new_tokens=_MAX_NEW_TOKENS.value)
      _, greedy_texts = generate_subgoals(
          model, tokenizer, contexts, do_sample=False, temperature=None,
          max_new_tokens=_MAX_NEW_TOKENS.value)

      # 2. Reward: how well the frozen synthesizer predicts the true step code
      #    given each subgoal. Summed CE, matching compute_weighted_cross_
      #    entropy(..., per_example=True).
      sampled_ctx = [prompts.synthesizer_context(c, s, r['step'])
                     for c, s, r in zip(contexts, sampled_texts, batch)]
      greedy_ctx = [prompts.synthesizer_context(c, g, r['step'])
                    for c, g, r in zip(contexts, greedy_texts, batch)]
      with torch.no_grad():
        sampled_lp, _ = score_completions(
            model, tokenizer, SYNTHESIZER, sampled_ctx, gt_codes,
            append_eos=False, grad=False, lookaheads=code_la)
        greedy_lp, _ = score_completions(
            model, tokenizer, SYNTHESIZER, greedy_ctx, gt_codes,
            append_eos=False, grad=False, lookaheads=code_la)
      window.append({
          'batch': batch,
          'contexts': contexts,
          'sampled_ids': sampled_ids,
          'sampled_texts': sampled_texts,
          'greedy_texts': greedy_texts,
          'synth_loss': -sampled_lp,
          'greedy_loss': -greedy_lp,
      })

    # 3. Self-critical advantage, centered once over the whole window
    #    (train.py:464-469, where the window is the entire batch).
    all_synth = torch.cat([w['synth_loss'] for w in window])
    all_greedy = torch.cat([w['greedy_loss'] for w in window])
    all_advantage = all_greedy - all_synth
    adv_avg = all_advantage.mean()
    adv_std = all_advantage.std()
    all_advantage = all_advantage - adv_avg
    if _NORMALIZE_ADVANTAGE.value:
      all_advantage = all_advantage / (adv_std + 1e-8)
    all_advantage = all_advantage.detach()

    running['mean_advantage'] += adv_avg.item()
    running['std_advantage'] += adv_std.item()
    running['mean_reward'] += (-all_synth).mean().item()
    # The greedy loss is what eval sees -- run_llm_experiment decodes at
    # temperature 0.0 -- whereas mean_reward comes from the temperature-1.0
    # sample. It was already computed for the baseline; logging it costs
    # nothing and it is the more predictive of the two.
    running['mean_greedy_loss'] += all_greedy.mean().item()
    running['num_identical'] += sum(
        s == g for w in window
        for s, g in zip(w['sampled_texts'], w['greedy_texts']))
    # Unlike num_identical (sampled vs greedy, a collapse diagnostic), this is
    # an accuracy: does the greedy subgoal match the ground truth? Both strings
    # come through clean_subgoal_sample and end in one newline.
    running['greedy_exact_match'] += sum(
        g == r['subgoal'] for w in window
        for g, r in zip(w['greedy_texts'], w['batch'])) / (
            _BATCH_SIZE.value * _GRAD_ACCUM.value)

    # Phase B: replay the window micro-batch by micro-batch for the backward
    # passes, using the advantages centered above.
    for index, entry in enumerate(window):
      batch = entry['batch']
      contexts = entry['contexts']
      sampled_ids = entry['sampled_ids']
      gt_subgoals = [r['subgoal'] for r in batch]
      subgoal_la = [r['subgoal_lookahead'] for r in batch]
      advantage = all_advantage[
          index * _BATCH_SIZE.value:(index + 1) * _BATCH_SIZE.value]

      # The RL and CE terms are summed in train.py:486, so their gradients add.
      # Backward each as soon as it is built instead of holding both graphs to
      # the end -- two live graphs over ~2500-token contexts OOM a 94 GB H100.
      rl_value, entropy_value, kl_value = 0.0, 0.0, 0.0

      if _RL_LOSS.value != 'supervised':
        # 4. Policy term on the tokens that were actually sampled.
        sum_log_probs, mean_entropy = score_sampled_ids(
            model, tokenizer, DECOMPOSER, contexts, sampled_ids, grad=True)
        rl_loss = -(advantage * sum_log_probs).mean()
        policy_term = rl_loss - _ENT_COEF.value * mean_entropy

        if _KL_BETA.value > 0:
          with torch.no_grad():
            ref_log_probs, _ = score_sampled_ids(
                model, tokenizer, REFERENCE, contexts, sampled_ids, grad=False)
          kl = (sum_log_probs - ref_log_probs).mean()
          policy_term = policy_term + _KL_BETA.value * kl
          kl_value = kl.item()

        (policy_term / _GRAD_ACCUM.value).backward()
        rl_value, entropy_value = rl_loss.item(), mean_entropy.item()
        del sum_log_probs, mean_entropy, rl_loss, policy_term

      # 5. Supervised CE on the ground-truth subgoal, which train.py always
      #    keeps in the total loss.
      #
      # append_eos=False even though this is a training target, because SFT's
      # target has no EOS either: `data.packed_example` labels the subgoal's own
      # tokens and nothing more, since inside a packed trajectory there is no end
      # after a subgoal -- the next step's context follows immediately. Asking
      # for EOS here scores a token the decomposer has never seen in this
      # position, measured at ~5.8 nats of the 22.4 `ce` started at. (The other
      # ~14 was the boundary token, now handled by `lookaheads` above.)
      # Generation does not need it: `hf_backend` stops on `_StopOnText`.
      gt_log_probs, _ = score_completions(
          model, tokenizer, DECOMPOSER, contexts, gt_subgoals,
          append_eos=False, grad=True, lookaheads=subgoal_la)
      mean_loss = -gt_log_probs.mean()
      ce_weight = (1.0 if _RL_LOSS.value == 'supervised'
                   else _CE_WEIGHT.value)
      (ce_weight * mean_loss / _GRAD_ACCUM.value).backward()
      ce_value = mean_loss.item()
      del gt_log_probs, mean_loss

      total_loss_value = rl_value - _ENT_COEF.value * entropy_value + (
          ce_weight * ce_value) + _KL_BETA.value * kl_value

      running['total_loss'] += total_loss_value / _GRAD_ACCUM.value
      running['rl_mean_loss'] += rl_value / _GRAD_ACCUM.value
      running['mean_loss'] += ce_value / _GRAD_ACCUM.value
      running['entropy'] += entropy_value / _GRAD_ACCUM.value
      running['kl'] += kl_value / _GRAD_ACCUM.value

    del window, all_advantage, all_synth, all_greedy

    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    model.set_adapter(DECOMPOSER)
    optimizer.step()

    if step % _LOG_STEPS.value == 0:
      # Every metric is already a per-step quantity: the window-level ones are
      # accumulated once per step, and the per-micro-batch ones are divided by
      # grad_accum as they are added. So the only averaging left is over steps.
      summary = {k: v / _LOG_STEPS.value for k, v in running.items()}
      for key, value in summary.items():
        writer.add_scalar(key, value, step)
      writer.flush()
      # Peak VRAM is how you size --batch_size: aim to fill the card (~80%),
      # since much below half is throughput left on the table.
      peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
      total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
      writer.add_scalar('peak_vram_gb', peak_gb, step)
      logging.info(
          'step %d (%.1fs) loss=%.4f rl=%.4f ce=%.4f reward=%.2f greedy=%.2f '
          'exact=%.3f adv=%.3f adv_std=%.3f ent=%.3f identical=%.1f/%d '
          'vram=%.1f/%.0fGB (%.0f%%)',
          step, time.time() - start_time, summary['total_loss'],
          summary['rl_mean_loss'], summary['mean_loss'],
          summary['mean_reward'], -summary['mean_greedy_loss'],
          summary['greedy_exact_match'], summary['mean_advantage'],
          summary['std_advantage'], summary['entropy'],
          summary['num_identical'],
          _BATCH_SIZE.value * _GRAD_ACCUM.value,
          peak_gb, total_gb, 100 * peak_gb / total_gb)
      running.clear()

    if _PROBE_STEPS.value and step % _PROBE_STEPS.value == 0:
      maybe_probe(step)

    if step % _SAVE_STEPS.value == 0 or step == _NUM_TRAIN_STEPS.value:
      model.set_adapter(DECOMPOSER)
      # PEFT writes a non-"default" adapter into <dir>/<adapter_name>, so this
      # lands in <run_dir>/decomposer -- which is the path to load later.
      model.save_pretrained(run_dir, selected_adapters=[DECOMPOSER])
      tokenizer.save_pretrained(adapter_dir(run_dir))
      logging.info('Saved decomposer adapter at step %d to %s',
                   step, adapter_dir(run_dir))

  writer.close()
  print(f'Done. Adapter in {adapter_dir(run_dir)}')


if __name__ == '__main__':
  torch.backends.cuda.matmul.allow_tf32 = True
  app.run(main)
