"""Loading the packed task records written by `build_sft_data.py`.

Each record is one task's whole ExeDec trajectory, cut into alternating segments
tagged with the role that has to predict them:

    [prefix, None] [subgoal_0, "decomposer"] [between, None] [code_0, "synth"] ...

Concatenating every segment gives the packed text. Two views are offered:

* `packed_example` -- one training sequence per task, with only one role's
  segments carrying a loss. Step j's evaluation context is exactly the trajectory
  prefix up to `Step j+1 computes:`, so with causal attention a single forward
  over the packed sequence reproduces the logits the per-step forwards produced,
  at ~1/3.3 of the tokens.
* `iter_steps` -- the older per-step view (`context`, `subgoal`, `step_code`),
  rebuilt by walking the same segments. SAD needs individual contexts to roll out
  from, so it consumes this rather than the packed form.
"""

import json

from spec_decomposition.llm_finetune import prompts

DECOMPOSER = 'decomposer'
SYNTHESIZER = 'synthesizer'
ROLES = (DECOMPOSER, SYNTHESIZER)


def load_records(path, max_records=None):
  """Reads a packed task-record JSONL written by build_sft_data.py."""
  records = []
  with open(path, 'r') as f:
    for line in f:
      records.append(json.loads(line))
      if max_records is not None and len(records) >= max_records:
        break
  return records


def packed_text(record):
  """The full trajectory text; every segment concatenated."""
  return ''.join(text for text, _ in record['segments'])


def iter_steps(record):
  """Rebuilds the per-step view: one dict per step of the task.

  Yields `{step, context, subgoal, step_code, ...}` where `context` is the
  decomposer prompt for that step (ending at `Step j computes:`) and the
  synthesizer prompt is `prompts.synthesizer_context(context, subgoal, step)`,
  exactly as before packing.

  Also yields `subgoal_lookahead` and `step_code_lookahead`: the text that
  follows each completion in the packed sequence. Pass them to
  `torch_utils.encode_example(..., lookahead=...)` so per-step scoring tokenizes
  the way SFT trained, which BPE otherwise breaks on the completion's trailing
  newline. It is not a constant -- a subgoal is always followed by
  `'\\nStep j code:'`, but the *last* step's code is followed by nothing, which
  is why 100% of decomposer spans but only 67% of synthesizer spans differ.
  """
  segments = list(record['segments'])
  prefix = ''
  step = 0
  pending = {}
  for index, (text, role) in enumerate(segments):
    if role is None:
      prefix += text
      continue
    # One character is enough to force or forbid the merge; the tokenizer never
    # merges across more than the immediately following character here.
    lookahead = segments[index + 1][0][:1] if index + 1 < len(segments) else ''
    if role == DECOMPOSER:
      pending = {'step': step, 'context': prefix, 'subgoal': text,
                 'subgoal_lookahead': lookahead}
    elif role == SYNTHESIZER:
      if not pending:
        raise ValueError(f'Synthesizer segment before a decomposer one in '
                         f'task {record.get("task_id")}')
      pending['step_code'] = text
      pending['step_code_lookahead'] = lookahead
      pending['dsl_program'] = record['dsl_program']
      pending['python_program'] = record['python_program']
      pending['inputs'] = record['inputs']
      pending['outputs'] = record['outputs']
      pending['task_id'] = record['task_id']
      yield pending
      pending = {}
      step += 1
    else:
      raise ValueError(f'Unknown segment role: {role}')
    prefix += text


def role_example(step_record, role):
  """The (prompt, completion) pair for one role of one *step* record.

  Used by SAD and by the data round-trip check, both of which work step by step.
  """
  if role == DECOMPOSER:
    return step_record['context'], step_record['subgoal']
  if role == SYNTHESIZER:
    return (
        prompts.synthesizer_context(
            step_record['context'], step_record['subgoal'],
            step_record['step']),
        step_record['step_code'],
    )
  raise ValueError(f'Unknown role: {role}')


def load_step_records(path, max_records=None):
  """Flattens packed task records into the per-step records SAD expects."""
  steps = []
  for record in load_records(path):
    for step_record in iter_steps(record):
      steps.append(step_record)
      if max_records is not None and len(steps) >= max_records:
        return steps
  return steps


def role_spans(record, role):
  """Character spans of the segments `role` must predict, in packed_text()."""
  spans, pos = [], 0
  for text, seg_role in record['segments']:
    if seg_role == role:
      spans.append((pos, pos + len(text)))
    pos += len(text)
  return spans


def packed_example(record, role, tokenizer, max_seq_len):
  """Tokenizes one task into a single sequence with only `role` unmasked.

  The whole trajectory is tokenized as one string and labels are recovered from
  character offsets, rather than tokenizing segment by segment. Measured on 60
  validation tasks (376 contexts), this matters:

    whole-string : 376/376 step contexts tokenize identically to `tok(context)`,
                   which is what inference feeds the model; 0 tokens straddle a
                   prompt/completion boundary.
    segment-wise : only 31.9% match -- BPE cannot merge across a segment break,
                   so every step after the first was shifted by 2-8 tokens
                   relative to what inference produces.

  So whole-string keeps training and inference on identical token sequences, and
  the boundaries here (always after a newline) happen never to be merged across,
  so the loss still starts exactly on the first token the model has to generate.

  Returns (input_ids, labels, num_truncated_tokens).
  """
  text = packed_text(record)
  spans = role_spans(record, role)
  encoded = tokenizer(text, add_special_tokens=False,
                      return_offsets_mapping=True)

  input_ids = [tokenizer.bos_token_id]
  labels = [-100]
  for token_id, (start, end) in zip(encoded['input_ids'],
                                    encoded['offset_mapping']):
    # Overlap, not containment. A completion ends in "\n" and the next prefix
    # begins with "\n", which BPE merges into one token (e.g. "]\n\n") -- that
    # happened on 200 of 240 spans. Requiring containment silently dropped the
    # final token of every completion, so the model was never taught to
    # terminate one. The token is genuinely part of the completion the model
    # must generate, so overlap is the correct rule.
    labelled = any(start < e and end > s for s, e in spans)
    input_ids.append(token_id)
    labels.append(token_id if labelled else -100)

  num_truncated = max(0, len(input_ids) - max_seq_len)
  if num_truncated:
    # Drop from the front, keeping BOS, so the latest steps always survive.
    input_ids = [tokenizer.bos_token_id] + input_ids[1 + num_truncated:]
    labels = [-100] + labels[1 + num_truncated:]
  return input_ids, labels, num_truncated


def compose_program(step_record, step_code_text, partial_program=None):
  """Appends one step's code to the partial program, as the eval loop does.

  Mirrors `run_llm_experiment.solve_problem_exedec`: on step 0 the function
  signature is borrowed from the ground-truth program, later steps strip the
  previous `return` and re-emit one for the newly bound variable.

  `step_code_text` is the bare statement, not the fenced block.
  """
  if partial_program is None:
    prefix = step_record['python_program'].splitlines()[0] + '\n'
  else:
    prefix = partial_program.rsplit('  return', 1)[0]
  new_var = step_code_text.split('=', 1)[0].strip()
  return f'{prefix}  {step_code_text.strip()}\n  return {new_var}'
