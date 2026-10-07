# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Train seq-to-seq model on random supervised training tasks."""

import collections
import functools
import os
import random
import sys
import time
from typing import Sequence
import re

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import linen as nn
from flax.core import freeze
import optax
from flax.training import train_state
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
import jax

import jax.numpy as jnp
import numpy as np
import tensorflow as tf

# pylint: disable=g-import-not-at-top

from models import base_models
from spec_decomposition import decomposition_models as models
from spec_decomposition import input_pipeline
from spec_decomposition import old_decode as decode
from tasks.lambdabeam import lambdabeam_dsl
from tasks.deepcoder import deepcoder_dsl
from tasks import operation_base
from tasks.robust_fill import dsl as robust_fill_dsl
from tasks.robust_fill import tokens as dsl_tokens

try:
  tf.config.set_visible_devices([], "GPU")
except Exception:
  pass

gfile = tf.io.gfile

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Fixed random seed for training.')
flags.DEFINE_float('lr', 2e-4, 'Learning rate.')
flags.DEFINE_float('weight_decay', 1e-1,
                   'Decay factor for AdamW-style weight decay.')
flags.DEFINE_integer('embedding_dim', 512, 'Embedding dimension.')
flags.DEFINE_integer('hidden_dim', 1024, 'Hidden dimension.')
flags.DEFINE_integer('num_heads', 4, 'Number of layers.')
flags.DEFINE_integer('num_layers', 3, 'Number of Transformer heads.')
flags.DEFINE_boolean('slow_decode', True, 'Use slow decoding for prediction?')
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate')
flags.DEFINE_float('attention_dropout_rate', 0.1, 'Attention dropout rate')

flags.DEFINE_string('dataset_dir', None,
                    'Directory to find TFRecord datasets for train and test.')
flags.DEFINE_string('experiment', 'NONE',
                    'Which compositional generalization experiment to use.')
flags.DEFINE_integer('per_device_batch_size', 16,
                     'Number of program tasks in a batch.')
flags.DEFINE_integer('num_examples', 4,
                     'Number of input/output strings per task.')
flags.DEFINE_integer('max_input_length', 120,
                     'Maximum number of characters in input/output strings.')
flags.DEFINE_integer('predict_max_input_length', 200,
                     'Maximum number of characters in input/output strings for '
                     'prediction.')
flags.DEFINE_integer('max_target_length', 200,
                     'Maximum number of characters in the target.')

flags.DEFINE_string('save_dir', None, 'Directory to save results to.')
flags.DEFINE_integer('num_train_steps', 500_000, 'Number of training steps.')
flags.DEFINE_integer('num_eval_steps', 10, 'Number of evaluation steps.')
flags.DEFINE_integer('num_quick_test_steps', 10,
                     'Number of test steps during training.')
flags.DEFINE_integer('num_final_test_steps', 10,
                     'Number of test steps after training is finished.')
flags.DEFINE_integer('log_freq', 2000, 'Number of steps between training logs.')
flags.DEFINE_integer('eval_freq', 10000, 'Number of steps between eval.')
flags.DEFINE_integer('predict_freq', 50000,
                     'Number of steps between prediction (beam search).')
flags.DEFINE_integer('checkpoint_freq', 50000,
                     'Number of steps between checkpoint saves.')
flags.DEFINE_bool('restore_checkpoints', True,
                  'Whether to restore from existing model checkpoints.')

flags.DEFINE_float('synthesizer_corrupted_next_part_rate', 0.0,
                   'The fraction of examples that use the corrupted next part.')

flags.DEFINE_bool('use_relative_attention', True,
                  'Whether to use relative positonal embeddings.')
flags.DEFINE_integer('num_position_buckets', 32,
                     'Number of relative attention position buckets.')
flags.DEFINE_integer('max_distance', 128,
                     'Max distance for relative attention positions.')
flags.DEFINE_integer('max_program_cross_embed_distance', 128,
                     'Max distance for relative attention positions.')
flags.DEFINE_bool('aligned_relative_attention', True,
                  'Whether to align relative attention positions between '
                  'targets and encoded I/O examples.')

flags.DEFINE_enum('dataset_type', 'deepcoder',
                  ['robustfill', 'deepcoder', 'lambdabeam'],
                  'The kind of dataset to use.')
flags.DEFINE_enum('model_type', 'spec_decomposer_model',
                  ['spec_decomposer_model', 'synthesizer_model', 'joint_model',
                   'baseline_model'],
                  'Which model to train.')

flags.DEFINE_bool('predict_only', False,
                  'Whether to only do beam search prediction, no training.')

flags.DEFINE_bool(
    'compute_false_positives', True,
    'Whether to compute false positives.')

_DECOMPOSITION_MODE = flags.DEFINE_enum(
    'decomposition_mode', 'standard',
    ['standard', 'gated', 'residual', 'coupled'],
    'Which mode to use for the decomposition model.')
_SYNTHESIZER_PATH_FORMAT = flags.DEFINE_string(
    'synthesizer_path_format', None,
    'Directory with saved weights for Synthesizer.')

flags.DEFINE_integer(
    'synthesizer_max_input_length', 120,
    'Maximum number of characters in the model input.')
flags.DEFINE_integer(
    'synthesizer_predict_max_input_length', 200,
    'Maximum number of characters in the model input for prediction.')
flags.DEFINE_integer(
    'synthesizer_max_target_length', 100,
    'Maximum number of tokens in the prediction target.')
flags.DEFINE_integer(
    'synthesizer_max_distance', 128,
    'Options for max relative attention distance.')
flags.DEFINE_integer(
    'synthesizer_max_program_cross_embed_distance', 128,
    'Options for max relative attention distance.')

_RL_LOSS = flags.DEFINE_enum(
    'rl_loss', 'supervised',
    ['supervised', 'grpo'],
    'Which SAD training option to use.')
_ENTROPY_COEF = flags.DEFINE_float(
    'entropy_coef', 0.001,
    'Weight of the entropy bonus in the SAD (grpo) loss.')
_SUP_LOSS_WEIGHT = flags.DEFINE_float(
    'sup_loss_weight', 1.0,
    'Weight of the supervised cross-entropy term in the SAD (grpo) loss. 0 '
    'drops it (the L_sup ablation).')



def create_learning_rate_scheduler(
    base_learning_rate=0.5,
    factors='constant * linear_warmup * rsqrt_normalized_decay',
    warmup_steps=16000,
    decay_factor=0.5,
    steps_per_decay=50000,
    steps_per_cycle=100000):
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    base_learning_rate: float, the starting constant for the lr schedule.
    factors: a string with factors separated by '*' that defines the schedule.
    warmup_steps: how many steps to warm up for in the warmup schedule.
    decay_factor: The amount to decay the learning rate by.
    steps_per_decay: How often to decay the learning rate.
    steps_per_cycle: Steps per cycle when using cosine decay.

  Returns:
    A function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split('*')]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == 'constant':
        ret *= base_learning_rate
      elif name == 'linear_warmup':
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(1.0, step - warmup_steps))
      elif name == 'rsqrt_normalized_decay':
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'decay_every':
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == 'cosine_decay':
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_decay))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError('Unknown factor %s.' % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


def compute_weighted_cross_entropy(logits, targets, weights=None, per_example=False):
  """
  Computes weighted cross entropy and entropy for log probs and targets.

  Args:
      logits: [batch, seq_len, num_classes] float array
      targets: [batch, seq_len] int array
      weights: None or [batch, seq_len] float array
      per_example: bool, if True returns [batch] per-example loss, else scalar sum

  Returns:
      If per_example=True: (per_example_loss: [batch], normalizing_factor: scalar)
      If per_example=False: (scalar_loss, normalizing_factor: scalar)
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))

  onehot_targets = common_utils.onehot(targets, logits.shape[-1])
  loss = -jnp.sum(onehot_targets * nn.log_softmax(logits), axis=-1)  # [batch, seq_len]
  normalizing_factor = jnp.prod(jnp.asarray(targets.shape))

  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  if per_example:
    per_example_loss = loss.sum(axis=-1)  # sum over seq_len → [batch]
    per_example_norm = jnp.sum(weights, axis=-1)
    return per_example_loss, per_example_norm
  else:
    scalar_loss = loss.sum()
    return scalar_loss, normalizing_factor


def compute_weighted_accuracy(logits, targets, weights=None, per_example=False):
  """Computes weighted accuracy for log probs and targets.

  Args:
   logits: `[batch, length, num_classes]` float array.
   targets: categorical targets `[batch, length]` int array.
   weights: None or array of shape [batch, length, 1]

  Returns:
    Tuple of scalar accuracy and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  acc = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  normalizing_factor = jnp.prod(jnp.asarray(targets.shape))

  if per_example:
    per_example_acc = (acc * weights).sum(axis=-1)  # sum over seq_len → [batch]
    per_example_norm = jnp.sum(weights, axis=-1)
    return per_example_acc, per_example_norm

  if weights is not None:
    acc = acc * weights
    normalizing_factor = weights.sum()

  return acc.sum(), normalizing_factor


def compute_weighted_entropy(logits, weights=None):
  """Computes weighted entropy for logits.

  Args:
   logits: `[batch, length, num_classes]` float array.
   weights: None or array of shape [batch, length] or [batch, length, 1]

  Returns:
    Tuple of scalar total entropy and batch normalizing factor.
  """
  # Numerical stability: use log_softmax directly
  log_probs = jax.nn.log_softmax(logits)
  probs = jnp.exp(log_probs)
  
  # Entropy per position: -sum(p * log_p)
  # result shape: [batch, length]
  entropy = -jnp.sum(probs * log_probs, axis=-1)
  
  normalizing_factor = jnp.prod(jnp.asarray(entropy.shape))
  
  if weights is not None:
    # Ensure weights are broadcastable to [batch, length]
    if weights.ndim == entropy.ndim + 1:
      weights = jnp.squeeze(weights, axis=-1)
    
    entropy = entropy * weights
    normalizing_factor = weights.sum()

  return entropy.sum(), normalizing_factor


def compute_metrics(logits, targets, weights):
  """Computes summary metrics."""
  loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights)
  acc, _ = compute_weighted_accuracy(logits, targets, weights)
  entropy, _ = compute_weighted_entropy(logits, weights)
  metrics = {
      'loss': loss,
      'accuracy': acc,
      'entropy': entropy,
      'denominator': weight_sum,
  }
  metrics = jax.lax.psum(metrics, 'batch')
  return metrics



def split_subgoals(logits_flat, sep_token=3, eos_token=2, pad_token=0, pad_size=60):
    """
    logits_flat: (batch, seq_len)
    Returns: (batch, n_subgoals, seq_len)
    Each subgoal left-aligned, ends with exactly one EOS, rest padded with pad_token.
    Fully JAX-friendly (works with jit/pmap).
    """

    n_subgoals = 4 if FLAGS.dataset_type == 'robustfill' else 3
    _, seq_len = logits_flat.shape
    pad_len = pad_size - seq_len
    logits_flat = jnp.pad(logits_flat, ((0,0), (0,pad_len)), constant_values=pad_token)

    # 1. Identify subgoal ids
    is_sep = (logits_flat == sep_token)
    subgoal_ids = jnp.cumsum(is_sep, axis=1)
    subgoal_ids = jnp.minimum(subgoal_ids, n_subgoals - 1)

    # 2. Mask out separators and EOS tokens
    clean_tokens = jnp.where((logits_flat == sep_token) | (logits_flat == eos_token),
                             0, logits_flat)

    # 3. One-hot mask per subgoal
    one_hot_masks = jax.nn.one_hot(subgoal_ids, n_subgoals)  # (batch, seq_len, n_subgoals)
    subgoal_tokens = clean_tokens[:, :, None] * one_hot_masks
    subgoal_tokens = jnp.transpose(subgoal_tokens, (0, 2, 1))  # (batch, n_subgoals, seq_len)

    # 4. Left-align using sort trick
    # Create mask: 0 for tokens, 1 for pad
    sort_key = jnp.where(subgoal_tokens != 0, 0, 1)
    indices = jnp.argsort(sort_key, axis=-1, stable=True)
    aligned_tokens = jnp.take_along_axis(subgoal_tokens, indices, axis=-1)

    # 5. Append EOS after last non-zero token
    def add_eos(seq):
        # Count non-zero tokens
        counts = jnp.sum(seq != 0)
        eos_pos = jnp.minimum(counts, seq.shape[0]-1)
        # Set EOS at eos_pos
        seq = seq.at[eos_pos].set(eos_token)
        return seq

    # aligned_tokens = jax.vmap(jax.vmap(add_eos))(aligned_tokens)
    return aligned_tokens.astype(jnp.int32)



# Train / eval / decode step functions.
# -----------------------------------------------------------------------------

def train_step(state, inputs, outputs, targets, config, learning_rate_fn,
               dropout_rng, sep_token, bos_token, eos_token, synth_config, synth_params=None, synth_targets=None):

      def main_forward(params, inputs, outputs, targets, dropout_rng):
        return models.DecomposeAttentionTransformer(config).apply(
            {'params': params},
            inputs,
            outputs,
            targets,
            rngs={'dropout': dropout_rng},
        )

      def synth_forward(params, inputs, subgoals, targets):
          return models.DecomposeAttentionTransformer(synth_config).apply(
              {'params': params},
              inputs,
              subgoals,
              targets,
          )

      dropout_rng, sample_rng, new_dropout_rng = jax.random.split(dropout_rng, 3)

      weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)

      def loss_fn(params):
        """Loss function used for training."""
        logits = main_forward(
            params,
            inputs,
            outputs,
            targets,
            dropout_rng,
        )

        loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights)
        mean_loss = loss / weight_sum
        total_loss = mean_loss

        synth_logits, greedy_synth_logits, rl_loss, adv_avg, adv_std = None, None, None, None, None
        if _DECOMPOSITION_MODE.value != 'standard' and FLAGS.model_type == 'spec_decomposer_model':
          
          temp = 1
          subgoals_sampled_ids = jax.random.categorical(sample_rng, logits / temp, axis=-1)

          subgoals_sampled = split_subgoals(subgoals_sampled_ids, sep_token=sep_token, eos_token=eos_token, pad_size=FLAGS.synthesizer_max_input_length)
          synth_weights = jnp.where(
          jnp.logical_and(synth_targets > 0,
                        jnp.logical_and(synth_targets != bos_token, 
                                        synth_targets != eos_token)),
          1.0, 0.0).astype(jnp.float32)

          synth_logits = synth_forward(
              synth_params,
              inputs,
              subgoals_sampled,
              synth_targets,
          )
          synth_logits = jax.lax.stop_gradient(synth_logits)

          synth_loss_per_example, _ = compute_weighted_cross_entropy(
            synth_logits, synth_targets, synth_weights, per_example=True)

          greedy_subgoals = split_subgoals(jnp.argmax(logits, axis=-1), sep_token=sep_token, eos_token=eos_token, pad_size=FLAGS.synthesizer_max_input_length)

          greedy_synth_logits = synth_forward(
              synth_params,
              inputs,
              greedy_subgoals,
              synth_targets,
          )
          greedy_synth_logits = jax.lax.stop_gradient(greedy_synth_logits)

          greedy_loss_per_example, _ = compute_weighted_cross_entropy(
            greedy_synth_logits, synth_targets, synth_weights, per_example=True)

          advantage = (greedy_loss_per_example - synth_loss_per_example)

          # Adv normalization
          adv_avg = advantage.mean() 
          adv_std = advantage.std()
          advantage = (advantage - adv_avg) # / (adv_std + 1e-8)

          sampled_log_probs = jax.nn.log_softmax(logits)
          sampled_log_probs = jnp.take_along_axis(
              sampled_log_probs, subgoals_sampled_ids[..., None], axis=-1
          ).squeeze(-1)
                    
          sum_log_probs_per_ex = jnp.sum(sampled_log_probs * weights, axis=-1)
          
          rl_loss = - jnp.mean(jax.lax.stop_gradient(advantage) * sum_log_probs_per_ex)

          total_ent, norm_ent = compute_weighted_entropy(logits, weights)
          mean_entropy = total_ent / norm_ent 

          if _RL_LOSS.value == "supervised":
            total_loss =  mean_loss
          elif _RL_LOSS.value == 'grpo':
            total_loss = (rl_loss - _ENTROPY_COEF.value * mean_entropy
                          + _SUP_LOSS_WEIGHT.value * mean_loss)
          else:
            raise NotImplementedError(f'Unkown RL loss option: {_RL_LOSS.value}')
        
        return total_loss, (logits, synth_logits, greedy_synth_logits, rl_loss, adv_avg, adv_std) 

      step = state.step
      lr = learning_rate_fn(step)
      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      (_, (logits, sampled_synth_logits, greedy_synth_logits, rl_mean_loss, avg_advantage, std_advantage)), grads = grad_fn(state.params)

      # Get metrics.
      metrics = compute_metrics(logits, targets, weights)
      metrics['learning_rate'] = lr

      aux_metrics = None
      if _DECOMPOSITION_MODE.value != 'standard' and FLAGS.model_type == 'spec_decomposer_model':
        synth_weights = jnp.where(
          jnp.logical_and(synth_targets > 0,
                        jnp.logical_and(synth_targets != bos_token, 
                                        synth_targets != eos_token)),
          1.0, 0.0).astype(jnp.float32)

        aux_metrics = compute_metrics(greedy_synth_logits, synth_targets, synth_weights)
        aux_metrics = {'greedy ' + k: v for k,v in aux_metrics.items()}

        sampled_metrics = compute_metrics(sampled_synth_logits, synth_targets, synth_weights)
        aux_metrics.update({'sampled ' + k: v for k,v in sampled_metrics.items()})

        aux_metrics.update({'rl_mean_loss': rl_mean_loss, 'mean_advantage': avg_advantage, 'std_advantage': std_advantage})

      grads = jax.lax.pmean(grads, 'batch')
      new_optimizer = state.apply_gradients(grads=grads)

      return new_optimizer, metrics, aux_metrics, new_dropout_rng


def eval_step(state, inputs, outputs, targets, sep_token, eos_token, config, synth_state=None, synth_params=None, synth_targets=None):
    """Collect metrics for evaluation during training."""
    weights = jnp.where(
        jnp.logical_and(targets > 0,
                        jnp.logical_and(targets != config.base_config.bos_token,
                                        targets != eos_token)),
        1, 0).astype(jnp.float32)

    # Use state.params to apply the model
    logits = state.apply_fn(
        {'params': state.params},  # Pass the parameters stored in the TrainState
        inputs,
        outputs,
        targets
    )

    metrics = compute_metrics(logits, targets, weights)

    aux_metrics = None
    if _DECOMPOSITION_MODE.value != 'standard' and FLAGS.model_type == 'spec_decomposer_model':
      subgoals = split_subgoals(jnp.argmax(logits, axis=-1), sep_token=sep_token, eos_token=eos_token, pad_size=FLAGS.synthesizer_max_input_length)
      synth_logits = synth_state.apply_fn(
          {'params': synth_params},
          inputs,
          subgoals,
          synth_targets
      )

      aux_weights = jnp.where(
          jnp.logical_and(synth_targets > 0,
                        jnp.logical_and(synth_targets != config.base_config.bos_token,
                                        synth_targets != eos_token)),
          1, 0).astype(jnp.float32)
      aux_metrics = compute_metrics(synth_logits, synth_targets, aux_weights)
    return metrics, aux_metrics


def initialize_cache(inputs, outputs, targets, max_decode_len, config):
  """Initializes a cache for a given input shape and max decode length."""
  target_shape = (targets.shape[0], max_decode_len)
  dtype = config.base_config.dtype
  initial_variables = models.DecomposeAttentionTransformer(config).init(
      jax.random.PRNGKey(0),
      jnp.ones(inputs.shape, dtype),
      jnp.ones(outputs.shape, dtype),
      jnp.ones(target_shape, dtype))
  return initial_variables['cache']


def predict_step(state,
                 inputs,
                 outputs,
                 cache,
                 beam_size,
                 eos_token,
                 max_decode_len,
                 config,
                 slow_decode=True):
    """Predict translation with fast decoding beam search on a batch."""
    # Prepare transformer fast-decoder call for beam search: for beam search, we
    # need to set up our decoder model to handle a batch size equal to
    # batch_size * beam_size, where each batch item's data is expanded in-place
    # rather than tiled.

    flat_encoded = decode.flat_batch_beam_expand(
        models.DecomposeAttentionTransformer(config).apply(
            {'params': freeze(state.params)}, 
            inputs,
            outputs,
            method=models.DecomposeAttentionTransformer.encode),
        beam_size
    )

    encoded_padding_mask = jnp.where(outputs > 0, 1, 0).astype(jnp.float32)
    flat_encoded_padding_mask = decode.flat_batch_beam_expand(
        encoded_padding_mask, beam_size
    )

    if slow_decode:
        def tokens_ids_to_logits(flat_ids):
            """Token slice to logits from decoder model."""
            # --> [batch * beam, 1, vocab]
            flat_logits = models.DecomposeAttentionTransformer(config=config).apply(
                {'params': freeze(state.params)},
                flat_ids,
                flat_encoded,
                flat_encoded_padding_mask,
                method=models.DecomposeAttentionTransformer.decode
            )
            return flat_logits
    else:
        def tokens_ids_to_logits(flat_ids, flat_cache):
            """Token slice to logits from decoder model."""
            # --> [batch * beam, 1, vocab]
            flat_logits, new_vars = models.DecomposeAttentionTransformer(
                config=config).apply(
                    {'params': freeze(state.params), 'cache': flat_cache},
                    flat_ids,
                    flat_encoded,
                    flat_encoded_padding_mask,
                    mutable=['cache'],
                    method=models.DecomposeAttentionTransformer.decode
                )
            new_flat_cache = new_vars['cache']
            # Remove singleton sequence-length dimension:
            # [batch * beam, 1, vocab] --> [batch * beam, vocab]
            flat_logits = flat_logits.squeeze(axis=1)
            return flat_logits, new_flat_cache

    # Using the above-defined single-step decoder function, run a
    # beam search over possible sequences given input encoding.
    beam_seqs, _ = decode.beam_search(
        inputs,
        cache,
        tokens_ids_to_logits,
        beam_size=beam_size,
        alpha=0.0,
        bos_token=config.base_config.bos_token,
        eos_token=eos_token,
        max_decode_len=max_decode_len,
        slow_decode=slow_decode
    )

    # Beam search returns [n_batch, n_beam, n_length] with beam dimension
    # sorted in increasing order of log-probability.
    return beam_seqs


# Util functions for prediction
# -----------------------------------------------------------------------------


def run_program(program, inputs):
  """Returns a list of outputs from running a program on a list of inputs.

  Args:
    program: A program returned from `decode_program()`.
    inputs: A list of inputs as returned by `decode_io`.
  """

  if FLAGS.dataset_type == 'robustfill':
    return [program(i) for i in inputs]
  elif FLAGS.dataset_type == 'deepcoder' or FLAGS.dataset_type == 'lambdabeam':
    # `program` is a lambdabeam_dsl.Statement or lambdabeam_dsl.Program.
    if program is None:
      return [None] * len(inputs)
    if FLAGS.dataset_type == 'lambdabeam':
      initial_states = [lambdabeam_dsl.ProgramState.from_str(i) for i in inputs]
      if FLAGS.model_type == 'baseline_model':
        result_states = [program.run(state.state) for state in initial_states]
      else:
        result_states = [program.run(state) for state in initial_states]

      """outputs = [lambdabeam_dsl.result_to_str(result_state.get_output())
                   if result_state and result_state.get_output() else None
                   for result_state in result_states]"""
      outputs = []
      for rs in result_states:
        if rs is not None:
          if isinstance(rs.get_output(), (int, bool)):
            outputs.append(lambdabeam_dsl.result_to_str(rs.get_output()))
          elif isinstance(rs.get_output(), list) and all(ele is not None for ele in rs.get_output()):
            outputs.append(lambdabeam_dsl.result_to_str(rs.get_output()))
          else:
            outputs.append(None)

        else:
          outputs.append(None)

    elif FLAGS.dataset_type == 'deepcoder':
      initial_states = [deepcoder_dsl.ProgramState.from_str(i) for i in inputs]
      if FLAGS.model_type == 'baseline_model':
        result_states = [program.run(state.state) for state in initial_states]
      else:
        result_states = [program.run(state) for state in initial_states]
      outputs = [deepcoder_dsl.result_to_str(result_state.get_output())
                 if result_state else None
                 for result_state in result_states]
    return outputs
  else:
    raise ValueError('Unhandled dataset_type {}'.format(FLAGS.dataset_type))


def eval_predicted_spec_decomposer_model(predicted, ground_truth, decode_spec):
  """Evaluate predicted program beams."""
  beams_target = [decode_spec(beam) for beam in predicted[::-1]]
  success = ground_truth in beams_target
  if success:
    return ground_truth, 1
  else:
    return beams_target[0], 0


def eval_predicted_synthesizer_model(predicted, inputs, outputs,
                                     decode_program, gt=None):
  """Evaluate predicted program beams."""
  best_program_str, best_score = None, -99

  # predicted shape [beam_size, length]
  for beam in predicted[::-1]:
    if FLAGS.dataset_type == 'robustfill':
      program = decode_program(beam)
      program = gt

      try:
        p_outs = run_program(program, inputs)
        score = (np.sum([p_out == out for p_out, out in zip(p_outs, outputs)])
                 / len(inputs))
        program_str = program.to_string()
      except:  # pylint: disable=bare-except
        score = -1
        program_str = 'did not compile'

    elif FLAGS.dataset_type == 'deepcoder' or FLAGS.dataset_type == 'lambdabeam':
      
      statement = decode_program(beam)
      if statement is None:
        score = -1
        program_str = 'did not compile'
      else:
        try:
          p_outs = run_program(statement, inputs)
          score = (np.sum([p_out == out for p_out, out in zip(p_outs, outputs)])
                   / len(inputs))
          program_str = str(statement)
        except (lambdabeam_dsl.RunError, deepcoder_dsl.RunError):
          score = -0.5
          program_str = 'encountered RunError'

    else:
      raise ValueError('Unhandled dataset_type {}'.format(FLAGS.dataset_type))

    if score > best_score:
      best_program_str, best_score = program_str, score

    if best_score >= 1:  # Found solution.
      break

  # best_program_str could be None if no RobustFill program compiles.
  return best_program_str, best_score


def pad_examples(x, desired_batch_size):
  """Expand batch to desired size by repeating last slice."""
  batch_pad = desired_batch_size - x.shape[0]
  tile_dims = [1] * len(x.shape)
  tile_dims[0] = batch_pad
  return np.concatenate([x, np.tile(x[-1], tile_dims)], axis=0)


def tohost(x):
  """Collect batches from all devices to host and flatten batch dimensions."""
  n_device, n_batch, *remaining_dims = x.shape
  return x.reshape((n_device * n_batch,) + tuple(remaining_dims))


def per_host_sum_pmap(in_tree):
  """Executes psum on in_tree's leaves over one device per host."""
  host2devices = collections.defaultdict(list)
  for d in jax.devices():
    host2devices[d.host_id].append(d)
  devices = [host2devices[k][0] for k in host2devices]
  host_psum = jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i', devices=devices)
  def pre_pmap(xs):
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (1,) + x.shape),
                                  xs)
  def post_pmap(xs):
    return jax.tree_util.tree_map(lambda x: x[0], xs)
  return post_pmap(host_psum(pre_pmap(in_tree)))


def shorten(key):
  splits = key.split('_')
  return ''.join(s[0] for s in splits)


def load_data(batches, rng):
  """Returns info from batches in dictionaries."""
  data_dict = common_utils.shard(batches)

  if FLAGS.model_type == 'synthesizer_model':
    rng, new_rng = jax.random.split(rng)
    # shape: [num_devices, per_device_batch_size]
    is_corrupted = (
        jax.random.uniform(rng, shape=data_dict['outputs'].shape[:2])
        < FLAGS.synthesizer_corrupted_next_part_rate)
    outputs = np.where(is_corrupted[..., None, None],
                       data_dict['corrupted_outputs'],
                       data_dict['outputs'])
  else:
    outputs = data_dict['outputs']
    new_rng = rng

  synth_targets = None
  if _DECOMPOSITION_MODE.value != 'standard' and FLAGS.model_type == 'spec_decomposer_model':
    synth_targets = data_dict['synth_target']

  return (data_dict['inputs'],
          outputs,
          data_dict['target'],
          synth_targets,
          new_rng)


def print_and_log(s):
  logging.info(s)
  print(s, flush=True)


def load_frozen_synthesizer(checkpoint_dir, config, io_shape, target_shape):
  """Load synthesizer params and return frozen params PyTree."""
  logging.info(f"Loading synthesizer from {checkpoint_dir} for {_DECOMPOSITION_MODE.value} training.")
  model = models.DecomposeAttentionTransformer(config)

  variables = model.init(
    jax.random.PRNGKey(0),
    jnp.ones(io_shape, jnp.float32),
    jnp.ones(io_shape, jnp.float32),
    jnp.ones(target_shape, jnp.float32),
  )

  synth_state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=optax.adam(0.0),  # dummy optimizer
  )

  if "gs://" in checkpoint_dir:
    # Restore the checkpoint (which was saved using flax.optim)
    old_optimizer = checkpoints.restore_checkpoint(checkpoint_dir, target=None)  # Specify None to get the full optimizer
    # Extract parameters from the old flax optimizer
    params = old_optimizer["target"]  # 'target' in flax.optim corresponds to 'params' in optax
    # Define the optimizer with optax
    optimizer = optax.adamw(learning_rate=1e-3, b1=0.9, b2=0.98, eps=1e-9, weight_decay=0.01)
    synth_state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=optimizer
    )
    synth_state = synth_state.replace(step=499999)
  else:
    params = variables['params']
    optimizer = optax.adamw(learning_rate=1e-3, b1=0.9, b2=0.98, eps=1e-9, weight_decay=0.01)
    synth_state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=optimizer
    )
    synth_state = checkpoints.restore_checkpoint(checkpoint_dir, synth_state)
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')]
    if checkpoint_files:
      steps = [int(re.search(r'checkpoint_(\d+)', f).group(1)) for f in checkpoint_files]
      step = max(steps)
    else:
      step = 0
    synth_state = synth_state.replace(step=step)
  print_and_log('Found model checkpointed at step %d.' % synth_state.step)
  print_and_log(f'Loaded model with {sum(jnp.size(p) for p in jax.tree_util.tree_leaves(params))} parameters.')

  frozen_state = synth_state.replace(params=freeze(synth_state.params))

  # Replicate across devices for pmap
  replicated_state = jax.device_put_replicated(frozen_state, jax.local_devices())
  frozen_synth_params = jax.tree_map(jax.lax.stop_gradient, replicated_state.params)

  return replicated_state, frozen_synth_params


def discard_last_subgoal(targets, sep_token=3, eos_token=2, pad_token=0):
    targets = np.asarray(targets)

    def remove_last(row):
        row = row.copy()
        eos_idx = np.flatnonzero(row == eos_token)
        sep_idx = np.flatnonzero(row == sep_token)
        if eos_idx.size == 0 or sep_idx.size == 0:
            return row

        last_eos = int(eos_idx[-1])
        last_sep = int(sep_idx[-1])
        row[last_sep] = eos_token
        if last_sep + 1 <= last_eos:
            row[last_sep + 1:last_eos + 1] = pad_token
        return row

    if targets.ndim == 2:
        out = np.stack([remove_last(row) for row in targets], axis=0)
        # Keep previous contract: 2D in -> 3D out with leading dim 1.
        return np.expand_dims(out, axis=0)
    if targets.ndim == 3:
        return np.stack(
            [np.stack([remove_last(row) for row in batch], axis=0) for batch in targets],
            axis=0,
        )
    raise ValueError(f"discard_last_subgoal expects 2D or 3D targets, got shape {targets.shape}")


def main(_):

  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  random.seed(FLAGS.seed)

  if not gfile.isdir(FLAGS.save_dir):
    gfile.makedirs(FLAGS.save_dir)

  # Distinguishes different runs in TensorBoard and checkpoints.
  hparam_dict = {
      'attention_dropout_rate': FLAGS.attention_dropout_rate,
      'aligned_relative_attention': FLAGS.aligned_relative_attention,
      'dropout_rate': FLAGS.dropout_rate,
      'experiment': FLAGS.experiment,
      'embedding_dim': FLAGS.embedding_dim,
      'hidden_dim': FLAGS.hidden_dim,
      'lr': FLAGS.lr,
      'max_distance': FLAGS.max_distance,
      'max_program_cross_embed_distance': (
          FLAGS.max_program_cross_embed_distance),
      'num_position_buckets': FLAGS.num_position_buckets,
      'seed': FLAGS.seed,
      'synthesizer_corrupted_next_part_rate': (
          FLAGS.synthesizer_corrupted_next_part_rate),
      'use_relative_attention': FLAGS.use_relative_attention,
      'decomposition_mode': FLAGS.decomposition_mode
  }
  hparam_str = ','.join([f'{shorten(k)}={v}' for k, v in hparam_dict.items()])

  # Number of local devices for this host.
  n_devices = jax.local_device_count()

  if jax.process_index() == 0:
    summary_writer = tensorboard.SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', hparam_str))
  else:
    # Only to appease pytype.
    summary_writer = tensorboard.SummaryWriter('')

  batch_size = FLAGS.per_device_batch_size * n_devices
  io_shape = (FLAGS.per_device_batch_size,
              FLAGS.num_examples,
              FLAGS.max_input_length)
  predict_io_shape = (FLAGS.per_device_batch_size,
                      FLAGS.num_examples,
                      FLAGS.predict_max_input_length)
  target_shape = (FLAGS.per_device_batch_size, FLAGS.max_target_length)
  synth_target_shape = (FLAGS.per_device_batch_size, FLAGS.synthesizer_max_target_length)

  # Setup DSL
  # ---------------------------------------------------------------------------

  # Build token tables.
  if FLAGS.dataset_type == 'robustfill':
    spec_vocab = robust_fill_dsl.CHARACTER + input_pipeline.SEPARATOR_TOKEN
    spec_id_token_table = {i+3: token for i, token in enumerate(spec_vocab)}
    bos_id = 1
    eos_id = 2
    spec_id_token_table[bos_id] = robust_fill_dsl.BOS
    spec_id_token_table[eos_id] = robust_fill_dsl.EOS
    spec_token_id_table = {token: id
                           for id, token in spec_id_token_table.items()}
    spec_vocab_size = len(spec_token_id_table) + 1  # For padding.
    program_id_token_table, _ = dsl_tokens.build_token_tables()
    program_vocab_size = len(program_id_token_table) + 1
    sep_id = spec_token_id_table[input_pipeline.SEPARATOR_TOKEN]

  elif FLAGS.dataset_type == 'deepcoder':
    id_to_token, token_to_id = deepcoder_dsl.vocab_tables()
    bos_id, eos_id = deepcoder_dsl.BOS_ID, deepcoder_dsl.EOS_ID
    vocab_size = len(id_to_token)  # Already includes padding.#

    spec_vocab_size = program_vocab_size = vocab_size
    program_id_token_table = spec_id_token_table = id_to_token
    spec_token_id_table = token_to_id
    sep_id = deepcoder_dsl.SEP_ID
  elif FLAGS.dataset_type == 'lambdabeam':
    id_to_token, token_to_id = lambdabeam_dsl.vocab_tables()
    bos_id, eos_id = lambdabeam_dsl.BOS_ID, lambdabeam_dsl.EOS_ID
    vocab_size = len(id_to_token)  # Already includes padding.

    spec_vocab_size = program_vocab_size = vocab_size
    program_id_token_table = spec_id_token_table = id_to_token
    spec_token_id_table = token_to_id
    sep_id = lambdabeam_dsl.SEP_ID

  else:
    raise ValueError('Unhandled dataset_type: {}'.format(FLAGS.dataset_type))

  print_and_log(f'Sep Id: {sep_id}, Eos ID: {eos_id}')

  # Parse io and program token sequences (for eval).
  def decode_io(inputs, outputs):
    """Converts from int tensors to strings."""
    if FLAGS.dataset_type == 'robustfill':
      def decode_str(s):
        """Decode string tokens."""
        return ''.join([spec_id_token_table[t_id] for t_id in s if t_id > 0])

      inps, outs = [], []
      for inp, out in zip(inputs, outputs):
        inps.append(decode_str(inp))
        outs.append(decode_str(out))
      return inps, outs

    elif FLAGS.dataset_type == 'deepcoder' or FLAGS.dataset_type == 'lambdabeam':
      def decode_str(s):
        return ' '.join(spec_id_token_table[i] for i in s if i > 0)
      inps = [decode_str(inp) for inp in inputs]
      outs = [decode_str(out) for out in outputs]
      return inps, outs

    else:
      raise ValueError('Unhandled dataset_type: {}'.format(FLAGS.dataset_type))

  def decode_spec(target):
    """Converts from int tensor to a string."""
    target = target[np.all([target != 0, target != bos_id, target != eos_id],
                           axis=0)].astype(np.int32)
    target = np.array(target)

    if FLAGS.dataset_type == 'robustfill':
      target = target[target != bos_id].tolist()
      return ''.join([spec_id_token_table[t_id] for t_id in target if t_id > 0])
    elif FLAGS.dataset_type == 'deepcoder' or FLAGS.dataset_type == 'lambdabeam':
      target = target[target != bos_id].tolist()
      return ' '.join([spec_id_token_table[t_id]
                       for t_id in target if t_id > 0])
    else:
      raise ValueError('Unhandled dataset_type: {}'.format(FLAGS.dataset_type))

  def decode_program(program):
    """Decode program tokens into a program object."""
    program = program[:np.argmax(program == eos_id) + 1].astype(np.int32)

    if FLAGS.dataset_type == 'robustfill':
      # Returns either a Concat program object, or None.
      program = program[program != bos_id].tolist()
      try:
        return robust_fill_dsl.decode_program(program, program_id_token_table)
      except:  # pylint: disable=bare-except
        return None  # Program does not compile.

    if FLAGS.dataset_type == 'deepcoder' or FLAGS.dataset_type == 'lambdabeam':
      tokens = [program_id_token_table[t_id] for t_id in program.tolist()
                if t_id > 0 and t_id != eos_id and t_id != bos_id]
                
      try:
        if FLAGS.model_type == 'baseline_model':
          # Parse the entire program.
          if FLAGS.dataset_type == 'lambdabeam':
            return lambdabeam_dsl.Program.from_tokens(tokens)
          else:
            return deepcoder_dsl.Program.from_tokens(tokens)
        else:
          # For DeepCoder, the model only predicts the RHS of the next
          # statement. Note that `output` is not a valid variable name token.
          # That should not matter if we only run this statement on a program
          # state, without constructing a full Program using this statement.
          statement_str = 'output = ' + ' '.join(tokens)
          if FLAGS.dataset_type == 'lambdabeam':
            return lambdabeam_dsl.Statement.from_str(statement_str,
                                                    check_variable_name=False)
          else:
            return deepcoder_dsl.Statement.from_str(statement_str,
                                                     check_variable_name=False)
      except (operation_base.ParseError, deepcoder_dsl.RunError, deepcoder_dsl.ParseError):
        return None  # Program does not compile.

    else:
      raise ValueError('Unhandled dataset_type: {}'.format(FLAGS.dataset_type))

  def decode_program_str(program):  # pylint: disable=unused-variable
    """Decode program tokens into a string."""
    if FLAGS.dataset_type == 'robustfill':
      try:
        return decode_program(program).to_string()  # pytype: disable=attribute-error
      except:  # pylint: disable=bare-except
        return 'did not compile'
    elif FLAGS.dataset_type == 'deepcoder' or FLAGS.dataset_type == 'lambdabeam':
      # This does not check if the program actually compiles.
      return ' '.join([spec_id_token_table[t_id] for t_id in program.tolist()
                        if t_id > 0 and t_id != eos_id and t_id != bos_id])
    else:
      raise ValueError(f'Unhandled dataset_type: {FLAGS.dataset_type}')

  # Load Dataset
  # ---------------------------------------------------------------------------
  print_and_log('Initializing dataset.')
  if not FLAGS.dataset_dir:
    raise ValueError('Must specify dataset_dir.')
  decomposition_or_entire_programs = (
      'entire_programs' if FLAGS.model_type == 'baseline_model'
      else 'decomposition_data')
  train_dataset_path = os.path.join(
      FLAGS.dataset_dir, f'{FLAGS.experiment}_data',
      f'{decomposition_or_entire_programs}_train.tf_records-*')
  test_dataset_path = os.path.join(
      FLAGS.dataset_dir, f'{FLAGS.experiment}_data',
      f'{decomposition_or_entire_programs}_test.tf_records-*')

  # Training dataset.
  print_and_log('Loading dataset from %s' % train_dataset_path)

  padded_shapes = {
      'inputs': io_shape[1:],
      'outputs': io_shape[1:],
      'target': target_shape[1:],
  }
  if FLAGS.model_type == 'spec_decomposer_model' and _DECOMPOSITION_MODE.value != 'standard':
    padded_shapes['synth_target'] = (FLAGS.synthesizer_max_target_length, )

  print_and_log('padded_shapes: %s' % padded_shapes)

  if FLAGS.dataset_type in ['robustfill', 'deepcoder', 'lambdabeam']:
    if FLAGS.dataset_type == 'robustfill':
      input_pipeline_fn = input_pipeline.create_robust_fill_dataset
      program_part_key = 'program_part'
    elif FLAGS.dataset_type == 'lambdabeam':
      input_pipeline_fn = input_pipeline.create_lambdabeam_dataset
      program_part_key = 'program_part_rhs'
    else:
      assert FLAGS.dataset_type == 'deepcoder'
      input_pipeline_fn = input_pipeline.create_deepcoder_dataset
      program_part_key = 'program_part_rhs'

    if FLAGS.model_type == 'spec_decomposer_model' and _DECOMPOSITION_MODE.value == 'standard':
      create_dataset_fn = functools.partial(
          input_pipeline_fn,
          renaming_dict={
              'inputs': 'inputs',
              'outputs': 'outputs',
              'target': 'joined_next_part',
          })
    elif FLAGS.model_type == 'spec_decomposer_model' and _DECOMPOSITION_MODE.value != 'standard':
      create_dataset_fn = functools.partial(
          input_pipeline_fn,
          renaming_dict={
              'inputs': 'inputs',
              'outputs': 'outputs',
              'synth_target': program_part_key,
              'target': 'joined_next_part',
          })
      padded_shapes['synth_target'] = synth_target_shape[1:]
    elif FLAGS.model_type == 'synthesizer_model':
      create_dataset_fn = functools.partial(
          input_pipeline_fn,
          renaming_dict={
              'inputs': 'inputs',
              'outputs': 'next_part',
              'corrupted_outputs': 'corrupted_next_part',
              'target': program_part_key,
          })
      padded_shapes['corrupted_outputs'] = io_shape[1:]
    elif FLAGS.model_type == 'joint_model':
      create_dataset_fn = functools.partial(
          input_pipeline_fn,
          renaming_dict={
              'inputs': 'inputs',
              'outputs': 'outputs',
              'target': program_part_key,
          })
    elif FLAGS.model_type == 'baseline_model':
      create_dataset_fn = functools.partial(
          input_pipeline_fn,
          renaming_dict={
              'inputs': 'inputs',
              'outputs': 'outputs',
              'target': 'program',
          })
    else:
      raise ValueError(f'Unhandled model_type: {FLAGS.model_type}')

  else:
    raise ValueError('Unhandled dataset_type: {}'.format(FLAGS.dataset_type))

  dataset = create_dataset_fn(
      train_dataset_path, spec_token_id_table, FLAGS.num_examples,
      entire_programs=(FLAGS.model_type == 'baseline_model'))

  dataset = dataset.padded_batch(
      batch_size,
      padded_shapes=padded_shapes,
      drop_remainder=True)


  # Split evaluation and training.
  eval_ds = dataset.take(FLAGS.num_eval_steps)

  # Decrease batch of predict dataset to handle beam search.
  predict_padded_shapes = padded_shapes.copy()
  predict_padded_shapes['inputs'] = predict_io_shape[1:]
  predict_padded_shapes['outputs'] = predict_io_shape[1:]

  if FLAGS.model_type == 'synthesizer':
    predict_padded_shapes['corrupted_outputs'] = predict_io_shape[1:]

  print_and_log('predict_padded_shapes: %s' % predict_padded_shapes)
  predict_ds = eval_ds.unbatch().padded_batch(
      1,  # int(np.ceil(batch_size / 10)),
      padded_shapes=predict_padded_shapes)

  train_ds = dataset.skip(FLAGS.num_eval_steps)
  train_ds = train_ds.repeat()

  test_dataset = create_dataset_fn(
      test_dataset_path, spec_token_id_table, FLAGS.num_examples,
      entire_programs=(FLAGS.model_type == 'baseline_model'))
  if FLAGS.model_type == 'baseline_model':
    test_dataset = test_dataset.padded_batch(
        1,
        padded_shapes=predict_padded_shapes,
        drop_remainder=False)
    test_batch_size = 8
    if test_batch_size % n_devices:
      raise ValueError(f'Test batch size {test_batch_size} should be divisible '
                       f'by n_devices {n_devices}')
    quick_test_dataset = (test_dataset
                          # In end-to-end predict, we used 1000 programs
                          # (not batches!).
                          .take(1000)
                          .unbatch()
                          .padded_batch(test_batch_size,
                                        padded_shapes=predict_padded_shapes,
                                        drop_remainder=False))
    final_test_dataset = quick_test_dataset
  else:
    """if FLAGS.model_type == 'spec_decomposer_model' and FLAGS.experiment == 'COMPOSE_DIFFERENT_CONCEPTS' and FLAGS.dataset_type == 'lambdabeam':
      predict_padded_shapes={
          'inputs': [None, None],   # (num_examples, seq_len)
          'outputs': [None, None],
          'target': [None]           # target could be [None, None] if 3D
      }"""
    test_dataset = test_dataset.padded_batch(
        batch_size,
        padded_shapes=predict_padded_shapes,
        drop_remainder=False)
    quick_test_dataset = (test_dataset
                          .take(FLAGS.num_quick_test_steps)
                          .unbatch()
                          .padded_batch(1,  # Make test bs=1 as during inference int(np.ceil(batch_size / 10)),
                                        padded_shapes=predict_padded_shapes))
    final_test_dataset = (test_dataset
                          .take(FLAGS.num_final_test_steps)
                          .unbatch()
                          .padded_batch(1,  # see above: int(np.ceil(batch_size / 10)),
                                        padded_shapes=predict_padded_shapes))

  # Build Model and Optimizer
  # ---------------------------------------------------------------------------
  if FLAGS.model_type == 'spec_decomposer_model':
    output_vocab_size = spec_vocab_size
  elif FLAGS.model_type in ['synthesizer_model', 'joint_model',
                            'baseline_model']:
    output_vocab_size = program_vocab_size
  else:
    raise ValueError(f'Unhandled model_type: {FLAGS.model_type}')

  base_config = base_models.TransformerConfig(
      vocab_size=spec_vocab_size,
      output_vocab_size=output_vocab_size,
      shift=True,
      emb_dim=FLAGS.embedding_dim,
      num_heads=FLAGS.num_heads,
      num_layers=FLAGS.num_layers,
      qkv_dim=FLAGS.embedding_dim,
      mlp_dim=FLAGS.hidden_dim,
      max_len=max(FLAGS.max_input_length, FLAGS.max_target_length),
      dropout_rate=FLAGS.dropout_rate,
      attention_dropout_rate=FLAGS.attention_dropout_rate,
      use_relative_attention=FLAGS.use_relative_attention,
      deterministic=False,
      decode=False,
      bos_token=bos_id,
      num_input_relative_position_buckets=FLAGS.num_position_buckets,
      max_input_distance=FLAGS.max_distance,
      num_output_relative_position_buckets=FLAGS.num_position_buckets,
      max_output_distance=FLAGS.max_distance,
      num_input_cross_output_relative_position_buckets=(
          FLAGS.num_position_buckets),
      max_input_cross_output_distance=FLAGS.max_distance,
      num_program_relative_position_buckets=FLAGS.num_position_buckets,
      max_program_distance=FLAGS.max_distance,
      num_program_cross_embed_relative_position_buckets=(
          FLAGS.num_position_buckets),
      max_program_cross_embed_distance=FLAGS.max_program_cross_embed_distance)

  separator_token_id = (sep_id if FLAGS.model_type == 'spec_decomposer_model'
                        else -1)
  train_config = models.DecomposeAttentionTransformerConfig(
      base_config=base_config,
      dataset_type=FLAGS.dataset_type,
      aligned_relative_attention=FLAGS.aligned_relative_attention,
      separator_token_id=separator_token_id)

  if _DECOMPOSITION_MODE.value != 'standard':
    train_config = train_config.replace(decomposition_mode=_DECOMPOSITION_MODE.value)

  eval_config = train_config.replace(
      base_config=base_config.replace(deterministic=True))
  predict_config = train_config.replace(
      base_config=base_config.replace(
          shift=False, deterministic=True,
          decode=not FLAGS.slow_decode,
          max_len=max(FLAGS.predict_max_input_length, FLAGS.max_target_length)))

  rng = jax.random.PRNGKey(FLAGS.seed)
  rng = jax.random.fold_in(rng, jax.process_index())
  rng, init_rng = jax.random.split(rng)
  dropout_rng = jax.random.split(rng, jax.local_device_count())

  m = models.DecomposeAttentionTransformer(eval_config)
  initial_variables = jax.jit(m.init)(
      init_rng,
      jnp.ones(io_shape, jnp.float32),
      jnp.ones(io_shape, jnp.float32),
      jnp.ones(target_shape, jnp.float32))

  num_params = sum(x.size for x in jax.tree_leaves(initial_variables['params']))
  print_and_log('Model has %d parameters (embedding_dim=%d, hidden_dim=%d, '
                'num_layers=%d, num_heads=%d).'
                % (num_params, FLAGS.embedding_dim, FLAGS.hidden_dim,
                   FLAGS.num_layers, FLAGS.num_heads))

  # Define the optimizer using optax
  tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(
        learning_rate=FLAGS.lr,
        weight_decay=FLAGS.weight_decay,
        b1=0.9,
        b2=0.98,
        eps=1e-9
    )
  )

  # Create a train state for holding parameters and optimizer state
  state = train_state.TrainState.create(
      apply_fn=m.apply,
      params=initial_variables['params'],
      tx=tx
  )

  synthesizer_state, synth_params, synth_train_config, synth_predict_config = None, None, None, None
  if _DECOMPOSITION_MODE.value != 'standard' and FLAGS.model_type == 'spec_decomposer_model':
    ed, hd = 512, 1024
    format_dict = {
      'seed': FLAGS.seed,
      'experiment': FLAGS.experiment,
      'embedding_dim': ed, 
      'hidden_dim': hd
    }
    print('synth max len', FLAGS.synthesizer_max_target_length)
    print('synth max distance', FLAGS.synthesizer_max_distance)
    print('synth max cross-embed distance', FLAGS.synthesizer_max_program_cross_embed_distance)
    synth_base_config = base_models.TransformerConfig(
      vocab_size=spec_vocab_size,
      output_vocab_size=program_vocab_size,
      shift=True,
      emb_dim=ed,
      num_heads=FLAGS.num_heads,
      num_layers=FLAGS.num_layers,
      qkv_dim=ed,
      mlp_dim=hd,
      max_len=FLAGS.synthesizer_max_target_length, # max(FLAGS.synthesizer_max_target_length, FLAGS.synthesizer_max_input_length),
      dropout_rate=0.1,
      attention_dropout_rate=0.1,
      use_relative_attention=True,
      deterministic=True,
      decode=not FLAGS.slow_decode,
      bos_token=bos_id,
      num_input_relative_position_buckets=32,
      max_input_distance=FLAGS.synthesizer_max_distance,
      num_output_relative_position_buckets=32,
      max_output_distance=FLAGS.synthesizer_max_distance,
      num_input_cross_output_relative_position_buckets=(
        32),
      max_input_cross_output_distance=FLAGS.synthesizer_max_distance,
      num_program_relative_position_buckets=32,
      max_program_distance=FLAGS.synthesizer_max_distance,
      num_program_cross_embed_relative_position_buckets=(
        32),
      max_program_cross_embed_distance=FLAGS.synthesizer_max_program_cross_embed_distance)
    

    separator_token_id = -1
    synth_config = models.DecomposeAttentionTransformerConfig(
      base_config=synth_base_config,
      dataset_type=FLAGS.dataset_type,
      aligned_relative_attention=False,
      separator_token_id=separator_token_id)

    synthesizer_state, synth_params = load_frozen_synthesizer(_SYNTHESIZER_PATH_FORMAT.value.format(**format_dict), synth_config, io_shape, synth_target_shape)

    synth_train_config = synth_config.replace(
        base_config=synth_base_config.replace(
          deterministic=True,
          ))


    synth_predict_config = synth_config.replace(
        base_config=synth_base_config.replace(
            shift=False, deterministic=True,
            decode=not FLAGS.slow_decode,
            max_len=max(FLAGS.synthesizer_predict_max_input_length, FLAGS.synthesizer_max_target_length)))

  del initial_variables  # Don't keep a copy of the initial model.

  start_step = 0
  if FLAGS.restore_checkpoints:
    # Restore unreplicated optimizer + model state from last checkpoint.

    state = checkpoints.restore_checkpoint(
        os.path.join(FLAGS.save_dir, 'checkpoints', hparam_str), state)
    # Grab last step.
    start_step = int(state.step)
    print_and_log('Found model checkpointed at step %d.' % start_step)

    if not FLAGS.predict_only:
      print_and_log('Skipping %s steps...' % start_step)
      train_ds = train_ds.skip(start_step)

      dummy_p_train_step = jax.pmap(
          lambda dropout_rng: jax.random.split(dropout_rng)[1])
      for _ in range(start_step):
        dropout_rng = dummy_p_train_step(dropout_rng)
      print_and_log('Finished skipping steps')
      print_and_log('Host %s has dropout_rng = %s'
                    % (jax.process_index(), dropout_rng))

  # Replicate optimizer.
  state = jax_utils.replicate(state)

  assert FLAGS.slow_decode, 'Fast decoding is not implemented yet.'

  learning_rate_fn = create_learning_rate_scheduler(
      base_learning_rate=FLAGS.lr)
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          learning_rate_fn=learning_rate_fn,
          eos_token=eos_id,
          bos_token=bos_id,
          sep_token=sep_id,
          config=train_config,
          synth_config=synth_train_config),
      axis_name='batch',
      devices=jax.local_devices())

  p_eval_step = jax.pmap(
      functools.partial(eval_step,
                        eos_token=eos_id,
                        sep_token=sep_id,
                        config=eval_config),
      axis_name='batch',
      devices=jax.local_devices())
  p_init_cache = jax.pmap(
      functools.partial(
          initialize_cache,
          max_decode_len=FLAGS.max_target_length,
          config=predict_config),
      axis_name='batch',
      devices=jax.local_devices())
  p_pred_step = jax.pmap(
      functools.partial(
          predict_step,
          eos_token=eos_id,
          max_decode_len=FLAGS.max_target_length,
          config=predict_config,
          slow_decode=FLAGS.slow_decode),
      axis_name='batch',
      static_broadcasted_argnums=(4,),
      devices=jax.local_devices())

  if synth_predict_config:
    p_synth_init_cache = jax.pmap(
      functools.partial(
          initialize_cache,
          max_decode_len=FLAGS.synthesizer_max_target_length,
          config=synth_predict_config),
      axis_name='batch',
      devices=jax.local_devices())
    p_synth_pred_step = jax.pmap(
        functools.partial(
            predict_step,
            eos_token=eos_id,
            max_decode_len=FLAGS.synthesizer_max_target_length,
            config=synth_predict_config,
            slow_decode=FLAGS.slow_decode),
        axis_name='batch',
        static_broadcasted_argnums=(4,),
        devices=jax.local_devices())

  # Main Train Loop
  # ---------------------------------------------------------------------------
  print_and_log('Starting training!')
  metrics_all, aux_metrics_all = [], []
  tick = time.time()
  train_iter = train_ds.as_numpy_iterator()
  if FLAGS.predict_only and start_step == FLAGS.num_train_steps:
    start_step -= 1


  for step in range(start_step, FLAGS.num_train_steps):
    is_last_step = step == FLAGS.num_train_steps - 1

    if not FLAGS.predict_only:

      try:
        inputs, outputs, targets, synth_targets, rng = load_data(next(train_iter), rng)
      except tf.errors.DataLossError as e:
        continue

      if FLAGS.compute_false_positives:
        # Extract held-out test sample for evaluation of program correctness
        # We do not evaluate whether a program successfully predicts a held-out test sample when training but for prediction and testing
        inputs, outputs  = inputs[:, :, :-1, :], outputs[:, :, :-1, :]
        if FLAGS.model_type == 'spec_decomposer_model':
          targets = discard_last_subgoal(targets, sep_token=sep_id)
    
      state, metrics, aux_metrics, dropout_rng = p_train_step(state, inputs, outputs, targets, dropout_rng=dropout_rng, synth_params=synth_params, synth_targets=synth_targets)
      metrics_all.append(jax.device_get(metrics))
      aux_metrics_all.append(jax.device_get(aux_metrics))

      # Periodic metric handling.

      # Training Metrics
      if (step and step % FLAGS.log_freq == 0) or is_last_step:
        print_and_log('Gathering training metrics.')
        metrics_all = common_utils.get_metrics(metrics_all)
        lr = metrics_all.pop('learning_rate').mean()
        metrics_sums = jax.tree_util.tree_map(jnp.sum, metrics_all)
        denominator = metrics_sums.pop('denominator')

        summary = jax.tree_util.tree_map(
            lambda x: x / denominator,  # pylint: disable=cell-var-from-loop
            metrics_sums)
        summary['learning_rate'] = lr
        # Calculate (clipped) perplexity after averaging log-perplexities:
        summary['perplexity'] = jnp.clip(jnp.exp(summary['loss']), a_max=1.0e4)

        if _DECOMPOSITION_MODE.value != 'standard' and FLAGS.model_type == 'spec_decomposer_model':
          
          if aux_metrics_all:
              aux_metrics_all_collated = common_utils.get_metrics(aux_metrics_all)
              aux_sums = jax.tree_util.tree_map(jnp.sum, aux_metrics_all_collated)
              aux_rl_mean_loss = jax.tree_util.tree_map(jnp.mean, aux_metrics_all_collated).pop('rl_mean_loss')
              aux_mean_adv = jax.tree_util.tree_map(jnp.mean, aux_metrics_all_collated).pop('mean_advantage')
              aux_std_adv = jax.tree_util.tree_map(jnp.mean, aux_metrics_all_collated).pop('std_advantage')
              aux_denom = aux_sums.pop('sampled denominator')

              # Average the aux metrics (pg_loss, entropy, synth loss)
              aux_summary = jax.tree_util.tree_map(lambda x: x / aux_denom, aux_sums)
              
              # Add to the main summary with a prefix
              summary.update({k: v for k,v in aux_summary.items()})
              summary['rl_mean_loss'] = aux_rl_mean_loss
              summary['mean_advantage'] = aux_mean_adv
              summary['std_advantage'] = aux_std_adv

        if jax.process_index() == 0:
          print_and_log('Step: %d | Loss: %.4f | Greedy Loss: %.4f | Sampled Loss: %.4f' 
                        % (step, summary['loss'], 
                           summary.get('greedy loss', 0.0),
                           summary.get('sampled loss', 0.0)
                           ))
          tock = time.time()
          steps_per_sec = FLAGS.log_freq / (tock - tick)
          tick = tock
          summary_writer.scalar('train/steps per second', steps_per_sec, step)
          for key, val in summary.items():
            summary_writer.scalar('train/' + key, val, step)
          summary_writer.flush()
        # Reset metric accumulation for next evaluation cycle.
        metrics_all, aux_metrics_all = [], []

      # Evaluation Metrics
      if (step and step % FLAGS.eval_freq == 0) or is_last_step:
        print_and_log('Gathering evaluation metrics.')
        t_evaluation_start = time.time()
        eval_metrics, aux_eval_metrics = [], []
        for batches in eval_ds.as_numpy_iterator():
          inputs, outputs, targets, synth_targets, rng = load_data(batches, rng)

          if FLAGS.compute_false_positives:
            # Extract held-out test sample for evaluation of program correctness
            # again we're not using the held-out test samples here
            inputs, outputs = inputs[:, :, :-1, :], outputs[:, :, :-1, :]
            if FLAGS.model_type == 'spec_decomposer_model':
              targets = discard_last_subgoal(targets, sep_token=sep_id)

          metrics, aux_metrics = p_eval_step(state, inputs, outputs, targets, synth_state=synthesizer_state, synth_params=synth_params, synth_targets=synth_targets)
          eval_metrics.append(jax.device_get(metrics))
          aux_eval_metrics.append(jax.device_get(aux_metrics))

        eval_metrics = common_utils.get_metrics(eval_metrics)
        eval_metrics_sums = jax.tree_util.tree_map(jnp.sum, eval_metrics)
        eval_denominator = eval_metrics_sums.pop('denominator')
        eval_summary = jax.tree_util.tree_map(
            lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
            eval_metrics_sums)
        eval_summary['aux loss'] = np.nan
        if FLAGS.model_type == 'spec_decomposer_model' and _DECOMPOSITION_MODE.value != 'standard':
          aux_eval_metrics = common_utils.get_metrics(aux_eval_metrics)
          aux_eval_metrics_sums = jax.tree_util.tree_map(jnp.sum, aux_eval_metrics)
          aux_eval_denominator = aux_eval_metrics_sums.pop('denominator')
          aux_eval_summary = jax.tree_util.tree_map(
              lambda x: x / aux_eval_denominator,  # pylint: disable=cell-var-from-loop
              aux_eval_metrics_sums)
          for k, v in aux_eval_summary.items():
            eval_summary['aux ' + k] = v

        if jax.process_index() == 0:
          print_and_log('Evaluation time: %.4f s step %d, loss: %.4f, aux loss: %.4f.'
                        % (time.time()-t_evaluation_start, step,
                           eval_summary['loss'], eval_summary['aux loss']))
          for key, val in eval_summary.items():
            summary_writer.scalar('eval/' + key, val, step)
          summary_writer.flush()

    # Beam search metrics.
    if (step and step % FLAGS.predict_freq == 0) or is_last_step:
      print_and_log('Gathering beam search metrics.')
      test_ds = final_test_dataset if is_last_step or FLAGS.predict_only else quick_test_dataset

      for dataset, predict_or_test in [(predict_ds, 'predict'), (test_ds, 'test')]:

        for beam_size in [1, 10]:
          t_inference_start = time.time()
          total_successes, total_synth_successes = 0, 0
          total_denominator = 0

          ios, targets_list, predictions, top_of_beams, scores = (
              [], [], [], [], [])

          
          for batches in dataset.as_numpy_iterator():
            pred_batch = batches

            # Handle final odd-sized batch by padding instead of dropping it.
            cur_pred_batch_size = pred_batch['inputs'].shape[0]
            if cur_pred_batch_size % n_devices:
              padded_size = int(
                  np.ceil(cur_pred_batch_size / n_devices) * n_devices)
              # pylint: disable=cell-var-from-loop
              pred_batch = jax.tree_util.tree_map(
                  lambda x: pad_examples(x, padded_size), pred_batch)
            inputs, outputs, targets, synth_targets, rng = load_data(pred_batch, rng)
            
            if FLAGS.compute_false_positives:
              inputs = inputs[:, :, :-1, :]
              outputs = outputs[:, :, :-1, :]

              if FLAGS.model_type == 'spec_decomposer_model':
                targets = discard_last_subgoal(targets, sep_token=sep_id)

            cache = (p_init_cache(inputs, outputs, targets)
              if not FLAGS.slow_decode else None)

            predicted = p_pred_step(state, inputs, outputs, cache, beam_size)
            predicted = tohost(predicted)

            if _DECOMPOSITION_MODE.value != "standard" and FLAGS.model_type == 'spec_decomposer_model':
              flat_predicted = predicted
              if beam_size == 10:
                flat_predicted = np.expand_dims(flat_predicted[:, -1, :], axis=1) # We currently only use the top beam to score the synthesizer during training
              subgoals = np.array([split_subgoals(p, eos_token=eos_id, sep_token=sep_id, pad_size=FLAGS.synthesizer_max_input_length) for p in flat_predicted])
              
              synth_cache = (p_synth_init_cache(inputs, subgoals, synth_targets)
                     if not FLAGS.slow_decode else None)
              predicted_subprograms = p_synth_pred_step(
                  synthesizer_state, inputs, subgoals, synth_cache, beam_size
              )
              predicted_subprograms = tohost(predicted_subprograms)
            inputs, outputs, targets = map(tohost, (inputs, outputs, targets))
            if synth_targets is not None:
              synth_targets = tohost(synth_targets)
            else:
              synth_score = 0

            for i, beams in enumerate(predicted):
              inps, outs = decode_io(inputs[i], outputs[i])

              if FLAGS.model_type == 'spec_decomposer_model':
                ground_truth = decode_spec(targets[i])
                best_prediction, score = eval_predicted_spec_decomposer_model(
                    beams, ground_truth, decode_spec)
                decode_to_str_fn = decode_spec

                if _DECOMPOSITION_MODE.value != "standard":
                  ground_truth_program = decode_program_str(synth_targets[i])
                  best_synth_prediction, synth_score = eval_predicted_synthesizer_model(
                        predicted_subprograms[i], inps, ground_truth.split('|') if FLAGS.dataset_type == 'robustfill' else ground_truth.split(' | '), decode_program, gt=decode_program(synth_targets[i]))
                  beams_synth_target = [decode_program_str(beam) for beam in predicted_subprograms[i]]
              elif FLAGS.model_type == 'synthesizer_model':
                ground_truth = decode_program_str(targets[i])
                best_prediction, score = eval_predicted_synthesizer_model(
                      beams, inps, outs, decode_program)
                decode_to_str_fn = decode_program_str
              elif FLAGS.model_type in ['joint_model', 'baseline_model']:
                ground_truth = decode_program_str(targets[i])
                ground_truth_program = decode_program(targets[i])
                ground_truth_outs = run_program(ground_truth_program, inps)
                best_prediction, score = eval_predicted_synthesizer_model(
                    beams, inps, ground_truth_outs, decode_program)
                decode_to_str_fn = decode_program_str
              else:
                raise ValueError(f'Unknown model type {FLAGS.model_type}')

              if score == 1:
                total_successes += 1
              if synth_score == 1:
                total_synth_successes += 1
              total_denominator += 1

              beams_target = [decode_to_str_fn(beam) for beam in beams]

              ios.append(' ; '.join(map(str, zip(inps, outs))))
              targets_list.append(ground_truth)
              predictions.append(best_prediction)
              scores.append(score)
              logging.info('')
              logging.info('ios: %s', ios[-1])
              logging.info('targets[%s]: %s', i, targets[i])
              logging.info('ground_truth: %s', ground_truth)
              logging.info('predicted beam: %s', '\t'.join(beams_target))
              logging.info('best_prediction: %s', best_prediction)
              if _DECOMPOSITION_MODE.value != "standard" and FLAGS.model_type == 'spec_decomposer_model':
                logging.info('ground_truth_program: %s', ground_truth_program)
                logging.info('best_predicted_program: %s', best_synth_prediction)
                logging.info('predicted synth beam: %s', '\t'.join(beams_synth_target))
                logging.info('synth_score: %s', synth_score)
              logging.info('score: %s', score)
              logging.info('beams: %s', beams)


              if not ground_truth:
                logging.warn('ground_truth is empty!')

              top_of_beam = []
              for index, beam in enumerate(beams[:-5:-1]):
                top_of_beam.append('index: {}, decoded: {}, tokens: {}'.format(
                    index, decode_to_str_fn(beam), beam))
              top_of_beams.append('\n\n'.join(top_of_beam))

          all_total_successes, all_total_denominator, all_total_synth_successes = per_host_sum_pmap(
              jax.tree_util.tree_map(np.array,
                                     (total_successes, total_denominator, total_synth_successes)))

          print_and_log('host %d %s: total_success=%d, total_denominator=%d, total_synth_success=%d. '
                        'all_total_successes=%d, all_total_denominator=%d, all_total_synth_success=%d.'
                        % (jax.process_index(), predict_or_test,
                           total_successes, total_denominator, total_synth_successes,
                           all_total_successes, all_total_denominator, all_total_synth_successes))

          # Record beam search results as text summaries.
          message = []
          for n in np.random.choice(np.arange(len(predictions)), 8):
            text = (f'ios: {ios[n]}\n\ntarget: {targets_list[n]}\n\n'
                    f'predicted: {predictions[n]}\n\n'
                    f'score: {scores[n]}\n\n'
                    f'top of beam:\n\n{top_of_beams[n]}\n\n')
            message.append(text)

          # Write to tensorboard.
          if jax.process_index() == 0:
            accuracy = 100 * all_total_successes / all_total_denominator
            synth_accuracy = 100 * all_total_synth_successes / all_total_denominator
            print_and_log(
                '%s results, step %d, beam size %d: %s / %s = %.2f%%, %s / %s = %.2f%% (%.2f s)'
                % (predict_or_test, step, beam_size,
                   all_total_successes, all_total_denominator, accuracy,
                   all_total_synth_successes, all_total_denominator, synth_accuracy,
                   time.time() - t_inference_start))
            predict_only_label = '_predict-only' if FLAGS.predict_only else ''
            summary_writer.scalar(
                '{}/beam-size-{}{}'.format(predict_or_test,
                                           beam_size,
                                           predict_only_label),
                accuracy, step)
            summary_writer.scalar(
                'synthesizer/{}/beam-size-{}{}'.format(predict_or_test,
                                           beam_size,
                                           predict_only_label),
                synth_accuracy, step)

            summary_writer.text(
                '{}-samples-beam-{}{}'.format(predict_or_test, beam_size,
                                              predict_only_label),
                '\n------\n'.join(message), step)
            summary_writer.flush()
      if FLAGS.predict_only:
        # If only prediction, then don't do it multiple times.
        break

    # Save a Checkpoint. Do this at the end of the training loop, so that if a
    # worker is descheduled during a round of prediction (which takes a while),
    # we will redo prediction upon restarting (to avoid losing data).
    if not FLAGS.predict_only and (
        (step % FLAGS.checkpoint_freq == 0 and step > 0) or is_last_step):
      # Save unreplicated optimizer + model state.
      checkpoints.save_checkpoint(
          os.path.join(FLAGS.save_dir, 'checkpoints', hparam_str),
          jax_utils.unreplicate(state),
          step,
          keep_every_n_steps=100_000)

if __name__ == '__main__':
  app.run(main)
