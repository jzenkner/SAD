# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keeps track of free and bound variables."""

import functools
from absl import flags
from tasks import value as value_module


MAX_NUM_FREE_VARS = 3
"""flags.DEFINE_integer(
    'max_num_free_vars', 2,
    'The maximum number of free variables per lambda function.')"""

MAX_NUM_BOUND_VARS = 3
"""flags.DEFINE_integer(
    'max_num_bound_vars', 2,
    'The maximum number of bound variables.')"""

MAX_NUM_ARGVS = max(MAX_NUM_FREE_VARS, MAX_NUM_BOUND_VARS)

ALL_FREE_VARS = [value_module.get_free_variable(i)
                 for i in range(MAX_NUM_FREE_VARS)]
ALL_BOUND_VARS = [value_module.get_bound_variable(i)
                  for i in range(MAX_NUM_BOUND_VARS)]

ARGV_MAP = {}
for i, argv in enumerate(ALL_FREE_VARS + ALL_BOUND_VARS):
  ARGV_MAP[argv] = i

ARGV_SET = set(ARGV_MAP)


@functools.lru_cache(maxsize=None)
def first_free_vars(n):
  return ALL_FREE_VARS[:n]


@functools.lru_cache(maxsize=None)
def first_bound_vars(n):
  return ALL_BOUND_VARS[:n]