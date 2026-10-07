r"""Builds the DeepCoder test sets for the LLM experiments.

Reads the generated DeepCoder test TFRecords ($DATA_DIR/deepcoder_data, see
tasks/deepcoder/dataset/run_data_generation.sh) and writes up to
MAX_PER_CATEGORY tasks per generalization split, each with few-shot examples
from the same split, to data/llm_data/deepcoder/<TASK>.jsonl -- the format
spec_decomposition/run_llm_experiment.py reads.

Run from the repo root:
    python -m data.build_llm_deepcoder_data
"""

from pathlib import Path
from collections import defaultdict
from tasks.deepcoder import deepcoder_dsl
from absl import flags
from absl import app
from spec_decomposition.end_to_end_predict import create_deepcoder_dataset
import os
import numpy as np
import json
import re
import random

flags.FLAGS([''])

DATA_DIR = Path(os.environ.get('DATA_DIR', './generated_data')) / 'deepcoder_data'
MAX_PER_CATEGORY = 200
id_to_token, token_to_id = deepcoder_dsl.vocab_tables()
NUM_EXAMPLES = 4

SAVE_DIR = Path("data/llm_data/deepcoder")

def parse_input_string(input_strings):
    """
    Convert a list of strings like:
    ['x0 = [ 4 1 2 ] | x1 = 0', 'x0 = [ 4 2 3 ] | x1 = 1', ...]
    into a dict:
    {'x0': [[4,1,2], [4,2,3], ...], 'x1': [0, 1, ...]}
    """
    inputs_dict = defaultdict(list)
    
    for s in input_strings:
        parts = [p.strip() for p in s.split('|')]
        for part in parts:
            # Try to match a list: x0 = [ 4 1 2 ]
            m_list = re.match(r"x(\d+)\s*=\s*\[\s*([^\]]*?)\s*\]", part)
            if m_list:
                var_idx, values = m_list.groups()
                var_name = f"x{var_idx}"
                if values.strip() == "":
                    parsed_values = []
                else:
                    parsed_values = [int(v) for v in values.strip().split()]
                inputs_dict[var_name].append(parsed_values)
                continue

            # Try to match a single number: x1 = 0
            m_int = re.match(r"x(\d+)\s*=\s*(-?\d+)", part)
            if m_int:
                var_idx, value = m_int.groups()
                var_name = f"x{var_idx}"
                inputs_dict[var_name].append(int(value))
                continue

            raise ValueError(f"Cannot parse input part: {part}")
    
    return dict(inputs_dict)


def parse_output_strings(output_strings):
    """
    Convert list of output strings like:
      ["[ -8 ]", "[ -9 -27 ]", "[ 11 ]", "[ -7 -35 ]"]
    into list of lists or numbers:
      [[-8], [-9, -27], [11], [-7, -35]]
    """
    parsed_outputs = []
    for s in output_strings:
        s = s.strip()
        if s.startswith('[') and s.endswith(']'):
            # Remove brackets and parse numbers
            numbers = [int(v) for v in s[1:-1].strip().split()]
            parsed_outputs.append(numbers)
        else:
            # Single number
            parsed_outputs.append(int(s))
    return parsed_outputs


def save_tasks_jsonl(tasks, few_shot_count=4):
    """
    Save tasks to JSONL files per category, keeping inputs/outputs as nested lists of integers.
    Adds `few_shot_examples` field with a few other tasks from the same category.
    """
    tasks_by_category = defaultdict(list)
    for t in tasks:
        tasks_by_category[t['category']].append(t)

    for category, category_tasks in tasks_by_category.items():
        save_path = SAVE_DIR / f"{category}.jsonl"
        with open(save_path, 'w', encoding='utf-8') as f:
            for idx, task in enumerate(category_tasks):
                # Sample few-shot examples from other tasks in the same category
                # Exclude the current task itself
                candidates = [t for t in category_tasks if t != task]
                few_shots = random.sample(candidates, min(few_shot_count, len(candidates)))

                # Build few-shot examples in same format
                few_shot_list = []
                for fs in few_shots:
                    few_shot_list.append({
                        "inputs": parse_input_string(fs["inputs"]),
                        "outputs": parse_output_strings(fs["outputs"]),
                        "program": fs["solution"]
                    })

                # Build JSON line
                json_line = {
                    "index": idx,
                    "test_problem": {
                        "inputs": parse_input_string(task["inputs"]),
                        "outputs": parse_output_strings(task["outputs"]),
                        "program": task["solution"]
                    },
                    "few_shot_examples": few_shot_list
                }
                f.write(json.dumps(json_line) + "\n")
        print(f"Saved {len(category_tasks)} tasks to {save_path}")


def decode_spec(target, dataset, spec_id_token_table, bos_id, eos_id):
    """Convert from int tensor to a string."""
    if dataset in ['robustfill', 'deepcoder', 'lambdabeam']:
      target = np.array(target)
      target = target[(target != 0) & (target != bos_id) & (target != eos_id)].astype(np.int32)

      separator = ' ' if dataset in ['deepcoder', 'lambdabeam'] else ''
      return separator.join([spec_id_token_table[t_id]
                             for t_id in target if t_id > 0])
    else:
      raise ValueError('Unhandled dataset_type: {}'.format(dataset))


def extract_tasks():
    seen_inputs = set()
    tasks_per_category_length = defaultdict(lambda: defaultdict(int))  # category -> length -> count
    extracted_tasks = []

    # Iterate over each category subdir
    for category_dir in DATA_DIR.iterdir():
        if not category_dir.is_dir():
            continue
        category_name = category_dir.name.split('_data')[0]
        max_prog_len = 4 # if category_name in ['SWITCH_CONCEPT_ORDER', 'COMPOSE_DIFFERENT_CONCEPTS', 'COMPOSE_NEW_OP'] else 3
        print(f'Extracting tasks for {category_name}.')

        file_path = os.path.join(category_dir, 'entire_programs_test.tf_records-*')
        dataset = create_deepcoder_dataset(file_path,
                                           token_to_id,
                                           NUM_EXAMPLES,
                                           'deepcoder')

        for data in dataset.as_numpy_iterator():
            hash_inputs = tuple(map(tuple, data['inputs']))  # make hashable
            outputs = tuple(map(tuple, data['outputs']))  # make hashable
            solution = data['target']

            inputs = [decode_spec(ipt, 'deepcoder', id_to_token, token_to_id['<BOS>'], token_to_id['<EOS>']) for ipt in hash_inputs]
            outputs = [decode_spec(ipt, 'deepcoder', id_to_token, token_to_id['<BOS>'], token_to_id['<EOS>']) for ipt in outputs]

            program = deepcoder_dsl.Program.from_tokens([id_to_token[int(p_id)]
                                                        for p_id in solution
                                                        if p_id > 0 and p_id != deepcoder_dsl.EOS_ID])
            
            # Skip empty test outputs
            if outputs[-1] == '[ ]':
                continue

            # Determine program length (number of non-INPUT lines)
            prog_length = len([ele for ele in str(program).split('|') if 'INPUT' not in ele])

            # Skip too long for most categories (except LENGTH_GENERALIZATION)
            if prog_length > max_prog_len and category_name != 'LENGTH_GENERALIZATION':
                continue

            # Skip duplicates
            if hash_inputs in seen_inputs:
                continue

            # Skip if we already reached MAX_PER_CATEGORY for this length
            if tasks_per_category_length[category_name][prog_length] >= MAX_PER_CATEGORY * 1.1 // 3 and category_name != 'LENGTH_GENERALIZATION':
                continue
            
            if sum(tasks_per_category_length[category_name].values()) >= MAX_PER_CATEGORY:
                print(tasks_per_category_length[category_name])
                break

            # Add task
            extracted_tasks.append({
                'category': category_name,
                'inputs': inputs,
                'outputs': outputs,
                'solution': str(program),
            })
            seen_inputs.add(hash_inputs)
            tasks_per_category_length[category_name][prog_length] += 1

        # Stop early if all lengths for this category reached max
        lengths_done = all(count >= MAX_PER_CATEGORY // 3 for count in tasks_per_category_length[category_name].values())
        if lengths_done:
            continue

    return extracted_tasks


def main(_):
    random.seed(0)  # Few-shot examples are sampled.
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    tasks = extract_tasks()
    save_tasks_jsonl(tasks)

if __name__ == '__main__':
  app.run(main)
