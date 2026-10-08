# Solver-Aware Decompositions for Programming-by-Example: When Dividing Requires Knowing how to Conquer

Code for the paper **Solver-Aware Decompositions for Programming-by-Example: When Dividing Requires Knowing how to Conquer**, Zenkner et al., NeurIPS 2026.
[Paper](https://arxiv.org/abs/2608.03461)

This repository contains the code needed to reproduce the paper's experiments.
It ships no datasets or model checkpoints. The scripts below generate the data,
and the pretrained ExeDec models are read directly from ExeDec's public Google
Cloud Storage bucket.

## Overview

The code follows the [ExeDec](https://github.com/google-deepmind/exedec) setup.
A **spec decomposer** predicts the next subgoal (the intermediate outputs of the
next program step), and a **synthesizer** writes the program step that realizes
it. ExeDec trains the decomposer only to imitate ground-truth subgoals, so it is
blind to whether the synthesizer can actually realize what it proposes.

SAD makes the decomposer *solver-aware*. It samples subgoals from the
decomposer and scores each one by the frozen synthesizer's cross-entropy of the
ground-truth program step given that subgoal. The decomposer is then trained
with a self-critical REINFORCE loss, using its greedy subgoal as the baseline,
together with the usual cross-entropy and an entropy bonus. The control arm
(`--rl_loss=supervised`) runs the identical pipeline with the cross-entropy
term only.

Experiments cover three domains, each with small Transformers trained from
scratch:
- **DeepCoder** (list manipulation) and **RobustFill** (string manipulation),
  where the frozen synthesizer is ExeDec's pretrained model.
- **LambdaBeam** (list manipulation with lambdas), where the synthesizer is
  trained here first.

The same recipe is also applied to Llama-3.1-8B LoRA adapters on DeepCoder.

## Repository layout

| Path | Contents |
|---|---|
| `tasks/` | DSLs (`deepcoder/`, `robust_fill/`, `lambdabeam/`), random program sampling and data generation (`*/dataset/`) |
| `models/` | Transformer encoder-decoder with relative attention |
| `spec_decomposition/train.py` | Training: ExeDec models, and SAD for the spec decomposer (`--decomposition_mode=coupled`) |
| `spec_decomposition/end_to_end_predict.py` | End-to-end synthesis with a decomposer and a synthesizer |
| `spec_decomposition/run_*.sh` | Entry points for data, training and evaluation, one per domain |
| `spec_decomposition/llm_finetune/` | LLM pipeline: SFT data, LoRA SFT, SAD for the decomposer LLM, reward diagnostics |
| `spec_decomposition/run_llm_experiment.py` | LLM evaluation harness (Ollama or Hugging Face backend) |
| `data/` | Builders for the Rule et al. benchmark and for the DeepCoder LLM test sets |
| `plotting/` | Notebook that produces the paper's tables and figures |

## Setup

The transformer experiments need Python 3.9 with JAX, Flax and TensorFlow. The
LLM experiments additionally need PyTorch, Transformers and PEFT. Separate
environments work best:

```bash
conda create -n sad python=3.9 && conda activate sad
pip install -r requirements.txt

conda create -n sad-llm python=3.9 && conda activate sad-llm
pip install -r requirements-llm.txt
```

The analysis notebook also needs
`pip install jupyter matplotlib seaborn matplotlib-venn statannotations`.

Run all commands from the repository root. The scripts write generated data
to `$DATA_DIR` (default `./generated_data`) and checkpoints, TensorBoard logs
and evaluation results to `$RESULTS_DIR` (default `./results`).

Each script reads its settings from environment variables and passes any extra
arguments through to the underlying Python launcher. By default it runs the
paper's full sweep, sequentially:
- generalization splits `NONE` and `LENGTH_GENERALIZATION`
- seeds 10, 20, 30, 40, 50
- both arms (`supervised` and `grpo`)

To run one job per configuration instead, for example under a cluster
scheduler, narrow the sweep:

```bash
EXPERIMENTS=NONE SEEDS=10 RL_OPTIONS=grpo bash spec_decomposition/run_deepcoder_training.sh
```

The other four ExeDec generalization splits (`COMPOSE_DIFFERENT_CONCEPTS`,
`SWITCH_CONCEPT_ORDER`, `COMPOSE_NEW_OP`, `ADD_OP_FUNCTIONALITY`) are supported
throughout.

The paper's runs used one 48 GB GPU per training job, and one 94 GB H100 for
LLM SFT and SAD.

## 1. Data

```bash
bash tasks/deepcoder/dataset/run_data_generation.sh
bash tasks/robust_fill/dataset/run_data_generation.sh
bash tasks/lambdabeam/dataset/run_data_generation.sh
```

These generate the full train/valid/test splits used in the paper. They are
large; set `GENERATE_FULL_DATA=false` for a small dataset to test the pipeline.
ExeDec also provides its DeepCoder and RobustFill test sets in the
[`gs://exedec`](https://console.developers.google.com/storage/browser/exedec)
bucket.

## 2. Training

**DeepCoder and RobustFill.** These train the decomposer against ExeDec's
pretrained synthesizer (`gs://exedec/trained_models/...`), so the machine needs
network access to GCS:

```bash
bash spec_decomposition/run_deepcoder_training.sh
bash spec_decomposition/run_robustfill_training.sh
```

If reading from GCS fails with `libcurl code 77 meaning 'Problem with the SSL CA
cert'`, TensorFlow is looking for the CA bundle at the Debian path. On
RHEL/CentOS-based systems, point it to the local bundle:
`export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt`.

**LambdaBeam.** There are no pretrained models, so first train the synthesizers
and then SAD:

```bash
# Stage 1: the joint_model is the frozen synthesizer SAD trains against; the
# synthesizer_model is only needed to evaluate with NSA=false.
MODEL_TYPE=joint_model RL_OPTIONS=supervised LR=2e-4 bash spec_decomposition/run_lambdabeam_training.sh
MODEL_TYPE=synthesizer_model RL_OPTIONS=supervised LR=1e-4 bash spec_decomposition/run_lambdabeam_training.sh
# Stage 2: SAD and the supervised control.
bash spec_decomposition/run_lambdabeam_training.sh
```

**Ablations.** Pass the extra flags through, and give each run its own title so
that it does not overwrite the main runs:

```bash
# Entropy weight (default 0.001).
RL_OPTIONS=grpo bash spec_decomposition/run_deepcoder_training.sh --entropy_coef=0.1 --exp_title=grpo_lambda01
# Without the supervised cross-entropy term (the L_sup ablation).
RL_OPTIONS=grpo bash spec_decomposition/run_deepcoder_training.sh --sup_loss_weight=0 --exp_title=grpo_nosup
```

## 3. Evaluation

```bash
bash spec_decomposition/run_deepcoder_end_to_end_predict.sh
bash spec_decomposition/run_robustfill_end_to_end_predict.sh
bash spec_decomposition/run_lambdabeam_end_to_end_predict.sh
```

The main settings (see the header of `run_deepcoder_end_to_end_predict.sh`):

- `TRAIN_OPTIONS`: which decomposers to evaluate (`supervised grpo` by
  default).
  - `exedec` (DeepCoder and RobustFill) evaluates the ExeDec baseline with
    ExeDec's own pretrained decomposer and synthesizer.
  - On LambdaBeam, the ExeDec-style baseline is
    `TRAIN_OPTIONS=supervised NSA=false`.
- `ORACLE_TYPE`:
  - `none`: the decomposer's own subgoals
  - `kbest`: the ground-truth subgoal whenever it is in the beam
  - `oracle`: always the ground-truth subgoal
- `NSA`:
  - `true` (default): use the joint_model synthesizer that SAD trains against
  - `false`: use the separately trained synthesizer_model

Results are written to
`$RESULTS_DIR/evaluation/<domain>_e2e_predict_1/<arm>_<oracle>_nsa<true|false>/`
as TensorBoard logs plus per-task JSON.

### Rule et al. (BIG-bench `list_functions`)

The out-of-distribution evaluation on LambdaBeam uses the Rule et al. concept
learning tasks (BIG-bench `list_functions`). Place them as
`data/rules_bigbench.csv`, with one row per I/O example and the columns
`id,trial,concept,input1,input2,output,run`. Then build the test set and
evaluate:

```bash
python -m data.synthesize_lambdabeam    # search a LambdaBeam program per concept
python -m data.extract_solved_io        # keep the I/O pairs those programs solve
python -m data.adapt_bigbench_solved    # fit lists/ints to the model's ranges
python -m data.convert_to_json_ruleetal # write RULES_ET_AL.jsonl
python -m data.convert_jsonl_to_tf_records \
  --input_dir=${DATA_DIR:-./generated_data}/rules_et_al/jsonl --tasks=RULES_ET_AL
RULES=true bash spec_decomposition/run_lambdabeam_end_to_end_predict.sh
```

`synthesize_lambdabeam` searches each concept under a wall-clock limit, so the
number of solved concepts, and therefore the test set, can vary slightly with
hardware. The test set used in the paper has 129 tasks.

## 4. LLM experiments

These use [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B),
a gated model, so you need accepted access and a Hugging Face login. They also
need the DeepCoder data from step 1. Fine-tuning uses the `NONE` split, but the
evaluation covers all six DeepCoder generalization splits, so first generate the
test split of the other four as well.

```bash
EXPERIMENTS="COMPOSE_DIFFERENT_CONCEPTS SWITCH_CONCEPT_ORDER COMPOSE_NEW_OP ADD_OP_FUNCTIONALITY" \
  SPLITS=test bash tasks/deepcoder/dataset/run_data_generation.sh
# LLM test sets: 200 DeepCoder tasks per split, with few-shot examples.
python -m data.build_llm_deepcoder_data
# SFT records for both roles.
python -m spec_decomposition.llm_finetune.build_sft_data --generalization_task=NONE --split=train --max_tasks=20000
python -m spec_decomposition.llm_finetune.build_sft_data --generalization_task=NONE --split=valid --max_tasks=500

# Optional: run every stage on a tiny slice first.
bash spec_decomposition/llm_finetune/run_smoke_tests.sh

bash spec_decomposition/run_llm_sft.sh      # LoRA SFT of the decomposer and synthesizer
python -m spec_decomposition.llm_finetune.reward_check \
  --synthesizer_adapter=./results/llm_sft/synthesizer_NONE/adapter  # does the reward discriminate?
bash spec_decomposition/run_llm_sad.sh      # SAD (grpo) and the CE-only control
bash spec_decomposition/run_llm_finetuned_eval.sh sad   # also: base, sft, ...
```

`run_llm_sad.sh` and `run_llm_finetuned_eval.sh` document the start-from
checkpoint variants used in the paper (`*_sft1000`, `*_basedec`, `*_base`) in
their headers. To evaluate a single split, pass it through, e.g.
`bash spec_decomposition/run_llm_finetuned_eval.sh sad --task=NONE`.

The few-shot prompting baseline (`spec_decomposition/run_llm.sh`) queries a
local [Ollama](https://ollama.com) server and also evaluates RobustFill. For
that, copy ExeDec's `data/llm_data/robustfill/` into `data/llm_data/robustfill/`.

## 5. Tables and figures

`plotting/analysis.ipynb` reads the results from `../results` (set
`RESULTS_DIR` in its second cell) and produces the paper's tables and plots.

## Tests

The unit tests use `absltest`, for example:

```bash
python -m tasks.deepcoder.deepcoder_dsl_test
python -m spec_decomposition.llm_finetune.prompts_test
```

## Acknowledgements and license

This code builds on [ExeDec](https://github.com/google-deepmind/exedec)
(Shi et al., ICLR 2024). Files that carry the DeepMind copyright header are
derived from ExeDec and were modified for this work. The LambdaBeam DSL follows
[LambdaBeam](https://arxiv.org/abs/2306.02049) (Shi et al., NeurIPS 2023).

## Citation

```bibtex
@article{zenkner2026solver,
  title={Solver-Aware Decompositions for Programming-by-Example: When Dividing Requires Knowing how to Conquer},
  author={Zenkner, Janis and Sesterhenn, Tobias and Grams, Tim and Bartelt, Christian},
  journal={arXiv preprint arXiv:2608.03461},
  year={2026}
}
```
