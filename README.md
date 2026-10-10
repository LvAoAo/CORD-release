# CORD

**Learning from rewards and skills with conflict-aware updates.**

CORD handles gradient conflicts between two training objectives. This repository provides **Skill-SD + CORD** for multi-turn agents on **ALFWorld** and **WebShop**, using Qwen2.5-3B-Instruct and bundled skill banks.

In this recipe, the reinforcement-learning update is protected: when the two gradients conflict, CORD projects the distillation gradient before combining them. Gradients are accumulated over the full PPO mini-batch before projection. The experiment settings are fixed, so the same commands cover training and independent checkpoint evaluation.

## Installation

The reference setup is **Linux, Python 3.12, CUDA 12.8, and 8 × NVIDIA A100 40 GB**. FlashAttention also requires a CUDA toolkit with `nvcc` and a C++ compiler. Run the commands below from the repository root.

```bash
git clone https://github.com/LvAoAo/CORD-release.git
cd CORD-release

conda create -n cord python=3.12 -y
conda activate cord
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
pip install -e . --no-deps
pip check
```

The default training launcher requires **96 GiB of free disk space** at startup, including a 64 GiB reserve. Model downloads, caches, logs, and checkpoints are stored locally. A separate logging-service account is not required.

## Prepare the environments

Choose either task, or prepare both. The skill banks and evaluation manifests are already included; benchmark assets are downloaded separately.

### ALFWorld

```bash
export ALFWORLD_DATA="$PWD/data/alfworld"
bash scripts/setup_alfworld.sh
```

The setup downloads the game assets and checks that the games in the fixed evaluation manifest are available. To reuse an existing installation, set `ALFWORLD_DATA` to its asset directory.

### WebShop

Install Java 11 and the WebShop dependencies in the same environment:

```bash
conda install -c conda-forge openjdk=11 -y
export JAVA_HOME="$CONDA_PREFIX/lib/jvm"
pip install -r requirements-webshop.txt
pip install --no-deps https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

bash scripts/setup_webshop.sh
export WEBSHOP_DATA="$PWD/data/webshop"
export WEBSHOP_INDEX="$PWD/data/webshop/indexes"
```

This downloads the fixed **1,000-product catalog** and builds its Lucene index. To reuse existing assets, point `WEBSHOP_DATA` at the directory containing `items_shuffle_1000.json`, `items_ins_v2_1000.json`, and `items_human_ins.json`, and `WEBSHOP_INDEX` at the matching index. Keep these environment variables set when training or evaluating.

## Train

Run Skill-SD + CORD on the task you prepared:

```bash
# ALFWorld
bash scripts/train.sh --task alfworld --method skillsd

# WebShop
bash scripts/train.sh --task webshop --method skillsd
```

Both recipes use **150 training steps**, learning rate **1e-6**, and distillation coefficient **0.001** with the importance-weighted k3 loss. CORD applies projection only when the gradients conflict. Training uses seed 0 and the included task-specific skills; validation runs every five steps on a fixed 128-assignment manifest.

Outputs go to `outputs/alfworld-skillsd/` or `outputs/webshop-skillsd/`. The final checkpoint is saved under `checkpoints/global_step_150/`. For another run, choose a new directory with `--output-dir`; existing checkpoints are never overwritten.

Use `--model /path/to/model` to load a local copy of Qwen2.5-3B-Instruct. GPU count, tensor parallelism, and micro-batch sizes can be adjusted through `--gpus`, `--tensor-parallel`, `--micro-batch-size`, and `--logprob-micro-batch-size`; see `bash scripts/train.sh --help`. The reference configuration uses eight GPUs and tensor parallelism of two.

## Evaluate

After training, reload the final checkpoint in a separate evaluation run:

```bash
# ALFWorld
bash scripts/evaluate.sh --task alfworld --method skillsd

# WebShop
bash scripts/evaluate.sh --task webshop --method skillsd
```

If training used a custom `--output-dir` or `--model`, pass the same values here. Use `--checkpoint /path/to/global_step_150` to select a different checkpoint. **Evaluation must use the same GPU count as training**, because model weights are saved as FSDP shards. Checkpoints support evaluation; optimizer state is not saved for exact training continuation.

Final evaluation uses four sampled passes at temperature 0.4:

| Task | Fixed evaluation set |
| --- | --- |
| ALFWorld | 128 assignments covering 91 distinct games |
| WebShop | 500 goals from the bundled test manifest |

Each evaluation writes `per_pass.json` and `summary.json`. The summary reports the mean and population standard deviation across the four passes, rather than across independently trained seeds.

```text
outputs/alfworld-skillsd/
├── checkpoints/global_step_150/actor/
└── runs/
    ├── train-<run-id>/run.log
    └── evaluate-<run-id>/evaluation/
        ├── per_pass.json
        └── summary.json
```

## Code layout

| Path | Contents |
| --- | --- |
| `cord/` | Task settings, data setup, and launch commands |
| `verl/trainer/ppo/cord_utils.py` | Distillation loss and CORD gradient combination |
| `verl/workers/actor/dp_actor.py` | Gradient accumulation and actor updates |
| `agent_system/` | Environment interfaces and multi-turn rollouts |
| `skills/` · `manifests/` | Fixed skill banks and evaluation assignments |

## License

Apache-2.0. See [LICENSE](LICENSE) for the license and third-party attribution notices.
