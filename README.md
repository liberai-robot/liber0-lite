# Liber0-Lite

This repository provides the supervised fine-tuning (SFT) code for Liber-0 on RoboDojo.

[Model weights](https://huggingface.co/LiberAI/Liber0-Lite-Robodojo) · [Installation](#installation) · [Fine-tuning](#fine-tuning) · [Inference](#inference) · [Evaluation](#robodojo-evaluation)

## Model checkpoints

| Checkpoint | Use | Availability |
| --- | --- | --- |
| **Liber0-Lite-Robodojo** | RoboDojo inference and continued fine-tuning | [Hugging Face](https://huggingface.co/LiberAI/Liber0-Lite-Robodojo) |
| **Pretraining initializer** | Initialize downstream SFT before task-specific adaptation | [Hugging Face](https://huggingface.co/LiberAI/Liber0-Lite-Robodojo/tree/main/pretrain) |

Use the pretraining initializer for a fresh downstream SFT run and the RoboDojo checkpoint for inference or continued fine-tuning.

## Installation

Use Linux, Python 3.10, and an NVIDIA GPU with a CUDA 13-compatible driver. From the repository root, in your Python environment:

```bash
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e .
```

Video loading uses TorchCodec. Install a compatible FFmpeg build with support for the codecs used by your dataset. Keep the policy and RoboDojo simulator in separate environments.

### Download weights

For training from the pretraining initializer:

```bash
python download_checkpoint.py --pretrained \
  --destination ./checkpoints/base \
  --assets-dir ./assets
```

This downloads `pretrain/model.pt` and the RoboDojo target configuration/statistics, without fetching the separate RoboDojo SFT weights. The action output head is omitted; robot-specific input/output projections are initialized by the training loader.

For inference or continued fine-tuning of the RoboDojo checkpoint:

```bash
python download_checkpoint.py \
  --destination ./checkpoints/robodojo \
  --assets-dir ./assets
```

Both modes also download the required [HiDream-O1-Image base assets](https://huggingface.co/HiDream-ai/HiDream-O1-Image). Releases are pinned to fixed revisions and model weights are verified against their SHA256 manifests. The script clears proxy environment variables before downloading.

## Fine-tuning

### 1. Prepare the data

The loader accepts **LeRobot v2.1 video datasets** produced by the [XPolicyLab RoboDojo converter](https://github.com/XPolicyLab/XPolicyLab/blob/main/scripts/transform_lerobot_v21_format.py). A dataset contains:

```text
dataset/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   └── tasks.jsonl
├── data/                   # Per-episode Parquet files
└── videos/                 # Per-camera RGB videos
```

File paths are read from `meta/info.json`. The required observation and action fields are:

| Field | Content |
| --- | --- |
| `observation.images.cam_high` | Head-camera RGB |
| `observation.images.cam_left_wrist` | Left-wrist RGB |
| `observation.images.cam_right_wrist` | Right-wrist RGB |
| `observation.state` | 14-D joint/gripper state |
| `action` | 14-D absolute joint/gripper targets |

State and action order is **left arm (6), left gripper (1), right arm (6), right gripper (1)**. Parquet files also provide `frame_index`, `episode_index`, and `task_index`.

Set your dataset path and run the preflight check:

```bash
export DATASET_ROOT=/path/to/robodojo_lerobot_v21

bash process_data.sh \
  --dataset "$DATASET_ROOT" \
  --checkpoint ./checkpoints/base
```

This checks the file inventory and decodes selected windows; it does not convert raw data. Fine-tuning reuses the checkpoint's normalization statistics. Samples contain current and future observations, a 32-step action chunk, and the episode's first head-camera frame as the visual cue. Windows stay within an episode, and padded tail actions are excluded from the loss.

### 2. Configure the run

The example recipe is in [`configs/robodojo_sft.yaml`](configs/robodojo_sft.yaml):

| Setting | Value |
| --- | --- |
| Training updates | 72,000 |
| Global batch size | 1,024 |
| Optimizer | AdamW |
| Peak learning rate | `1e-4` |
| Schedule | 3,600 warmup updates, then cosine decay to 1% of peak LR |
| Checkpoint interval | 5,000 updates and the final update |
| Input / target image size | 288 × 384 |

All episodes are used for training; no validation split is created. Decoder compilation is enabled by default. Set `gradient_checkpointing: true` in the config if needed, and choose a per-GPU batch size appropriate for your hardware.

### 3. Launch training

Set `NPROC_PER_NODE` to the number of GPUs per node, `MICRO_BATCH` to the per-GPU batch size, `ACCUMULATION` to the accumulation count, and `NUM_WORKERS` to the data-loader worker count. The launcher requires:

```text
nodes × GPUs per node × microbatch × accumulation = global batch size
```

With those variables set, launch from the repository root:

```bash
export OUTPUT_DIR=./outputs/robodojo-sft

bash train.sh --config configs/robodojo_sft.yaml \
  --dataset "$DATASET_ROOT" \
  --checkpoint ./checkpoints/base \
  --init-weights ./checkpoints/base/pretrain/model.pt \
  --assets ./assets \
  --output "$OUTPUT_DIR" \
  --batch-size "$MICRO_BATCH" \
  --accumulation "$ACCUMULATION" \
  --num-workers "$NUM_WORKERS"
```

For continued fine-tuning instead, set `--checkpoint ./checkpoints/robodojo` and omit `--init-weights`. Both start a fresh optimizer and schedule; use `--resume` to continue an existing training run.

For multiple nodes, also set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`, then run the same command on every node. Input assets and the output directory must be accessible to all ranks. `train.sh` launches workers; it does not request cluster resources.

For a short startup check, append `--stop-after N`. This stops at optimizer update `N` and saves a checkpoint without changing the learning-rate schedule. Use a new output directory for each fresh run.

### Resume training

Run the same command with the same output directory and append `--resume "$OUTPUT_DIR/step_005000"`.

Resume restores model and optimizer weights, the scheduler, per-rank random states, and the data cursor. Keep the world size, batch settings, dataset, and training configuration unchanged. Changing data-loader worker count is supported. Only load resume state from trusted training runs.

## Inference

The downloaded checkpoint and exported training checkpoints use the same `Policy` interface:

```python
import os
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'

from pathlib import Path
from liber0.runtime import Policy

policy = Policy(
    checkpoint_root=Path('checkpoints/robodojo'),
    assets=Path('assets'),
    device='cuda:0',
    weights_file='model.pt',
    num_inference_steps=10,
)
```

Given current RGB arrays (`uint8`, height × width × 3) and a 14-D joint/gripper state, request an action chunk as follows. Capture `first_head_rgb` once at the start of each episode and keep it fixed:

```python
observation = {
    'head_camera_rgb': head_rgb,
    'left_camera_rgb': left_wrist_rgb,
    'right_camera_rgb': right_wrist_rgb,
    'visual_cue_rgb': first_head_rgb,
    'joint_action_vector': joint_state,
}

actions = policy.infer_action_chunk(
    observation,
    instruction='Place the cup on the table.',
    request_seed=0,
)
# actions.shape == (32, 14)
```

For a trained checkpoint, replace `checkpoint_root` with a completed `step_XXXXXX` directory. Its `model.pt`, `config.yaml`, and `dataset_stats.json` are sufficient for inference alongside the base assets. `state.pt` and training metadata are only needed for resuming training, not for distribution or deployment.

## RoboDojo evaluation

Use the `Liber_0_lite` adapter in [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) for simulator evaluation. Follow its installation and evaluation instructions, point it to the downloaded or exported checkpoint directory, and provide the base-assets directory through `LIBER0_MODEL_PATH`. The adapter handles the simulator observation/action interface; simulator assets and dependencies are installed separately.

## Release status

Data preprocessing, training objectives, optimizer recovery, distributed accumulation, and a reduced-size decoder have CPU test coverage. Full-checkpoint CUDA training, CUDA compilation parity, and clean-environment installation are still being validated. GPU memory requirements and end-to-end training results for this standalone release are not yet published.

## Acknowledgments

This implementation builds on [HiDream-O1-Image](https://github.com/HiDream-ai/HiDream-O1-Image) and the Qwen / Hugging Face model implementations. We use [RoboDojo](https://robodojo-benchmark.com/) and [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) for the benchmark interface. Third-party licenses and attribution are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
