# soft-actor-critic
Experimenting with SAC RL

## Installation

Create a venv and install dependencies:

```bash
uv venv
uv sync
```

## Training

Run training with default hyperparameters:

```bash
uv run python main.py
```

Training automatically saves checkpoints to `checkpoints/` and logs to `runs/` for TensorBoard.

### Training options

```bash
# Custom hyperparameters
uv run python main.py --rollout-len 256 --batch-size 256 --num-batches 256

# Different environment
uv run python main.py --env-id Pendulum-v1

# Custom model architecture
uv run python main.py --actor-d-model 256 --critic-d-model 256 --actor-lr 1e-3 --critic-lr 3e-4

# Adjust evaluation frequency
uv run python main.py --eval-every 1000 --eval-episodes 10
```

## Evaluation

Watch a trained agent with human-rendered visualization:

```bash
uv run python main.py --eval
```

This will:
- Load the best checkpoint (matching your current hyperparameters)
- Render the environment in a window
- Run episodes indefinitely until you press Ctrl+C

### Loading a specific checkpoint

```bash
# Load a specific checkpoint file (skips config validation)
uv run python main.py --eval --checkpoint checkpoints/my_checkpoint.pth
```

**Note:** When using `--eval` without `--checkpoint`, the hyperparameters (model dimensions, learning rates, etc.) must match the saved checkpoint. If you trained with custom hyperparameters, pass the same values:

```bash
# If you trained with custom model size
uv run python main.py --eval --actor-d-model 256 --critic-d-model 256
```

## Monitoring

View training metrics in TensorBoard:

```bash
uv run tensorboard --logdir runs/
```

## All options

```bash
uv run python main.py --help
```
