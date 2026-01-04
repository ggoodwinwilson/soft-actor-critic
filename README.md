# soft-actor-critic
Experimenting with SAC RL

## Installation

Create a venv and install dependencies:

```bash
uv venv
uv sync
```

## Training

Run training (uses environment-specific defaults from `env_configs.py`):

```bash
# Pendulum (default)
uv run python main.py

# Other environments
uv run python main.py --env-id HalfCheetah-v5
uv run python main.py --env-id Humanoid-v5
```

Training automatically saves checkpoints to `checkpoints/` and logs to `runs/` for TensorBoard.

### CLI overrides

CLI arguments override config values for quick experimentation:

```bash
# Override training params
uv run python main.py --total-train-steps 500000 --batch-size 512

# Override model architecture
uv run python main.py --actor-d-model 256 --critic-d-model 256

# Override reward scaling
uv run python main.py --reward-scale 0.1
```

## Environment Configuration

All environment-specific settings are centralized in `env_configs.py`:

```python
# env_configs.py
ENV_CONFIGS = {
    "Pendulum-v1": {
        "env_kwargs": {"g": 9.81},      # Environment constructor args
        "reward_scale": 0.01,            # Reward scaling factor
        "actor": {"d_model": 64},        # Actor network size
        "critic": {"d_model": 64},       # Critic network size
        "training": {
            "total_steps": 100_000,
            "warmup_steps": 2000,
        },
    },
    "Humanoid-v5": {
        "actor": {"d_model": 512},
        "critic": {"d_model": 512},
        "training": {
            "total_steps": 5_000_000,
            "warmup_steps": 25000,
        },
    },
    # ... more environments
}
```

### Adding a new environment

1. Open `env_configs.py`
2. Add an entry to `ENV_CONFIGS` (only specify values that differ from `DEFAULTS`)
3. Run with `--env-id YourEnv-v1`

Unspecified values inherit from `DEFAULTS`.

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

**Note:** When using `--eval` without `--checkpoint`, the hyperparameters must match the saved checkpoint. The easiest way is to just use the same `--env-id`:

```bash
uv run python main.py --eval --env-id HalfCheetah-v5
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
