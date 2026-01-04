"""
Dataclass definitions for configs.
Default VALUES live in env_configs.py - these are just the structure/schema.
"""
from dataclasses import dataclass, asdict
import torch


@dataclass(frozen=True)
class ActorModelConfig:
    model_type: str
    d_in: int
    d_model: int
    d_out: int
    learning_rate: float
    dtype: torch.dtype = torch.float32

    def as_dict(self):
        return {k: str(v) if isinstance(v, torch.dtype) else v for k, v in asdict(self).items()}


@dataclass(frozen=True)
class CriticModelConfig:
    model_type: str
    action_dim: int
    state_dim: int
    d_model: int
    d_out: int
    learning_rate: float
    dtype: torch.dtype = torch.float32

    def as_dict(self):
        return {k: str(v) if isinstance(v, torch.dtype) else v for k, v in asdict(self).items()}


@dataclass(frozen=True)
class SACConfig:
    rl_type: str
    rollout_len: int
    batch_size: int
    num_batches: int
    alpha_lr: float
    gamma: float
    total_train_steps: int
    tau: float
    warmup_steps: int
    dtype: torch.dtype = torch.float32

    def as_dict(self):
        return {k: str(v) if isinstance(v, torch.dtype) else v for k, v in asdict(self).items()}


def make_hparams_dict(*dicts):
    """Flatten multiple config dicts into one for logging."""
    result = {}
    for cfg in dicts:
        for k, v in cfg.items():
            if isinstance(v, torch.dtype):
                v = str(v)
            result[k] = v
    return result
