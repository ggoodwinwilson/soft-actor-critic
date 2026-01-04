from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ActorModelConfig:
    model_type: str = "mlp"
    d_in: int = 180
    d_model: int = 256
    d_out: int = 10
    learning_rate: float = 3e-4
    dtype: torch.dtype = torch.float32

    def as_dict(self):
        return self.__dict__
    
@dataclass(frozen=True)
class CriticModelConfig:
    model_type: str = "mlp"
    action_dim: int = 180
    state_dim: int = 180
    d_model: int = 256
    d_out: int = 10
    learning_rate: float = 3e-4
    dtype: torch.dtype = torch.float32

    def as_dict(self):
        return self.__dict__
    
    
@dataclass(frozen=True)
class SACConfig:
    rl_type: str = 'sac'
    rollout_len: int = 128
    batch_size: int = 256
    num_batches: int = 128
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    total_train_steps: int = 100_000
    dtype: torch.dtype = torch.float32
    tau: float = 0.005
    warmup_steps: int = 5_000
    

    def as_dict(self):
        return self.__dict__
    
sac_config = SACConfig()

def make_hparams_dict(*dicts):

    d = {}
    for cfg in dicts:
        for k, v in cfg.items():
            if isinstance(v, torch.dtype):
                v = str(v)
            d[k] = v

    return d
