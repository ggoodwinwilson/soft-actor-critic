from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ActorModelConfig:
    model_type = "mlp"
    d_in: int = 180
    d_model: int = 256
    d_out: int = 10
    learning_rate: float = 1e-3
    dtype: torch.dtype = torch.float32

    def as_dict(self):
        return self.__dict__
    
@dataclass(frozen=True)
class CriticModelConfig:
    model_type = "mlp"
    action_dim: int = 180
    state_dim: int = 180
    d_model: int = 256
    d_out: int = 10
    learning_rate: float = 1e-3
    dtype: torch.dtype = torch.float32

    def as_dict(self):
        return self.__dict__
    
    
@dataclass(frozen=True)
class SACConfig:
    rl_type: str = 'sac'
    rollout_len: int = 256
    batch_size: int = 256
    num_batches: int = 10
    alpha_lr: float = 1e-3
    gamma: float = 0.99
    total_train_steps: int = 100_000
    dtype: torch.dtype = torch.float32
    tau: float = 0.005
    warmup_steps: int = 10_000
    

    def as_dict(self):
        return self.__dict__
    
sac_config = SACConfig()

def make_hparams_dict(*dicts):

    d = {}
    for cfg in dicts:
        for k, v in cfg:
            if isinstance(v, torch.dtype):
                v = str(v)
            d[k] = v

    return d