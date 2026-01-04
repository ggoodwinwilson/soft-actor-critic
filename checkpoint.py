import dataclasses
import os
import random

import numpy as np
import torch

from agent import Agent

class CheckpointManager:
    def __init__(self, agent: Agent, rl_config, actor_config=None, critic_config=None):
        self.agent = agent
        self.rl_config = rl_config
        self.actor_config = actor_config
        self.critic_config = critic_config

        # training state you want to track
        self.epoch = 0
        self.training_step = 0
        self.batch_idx = 0
        self.max_cum_rewards = 0
        self.max_1k_rew = float("-inf")

    # Helpers for scalar training state
    def _train_keys(self):
        # Add new scalar fields here only
        return [
            "epoch",
            "training_step",
            "batch_idx",
            "max_cum_rewards",
            "max_1k_rew",
        ]

    def _get_train_state(self):
        return {k: getattr(self, k) for k in self._train_keys()}

    def _set_train_state(self, d):
        for k in self._train_keys():
            if k in d:
                setattr(self, k, d[k])

    # RNG helpers
    def _collect_rng(self):
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def _restore_rng(self, state):
        if not state:
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch_state = state["torch"]
        if torch.is_tensor(torch_state):
            torch_state = torch_state.cpu()
        torch.set_rng_state(torch_state)
        if torch.cuda.is_available() and state.get("torch_cuda") is not None:
            cuda_states = []
            for s in state["torch_cuda"]:
                cuda_states.append(s.cpu() if torch.is_tensor(s) else s)
            torch.cuda.set_rng_state_all(cuda_states)

    # ---- main API ----
    def state_dict(self):
        config = self._current_config_snapshot()
        return {
            "agent": self.agent.state_dict(),
            "rng_state": self._collect_rng(),
            "train_state": self._get_train_state(),
            "config": config,
        }

    def load_state_dict(self, ckpt):
        self.agent.load_state_dict(ckpt["agent"])
        self._restore_rng(ckpt.get("rng_state"))
        self._set_train_state(ckpt.get("train_state", {}))
        # config is optional to actually use on load

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def _current_config_snapshot(self):
        config = {}
        if dataclasses.is_dataclass(self.rl_config):
            config["rl_config"] = dataclasses.asdict(self.rl_config)
        if dataclasses.is_dataclass(self.actor_config):
            config["actor_config"] = dataclasses.asdict(self.actor_config)
        if dataclasses.is_dataclass(self.critic_config):
            config["critic_config"] = dataclasses.asdict(self.critic_config)
        return config

    def _configs_compatible(self, saved_config):
        if not saved_config:
            return False
        return saved_config == self._current_config_snapshot()

    def load(self, path: str, map_location="cpu"):
        raw = torch.load(path, map_location=map_location, weights_only=False)
        if not self._configs_compatible(raw.get("config")):
            return False
        self.load_state_dict(raw)
        return True


def _config_tag(config, prefix):
    if not dataclasses.is_dataclass(config):
        return None
    model_type = getattr(config, "model_type", None)
    d_model = getattr(config, "d_model", None)
    parts = [prefix, model_type]
    if d_model is not None:
        parts.append(f"dim{d_model}")
    return "_".join(p for p in parts if p)


def make_run_id(agent: Agent, env_name, rl_type, tag=None, actor_config=None, critic_config=None):
    model_tag = None
    if hasattr(agent, "model_config"):
        model_type = getattr(agent.model_config, "model_type", None)
        d_model = getattr(agent.model_config, "d_model", None)
        model_parts = [model_type]
        if d_model is not None:
            model_parts.append(f"dim{d_model}")
        model_tag = "_".join(p for p in model_parts if p)
    actor_tag = _config_tag(actor_config, "actor")
    critic_tag = _config_tag(critic_config, "critic")
    parts = [
        env_name,
        rl_type,
        model_tag,
        actor_tag,
        critic_tag,
        tag,
    ]
    return "_".join(p for p in parts if p)

def make_paths(agent: Agent, env_name, rl_type, tag=None, actor_config=None, critic_config=None):
    run_id = make_run_id(agent, env_name, rl_type, tag, actor_config, critic_config)

    return {
        "run_id": run_id,
        "tensorboard_dir": f"runs/{run_id}",
        "checkpoint_recent": f"checkpoints/{run_id}_recent.pth",
        "checkpoint_best": f"checkpoints/{run_id}_best.pth",
    }
