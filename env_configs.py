# Environment-specific configurations
# Add new environments here - only specify values that differ from DEFAULTS

DEFAULTS = {
    "env_kwargs": {},
    "reward_scale": 1.0,
    "actor": {
        "d_model": 256,
        "lr": 3e-4,
    },
    "critic": {
        "d_model": 256,
        "lr": 3e-4,
    },
    "sac": {
        "gamma": 0.99,
        "tau": 0.005,
        "alpha_lr": 3e-4,
    },
    "training": {
        "total_steps": 1_000_000,
        "warmup_steps": 10000,
        "batch_size": 256,
        "rollout_len": 256,
        "num_batches": 256,
        "eval_every": 1000,
        "eval_episodes": 10,
    },
}

ENV_CONFIGS = {
    # Classic control
    "Pendulum-v1": {
        "env_kwargs": {"g": 9.81},
        "reward_scale": 0.01,
        "actor": {"d_model": 64},
        "critic": {"d_model": 64},
        "training": {
            "total_steps": 100_000,
            "warmup_steps": 2000,
        },
    },

    # MuJoCo locomotion
    "HalfCheetah-v5": {
        "actor": {"d_model": 256},
        "critic": {"d_model": 256},
        "training": {
            "total_steps": 1_000_000,
            "warmup_steps": 10000,
        },
    },
    "Hopper-v5": {
        "actor": {"d_model": 256},
        "critic": {"d_model": 256},
        "training": {
            "total_steps": 1_000_000,
            "warmup_steps": 10000,
        },
    },
    "Walker2d-v5": {
        "actor": {"d_model": 256},
        "critic": {"d_model": 256},
        "training": {
            "total_steps": 1_000_000,
            "warmup_steps": 10000,
        },
    },
    "Ant-v5": {
        "actor": {"d_model": 256},
        "critic": {"d_model": 256},
        "training": {
            "total_steps": 2_000_000,
            "warmup_steps": 10000,
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
    "HumanoidStandup-v5": {
        "actor": {"d_model": 512},
        "critic": {"d_model": 512},
        "training": {
            "total_steps": 5_000_000,
            "warmup_steps": 25000,
        },
    },
}


def deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base dict."""
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_config(env_id: str) -> dict:
    """Get merged config for an environment."""
    env_overrides = ENV_CONFIGS.get(env_id, {})
    return deep_merge(DEFAULTS, env_overrides)
