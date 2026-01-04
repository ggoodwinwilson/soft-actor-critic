import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import os
import argparse

from configuration import make_hparams_dict, SACConfig
from checkpoint import CheckpointManager, make_paths
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from agent import Agent
from mlp import ActorNetwork, CriticNetwork
from configuration import ActorModelConfig, CriticModelConfig
from env_configs import get_config

class ActionScaleWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.low = torch.as_tensor(self.action_space.low, dtype=torch.float32)
        self.high = torch.as_tensor(self.action_space.high, dtype=torch.float32)

    def action(self, action):
        action_t = torch.as_tensor(action, dtype=torch.float32)
        scaled = self.low + (action_t + 1.0) * 0.5 * (self.high - self.low)
        return scaled.cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC on a Gymnasium environment.")
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--render-mode", default="rgb_array")
    parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (no training, human render)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file (skips config validation)")
    # All values below default to None - actual defaults come from env_configs.py
    # CLI args override config values when explicitly provided
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--rollout-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--total-train-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--actor-d-model", type=int, default=None)
    parser.add_argument("--critic-d-model", type=int, default=None)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--reward-scale", type=float, default=None)
    return parser.parse_args()

def run_eval(eval_env, agent, dtype, episodes):
    returns = []
    for _ in range(episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_return = 0.0
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=dtype, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                action_t, _ = agent.actor.forward(obs_tensor, deterministic=True)
            action = action_t.detach().cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)
    return float(sum(returns)) / max(len(returns), 1)


def apply_cli_overrides(config: dict, args) -> dict:
    """Apply CLI argument overrides to config. Only overrides if CLI arg is not None."""
    # Training overrides
    if args.rollout_len is not None:
        config["training"]["rollout_len"] = args.rollout_len
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_batches is not None:
        config["training"]["num_batches"] = args.num_batches
    if args.total_train_steps is not None:
        config["training"]["total_steps"] = args.total_train_steps
    if args.warmup_steps is not None:
        config["training"]["warmup_steps"] = args.warmup_steps
    if args.eval_every is not None:
        config["training"]["eval_every"] = args.eval_every
    if args.eval_episodes is not None:
        config["training"]["eval_episodes"] = args.eval_episodes

    # SAC overrides
    if args.gamma is not None:
        config["sac"]["gamma"] = args.gamma
    if args.tau is not None:
        config["sac"]["tau"] = args.tau
    if args.alpha_lr is not None:
        config["sac"]["alpha_lr"] = args.alpha_lr

    # Actor overrides
    if args.actor_d_model is not None:
        config["actor"]["d_model"] = args.actor_d_model
    if args.actor_lr is not None:
        config["actor"]["lr"] = args.actor_lr

    # Critic overrides
    if args.critic_d_model is not None:
        config["critic"]["d_model"] = args.critic_d_model
    if args.critic_lr is not None:
        config["critic"]["lr"] = args.critic_lr

    # Reward scale override
    if args.reward_scale is not None:
        config["reward_scale"] = args.reward_scale

    return config


if __name__ == '__main__':
    args = parse_args()
    eval_mode = args.eval

    # Load environment config and apply CLI overrides
    config = get_config(args.env_id)
    config = apply_cli_overrides(config, args)

    # In eval mode, force human rendering
    render_mode = "human" if eval_mode else args.render_mode

    # Create environment with env-specific kwargs
    env = ActionScaleWrapper(gym.make(args.env_id, render_mode=render_mode, **config["env_kwargs"]))
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    # Build configs from env_configs
    rl_config = SACConfig(
        rl_type="sac",
        rollout_len=config["training"]["rollout_len"],
        batch_size=config["training"]["batch_size"],
        num_batches=config["training"]["num_batches"],
        alpha_lr=config["sac"]["alpha_lr"],
        gamma=config["sac"]["gamma"],
        total_train_steps=config["training"]["total_steps"],
        tau=config["sac"]["tau"],
        warmup_steps=config["training"]["warmup_steps"],
    )
    actor_config = ActorModelConfig(
        model_type="mlp",
        d_in=obs_dim,
        d_model=config["actor"]["d_model"],
        d_out=act_dim,
        learning_rate=config["actor"]["lr"],
    )
    critic_config = CriticModelConfig(
        model_type="mlp",
        action_dim=act_dim,
        state_dim=obs_dim,
        d_model=config["critic"]["d_model"],
        d_out=1,
        learning_rate=config["critic"]["lr"],
    )
    reward_scale = config["reward_scale"]

    # Model is defined in the Agent class
    actor = ActorNetwork(actor_config)
    critic = CriticNetwork(critic_config)
    critic_target = CriticNetwork(critic_config)

    # critic_target needs to be initialized with critic's parameters
    critic_target.load_state_dict(critic.state_dict())
    agent = Agent(rl_config, actor, critic, critic_target)
    ckpt = CheckpointManager(agent, rl_config, actor_config, critic_config)
    paths = make_paths(agent, env.spec.id, rl_config.rl_type, actor_config=actor_config, critic_config=critic_config)

    # Load checkpoint
    resumed = False
    if args.checkpoint:
        # Explicit checkpoint path - force load without config validation
        raw = torch.load(args.checkpoint, map_location=agent.device, weights_only=False)
        ckpt.load_state_dict(raw)
        resumed = True
        print(f"Loaded checkpoint from {args.checkpoint}")
    elif os.path.exists(paths["checkpoint_best"] if eval_mode else paths["checkpoint_recent"]):
        # Auto-load: best for eval, recent for training
        ckpt_path = paths["checkpoint_best"] if eval_mode else paths["checkpoint_recent"]
        resumed = ckpt.load(ckpt_path, map_location=agent.device)
        if resumed:
            print(f"Loaded checkpoint from {ckpt_path}")

    if eval_mode:
        if not resumed:
            print(f"Warning: No checkpoint found. Running with untrained model.")
            print(f"Expected checkpoint at: {paths['checkpoint_best']}")

        # Eval mode: run episodes indefinitely with deterministic policy
        print("Running in evaluation mode (Ctrl+C to exit)")
        episode_idx = 0
        try:
            while True:
                obs, _ = env.reset()
                done = False
                ep_return = 0.0
                while not done:
                    obs_tensor = torch.as_tensor(obs, dtype=rl_config.dtype, device=agent.device).unsqueeze(0)
                    with torch.no_grad():
                        action_t, _ = agent.actor.forward(obs_tensor, deterministic=True)
                    action = action_t.detach().cpu().numpy()[0]
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_return += reward
                    done = terminated or truncated
                episode_idx += 1
                print(f"Episode {episode_idx}: return = {ep_return:.2f}")
        except KeyboardInterrupt:
            print("\nExiting evaluation mode.")
        finally:
            env.close()
    else:
        # Training mode
        raw_eval_env = gym.make(args.env_id, **config["env_kwargs"])
        max_steps = getattr(raw_eval_env.spec, "max_episode_steps", None) or 200
        if not isinstance(raw_eval_env, TimeLimit):
            raw_eval_env = TimeLimit(raw_eval_env, max_episode_steps=max_steps)
        eval_env = ActionScaleWrapper(raw_eval_env)
        eval_every = config["training"]["eval_every"]
        eval_episodes = config["training"]["eval_episodes"]

        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(int(1e6), compilable=True),
            batch_size=rl_config.batch_size
        )
        writer = SummaryWriter(paths["tensorboard_dir"])
        writer.add_text("hparams", str(make_hparams_dict(rl_config.as_dict(), actor_config.as_dict(), critic_config.as_dict())))

        if not resumed:
            ckpt.max_1k_rew = float('-inf')

        state_t1, _ = env.reset()
        rewards_fifo = deque(maxlen=10000)
        episode_return = 0.0
        episode_len = 0
        episode_idx = 0
        global_step = 0

        while ckpt.training_step < rl_config.total_train_steps:

            for _ in range(rl_config.rollout_len):

                # Collect s, a, r, s', d
                state_t = state_t1

                # Convert state to tensor before passing to actor
                state_tensor = torch.as_tensor(state_t, dtype=rl_config.dtype).to(agent.device).unsqueeze(0)

                if global_step > rl_config.warmup_steps:
                    with torch.no_grad():
                        action_t_tensor, _ = actor.forward(state_tensor, deterministic=False)
                else:
                    action_t_tensor = torch.empty(1, 1).uniform_(-1, 1)

                # Gymnasium needs a numpy array, detach from graph
                action_t = action_t_tensor.detach().cpu().numpy()[0]

                # Step the environment with normalized actions
                state_t1, reward, terminated, truncated, info = env.step(action_t)
                needs_reset = terminated or truncated
                done_t = float(terminated)
                reward_t = reward * reward_scale
                episode_return += reward_t
                episode_len += 1

                data = TensorDict({
                    "state": torch.as_tensor(state_t, dtype=rl_config.dtype),
                    "action": torch.as_tensor(action_t, dtype=rl_config.dtype),
                    "reward": torch.as_tensor(reward_t, dtype=rl_config.dtype),
                    "next_state": torch.as_tensor(state_t1, dtype=rl_config.dtype),
                    "dones": torch.as_tensor(done_t, dtype=rl_config.dtype),
                }, batch_size=[])
                replay_buffer.add(data)

                if needs_reset:
                    ckpt.max_cum_rewards = max(ckpt.max_cum_rewards, episode_return)
                    if writer is not None:
                        writer.add_scalar("Episode/return", episode_return, global_step=episode_idx)
                        writer.add_scalar("Episode/length", episode_len, global_step=episode_idx)
                    episode_idx += 1
                    episode_return = 0.0
                    episode_len = 0

                rewards_fifo.append(reward_t)

                # If done, start a new game
                if needs_reset:
                    state_t1, _ = env.reset()

                global_step += 1

            # Make sure warmup phase is completed before we start training the model
            if global_step >= rl_config.warmup_steps:

                # Train the model
                agent.learn(replay_buffer, writer, ckpt.training_step)
                ckpt.training_step += 1

                # Log training stats and save models
                last_1k_rew = sum(list(rewards_fifo)[-1000:])
                if last_1k_rew > ckpt.max_1k_rew:
                    ckpt.max_1k_rew = last_1k_rew
                    ckpt.save(paths["checkpoint_best"])
                if ckpt.training_step % 5 == 0:
                    ckpt.save(paths["checkpoint_recent"])
                writer.add_scalar("Train/last_1k_reward_sum", last_1k_rew, global_step=ckpt.training_step)
                writer.add_scalar("Train/max_episode_return", ckpt.max_cum_rewards, global_step=ckpt.training_step)
                writer.add_scalar("Train/buffer_size", len(replay_buffer), global_step=ckpt.training_step)
                if eval_every and ckpt.training_step % eval_every == 0:
                    avg_return = run_eval(eval_env, agent, rl_config.dtype, eval_episodes)
                    writer.add_scalar("Eval/avg_return", avg_return, global_step=ckpt.training_step)
                print(
                    f"Reward: {reward_t:.2f},\t Done: {done_t},\t"
                    f"Action: {action_t},\t Timestep: {global_step},\t Last 1k Rew: {last_1k_rew:.2f}"
                )

        writer.close()
        env.close()
        eval_env.close()
