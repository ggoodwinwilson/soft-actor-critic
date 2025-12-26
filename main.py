import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import os

from configuration import sac_config, mlp_config, transformer_config, make_hparams_dict
from checkpoint import CheckpointManager, make_paths
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from agent import Agent
from configuration import MLPConfig

run_mode = "train"
# run_mode = "eval"
rl_config = sac_config
actor_config = MLPConfig(d_in=3, d_model=256, d_out=1)
critic_config = MLPConfig(d_in=3, d_model=256, d_out=1)

if __name__ == '__main__':

    env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)
    # env = gym.make("HalfCheetah-v5")

    # Model is defined in the Agent class
    agent = Agent(rl_config, actor_config, critic_config)
    ckpt = CheckpointManager(agent, rl_config)
    paths = make_paths(agent, "flappy_bird", rl_config.rl_type)
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(),
        batch_size=model_config.batch_size
    )
    writer = SummaryWriter(paths["tensorboard_dir"])
    writer.add_text("hparams", str(make_hparams_dict(rl_config, model_config)))
    
    # Resume checkpoint if it exists
    if os.path.exists(paths["checkpoint_recent"]):
        ckpt.load(paths["checkpoint_recent"], map_location=agent.device)

    # Initialize the first 9 observations in the episode for xfmr model
    obs_t1, _ = env.reset()
    terminated = False
    rewards_fifo = deque(maxlen=10000)
    max_high_score = 0
    ckpt.max_1k_rew = float('-inf')

    while ckpt.training_step < rl_config.total_train_steps:
        
        for _ in range(rl_config.rollout_len):
            obs_t = obs_t1
            action_dist = agent.forward(torch.tensor(obs_t).unsqueeze(0).to(agent.device))
            value_t = value_t.item()
            if run_mode == "train":
                action = action_dist.sample()
            else:
                action = torch.argmax(action_dist.probs)
            action_t = int(action.item())
            log_prob_t = action_dist.log_prob(action).item()
            
            # Step the environment
            obs_t1, reward, terminated, truncated, info = env.step(action_t)         
            done_t = terminated
            needs_reset = terminated or truncated
            reward_t = reward

            if run_mode == "train":
                data = TensorDict({
                    "obs": torch.tensor(obs_t).unsqueeze(0),
                    "next_obs": torch.tensor(obs_t1).unsqueeze(0), 
                    "reward": torch.tensor(reward_t).unsqueeze(0), 
                    "action": torch.tensor(action_t).unsqueeze(0), 
                    "dones": torch.tensor(done_t).unsqueeze(0)
                }, batch_size=[1])
                replay_buffer.add(data)
            ckpt.max_high_score = max(info.get("score"), ckpt.max_high_score)

            rewards_fifo.append(reward_t)

            # If done, start a new game
            if needs_reset:
                obs_t1, _ = env.reset()
                obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype)
        
        if run_mode == "train":
            
            # Train the model
            agent.learn(replay_buffer, writer, ckpt.training_step)
            ckpt.training_step += 1
            
            # Log training statsand save models
            last_1k_rew = sum(list(rewards_fifo)[-1000:])
            if last_1k_rew > ckpt.max_1k_rew:
                ckpt.max_1k_rew = last_1k_rew
                ckpt.save(paths["checkpoint_best"])
            if ckpt.training_step % 5 == 0:
                ckpt.save(paths["checkpoint_recent"])
            writer.add_scalar("Total 1000 Step Reward", last_1k_rew, global_step=ckpt.training_step)
            writer.add_scalar("Max High Score", ckpt.max_high_score, global_step=ckpt.training_step)
            print(
                f"Value: {value_t:.2f},\tReward: {reward_t:.2f},\tDone: {done_t},\t"
                f"Action: {action_t},\tLog prob: {log_prob_t:.2f}\t, Info: {info},\t Last 1k Rew: {last_1k_rew:.2f}"
            )
    
    writer.close()
    env.close()
