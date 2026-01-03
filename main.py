import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import os

from configuration import sac_config, make_hparams_dict
from checkpoint import CheckpointManager, make_paths
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from agent import Agent
from mlp import ActorNetwork, CriticNetwork
from configuration import ActorModelConfig, CriticModelConfig

eval_mode = False
rl_config = sac_config
actor_config = ActorModelConfig(d_in=3, d_model=256, d_out=1)
critic_config = CriticModelConfig(action_dim=1, state_dim=3, d_model=256, d_out=1)

if __name__ == '__main__':

    env = gym.make("Pendulum-v1", render_mode="human", g=9.81)
    # env = gym.make("HalfCheetah-v5")

    # Model is defined in the Agent class
    actor = ActorNetwork(actor_config)
    critic = CriticNetwork(critic_config)
    critic_target = CriticNetwork(critic_config)
    agent = Agent(rl_config, actor, critic, critic_target)
    ckpt = CheckpointManager(agent, rl_config, actor_config, critic_config)
    paths = make_paths(agent, "flappy_bird", rl_config.rl_type)
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(rl_config.total_train_steps * rl_config.rollout_len),
        batch_size=sac_config.batch_size
    )
    writer = SummaryWriter(paths["tensorboard_dir"])
    writer.add_text("hparams", str(make_hparams_dict(rl_config.as_dict(), actor_config.as_dict(), critic_config.as_dict())))
    
    resumed = False
    # Resume checkpoint if it exists
    if os.path.exists(paths["checkpoint_recent"]):
        resumed = ckpt.load(paths["checkpoint_recent"], map_location=agent.device)

    state_t1, _ = env.reset()
    terminated = False
    rewards_fifo = deque(maxlen=10000)
    max_cum_rewards = 0
    if not resumed:
        ckpt.max_1k_rew = float('-inf')
    global_step = 0
    

    while ckpt.training_step < rl_config.total_train_steps:
        
        for _ in range(rl_config.rollout_len):
            
            # Collect s, a, r, s', d
            state_t = state_t1

            # Convert state to tensor before passing to actor
            state_tensor = torch.as_tensor(state_t, dtype=rl_config.dtype).to(agent.device).unsqueeze(0)
            
            if global_step > rl_config.warmup_steps:
                action_t_tensor, _ = actor.forward(state_tensor, deterministic=eval_mode)
            else:
                action_t_tensor = torch.empty(1, 1).uniform_(-1, 1)

            # Gymnasium needs a numpy array, detach from graph
            action_t = action_t_tensor.detach().cpu().numpy()[0]
            
            # Step the environment, action is from [-2.0, 2.0] so scaled by 2x
            state_t1, reward, terminated, truncated, info = env.step(action_t * 2.0)         
            needs_reset = terminated or truncated
            done_t = float(needs_reset)
            reward_t = reward

            if eval_mode == False:
                data = TensorDict({
                    "state": torch.tensor(state_t, dtype=rl_config.dtype).unsqueeze(0),
                    "action": torch.tensor(action_t, dtype=rl_config.dtype).unsqueeze(0), 
                    "reward": torch.tensor(reward_t, dtype=rl_config.dtype).unsqueeze(0), 
                    "next_state": torch.tensor(state_t1, dtype=rl_config.dtype).unsqueeze(0), 
                    "dones": torch.tensor(done_t, dtype=rl_config.dtype).unsqueeze(0)
                }, batch_size=[1])
                replay_buffer.add(data)
            # ckpt.max_cum_rewards = max(info.get("score"), ckpt.max_cum_rewards)

            rewards_fifo.append(reward_t)

            # If done, start a new game
            if needs_reset:
                state_t1, _ = env.reset()

            global_step += 1
        
        # Make sure warmup phase is completed before we start training the model
        if eval_mode == False and global_step >= rl_config.warmup_steps:
            
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
            writer.add_scalar("Max High Score", ckpt.max_cum_rewards, global_step=ckpt.training_step)
            print(
                f"Reward: {reward_t:.2f},\t Done: {done_t},\t"
                f"Action: {action_t},\t Info: {info},\t Last 1k Rew: {last_1k_rew:.2f}"
            )
        
    writer.close()
    env.close()
