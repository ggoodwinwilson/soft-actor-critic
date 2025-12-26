from torch import nn, optim
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from configuration import PPOConfig
from transformer import Transformer
from mlp import MLP
from torchrl.data import ReplayBuffer
    
class Agent:
    def __init__(self, rl_config, actor_config, critic_config):
        
        # RL Hyperparmas
        self.rl_config = rl_config
        self.rollout_len = self.rl_config.rollout_len
        self.gamma = self.rl_config.gamma
        self.td_lambda = self.rl_config.td_lambda
        self.eps_clip = self.rl_config.eps_clip
        self.ent_coef = self.rl_config.ent_coef
        self.critic_coef = self.rl_config.critic_coef

        # Model Hyperparams
        self.actor_config = actor_config
        self.critic_config = critic_config
        self.batch_size = self.model_config.batch_size
        self.num_batches = self.model_config.num_batches
        self.dtype = self.model_config.dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def learn(self, replay_buffer: ReplayBuffer, writer:SummaryWriter, global_timestep:int):
        
        for batch in num_batches:
        batch = replay_buffer.sample()

        action_buf = batch["action"].to(self.device)
        rewards_buf = batch["reward"].to(self.device)
        obs_buf = batch["obs"].to(self.device)
        next_obs_buf = batch["next_obs"].to(self.device)
        dones_buf = batch["dones"].to(self.device)

        with torch.no_grad():
            
            q_min = min()
            target_q = rewards_buf + self.gamma * (1 - dones_buf) * (min_q - self.alpha *  

        # Sample an action from the current policy for the next state
        # a' ~ π(-|s')
        next_action_dist, _ = self.forward(next_obs_buf)
        next_action = next_action_dist.sample()
        next_action_log_prob = next_action_dist.log_prob(next_action)

        # Sample an action from the current policy for the current state
        action_dist, _ = self.forward(obs_buf)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)

        q1 = self.model.critic1(obs_buf)
        q1_next = self.model.critic1(next_obs_buf, next_action)
        q2_next = self.model.critic2(next_obs_buf, next_action)

        # L_q = E[(Q(s,a) − (r + γ(imin​Q′(s′,a′) − αlogπ(a′∣s′))))2]
        loss_q = ((q - (r + self.gamma*(min(q1_next.detach(), q2_next.detach())) - self.alpha.detach() * next_action_log_prob))^2).mean()

        # L_pi = E[αlogπ(a∣s)−Q(s,a)]
        loss_pi = (self.alpha * action_log_prob - q).mean()

        # L_alpha = E[−α(logπ(a∣s) + Htarget​)]
        loss_alpha = (-self.alpha * (action_log_prob + H_target)).mean()

        self.model.optim_zero_grad()
        total_loss.backward()
        self.model.optim_step()

    def critic_target_update(self):
        with torch.no_grad():
            for p, p_targ in zip(critic.parameters(),
                                critic_target.parameters()):
                p_targ.data.mul_(1 - tau)
                p_targ.data.add_(tau * p.data)
    
    def forward(self, x):
        return self.model.forward(x)
        
    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "actor_opt": getattr(self.model, "actor_optimizer", None) and self.model.actor_optimizer.state_dict(),
            "critic_opt": getattr(self.model, "critic_optimizer", None) and self.model.critic_optimizer.state_dict(),
            "optimizer": getattr(self.model, "optimizer", None) and self.model.optimizer.state_dict(),
            "scheduler": getattr(self, "scheduler", None) and self.scheduler.state_dict(),
            "scaler": getattr(self, "scaler", None) and self.scaler.state_dict(),
            "config": {
                "ppo": self.rl_config.as_dict(),
                "model": self.model_config.as_dict(),
            },
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        if state.get("actor_opt") and hasattr(self.model, "actor_optimizer"):
            self.model.actor_optimizer.load_state_dict(state["actor_opt"])
        if state.get("critic_opt") and hasattr(self.model, "critic_optimizer"):
            self.model.critic_optimizer.load_state_dict(state["critic_opt"])
        if state.get("optimizer") and hasattr(self.model, "optimizer"):
            self.model.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") and hasattr(self, "scheduler"):
            self.scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") and hasattr(self, "scaler"):
            self.scaler.load_state_dict(state["scaler"])