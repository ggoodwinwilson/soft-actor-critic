from torch import nn, optim
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from configuration import SACConfig
from transformer import Transformer
from mlp import ActorNetwork, CriticNetwork
from torchrl.data import ReplayBuffer
    
class Agent:
    def __init__(self, rl_config:SACConfig, actor:ActorNetwork, critic:CriticNetwork, critic_target:CriticNetwork):
        
        # RL Hyperparmas
        self.rl_config = rl_config
        self.rollout_len = rl_config.rollout_len
        self.gamma = rl_config.gamma
        self.tau = rl_config.tau
        self.alpha_lr = rl_config.alpha_lr
        self.batch_size = rl_config.batch_size
        self.num_batches = rl_config.num_batches
        self.dtype = rl_config.dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.target_entropy = -actor.output_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.opt_alpha = optim.Adam([self.log_alpha], lr=self.alpha_lr)

    def learn(self, replay_buffer: ReplayBuffer, writer:SummaryWriter, global_timestep:int):
        
        for _ in range(self.num_batches):
            batch = replay_buffer.sample()

            # s, a, r, s', d buffers
            state_buf = batch["state"].to(self.device)
            action_buf = batch["action"].to(self.device)
            reward_buf = batch["reward"].to(self.device)
            next_state_buf = batch["next_state"].to(self.device)
            dones_buf = batch["dones"].to(self.device)

            # a ~ π(-∣s), logπ(a∣s) 
            action, action_logp = self.actor.forward(state_buf)

            # Q(s, a)
            q1, q2 = self.critic.forward(state_buf, action)

            # α
            alpha = self.log_alpha.exp()

            # Don't calc gradient for these variables
            with torch.no_grad():
                # a' ~ π(-|s'), logπ(a'|s')
                next_action, next_action_logp = self.actor.forward(next_state_buf)

                # Q'(s', a')
                q_target1, q_target2 = self.critic_target.forward(next_state_buf, next_action)
                q_target = torch.min(q_target1, q_target2)

                # r + γ(imin​Q′(s′,a′) − αlogπ(a′∣s′))
                target = reward_buf + self.gamma * (1 - dones_buf) * (q_target - alpha * next_action_logp)

            # L_q = E[(Q(s,a) − (r + γ(imin​Q′(s′,a′) − αlogπ(a′∣s′))))2]
            loss_q = F.mse_loss(q1, target) + F.mse_loss(q2, target)

            # L_pi = E[αlogπ(a∣s)−Q(s,a)]
            loss_pi = (alpha * action_logp - q.detach()).mean()

            # L_alpha = E[−α(logπ(a∣s) + Htarget​)]
            loss_alpha = (-alpha * (action_logp.detach() + self.target_entropy)).mean()

            self.critic.optim.zero_grad()
            loss_q.backward()
            self.critic.optim.step()

            self.actor.optim.zero_grad()
            loss_pi.backward()
            self.actor.optim.step()

            self.opt_alpha.zero_grad()
            loss_alpha.backward()
            self.opt_alpha.step()

            self.critic_target_update()

    def critic_target_update(self):
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(),
                                self.critic_target.parameters()):
                p_targ.data.mul_(1 - self.tau)
                p_targ.data.add_(self.tau * p.data)

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