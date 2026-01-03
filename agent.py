from torch import nn, optim
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from configuration import SACConfig
# from transformer import Transformer
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

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.critic_target = critic_target.to(self.device)
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
            q1, q2 = self.critic.forward(state_buf, action_buf)
            q = torch.min(q1, q2)

            q1_pi, q2_pi = self.critic.forward(state_buf, action)
            q_pi = torch.min(q1_pi, q2_pi)

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
                target = reward_buf + self.gamma * (1.0 - dones_buf) * (q_target - alpha * next_action_logp)

            # L_q = E[(Q(s,a) − (r + γ(imin​Q′(s′,a′) − αlogπ(a′∣s′))))2]
            loss_q = F.mse_loss(q1, target) + F.mse_loss(q2, target)

            # L_pi = E[αlogπ(a∣s)−Q(s,a)]
            loss_pi = (alpha.detach() * action_logp - q_pi).mean()

            # L_alpha = E[−α(logπ(a∣s) + Htarget​)]
            loss_alpha = (-alpha * (action_logp.detach() + self.target_entropy)).mean()

            # Backward passes (Grabs gradients for all networks)
            self.critic.optim.zero_grad()
            loss_q.backward()

            self.actor.optim.zero_grad()
            loss_pi.backward()

            self.opt_alpha.zero_grad()
            loss_alpha.backward()

            # Update steps (Modifies weights only after all grads are computed)
            self.critic.optim.step()
            self.actor.optim.step()
            self.opt_alpha.step()

            # Target update
            self.critic_target_update()

    def critic_target_update(self):
        # This block is the "modern" safety net
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                # Polyak averaging: target = (1 - tau) * target + tau * source
                # We use .lerp_ (Linear Interpolation) which is exactly this math
                p_targ.lerp_(p, self.tau)

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor.optim.state_dict(),
            "critic_opt": self.critic.optim.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "opt_alpha": self.opt_alpha.state_dict(),
            "config": {
                "sac": self.rl_config.as_dict(),
            },
        }

    def load_state_dict(self, state):
        if state.get("actor") is not None:
            self.actor.load_state_dict(state["actor"])
        if state.get("critic") is not None:
            self.critic.load_state_dict(state["critic"])
        if state.get("critic_target") is not None:
            self.critic_target.load_state_dict(state["critic_target"])
        if state.get("actor_opt"):
            self.actor.optim.load_state_dict(state["actor_opt"])
        if state.get("critic_opt"):
            self.critic.optim.load_state_dict(state["critic_opt"])
        if state.get("log_alpha") is not None:
            self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        if state.get("opt_alpha"):
            self.opt_alpha.load_state_dict(state["opt_alpha"])
