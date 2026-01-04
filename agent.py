from torch import nn, optim
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from configuration import SACConfig
# from transformer import Transformer
from mlp import ActorNetwork, CriticNetwork
from torchrl.data import ReplayBuffer
from mlp import plot_policy_pdf
    
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
        self.log_alpha = torch.tensor([np.log(0.1)], dtype=rl_config.dtype, requires_grad=True, device=self.device)
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
            reward_buf = reward_buf.unsqueeze(-1)
            dones_buf = dones_buf.unsqueeze(-1)

            # α
            alpha = self.log_alpha.exp()

            # ========== CRITIC UPDATE ==========
            # Q(s, a) using actions from replay buffer
            q1, q2 = self.critic.forward(state_buf, action_buf)
            q = torch.min(q1, q2)

            # Compute target (no gradients needed)
            with torch.no_grad():
                # a' ~ π(-|s'), logπ(a'|s')
                next_action, next_action_logp = self.actor.forward(next_state_buf)
                next_action_logp = next_action_logp.unsqueeze(-1)

                # Q'(s', a')
                q_target1, q_target2 = self.critic_target.forward(next_state_buf, next_action)
                q_target = torch.min(q_target1, q_target2)

                # r + γ(min Q′(s′,a′) − α logπ(a′∣s′))
                target = reward_buf + self.gamma * (1.0 - dones_buf) * (q_target - alpha * next_action_logp)

            # L_q = E[(Q(s,a) − target)²]
            loss_q = F.mse_loss(q1, target) + F.mse_loss(q2, target)

            self.critic.optim.zero_grad()
            loss_q.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic.optim.step()

            # ========== ACTOR UPDATE ==========
            # Sample fresh actions from current policy
            action, action_logp = self.actor.forward(state_buf)
            action_logp = action_logp.unsqueeze(-1)

            # Q(s, π(s)) - freeze critic so gradients only flow through action → actor
            for p in self.critic.parameters():
                p.requires_grad = False
            q1_pi, q2_pi = self.critic.forward(state_buf, action)
            q_pi = torch.min(q1_pi, q2_pi)
            for p in self.critic.parameters():
                p.requires_grad = True

            # L_pi = E[α logπ(a∣s) − Q(s,a)]
            loss_pi = (alpha.detach() * action_logp - q_pi).mean()

            self.actor.optim.zero_grad()
            loss_pi.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor.optim.step()

            # ========== ALPHA UPDATE ==========
            # L_alpha = E[−α(logπ(a∣s) + H_target)]
            loss_alpha = (-alpha * (action_logp.detach() + self.target_entropy)).mean()

            self.opt_alpha.zero_grad()
            loss_alpha.backward()
            self.opt_alpha.step()

            # ========== TARGET UPDATE ==========
            self.critic_target_update()

            # Log losses
            if writer is not None:
                step = global_timestep * self.num_batches + _
                td_error = torch.abs(target - q).mean().item()
                writer.add_scalar("Loss/critic", loss_q.item(), step)
                writer.add_scalar("Loss/actor", loss_pi.item(), step)
                writer.add_scalar("Loss/alpha", loss_alpha.item(), step)
                writer.add_scalar("Values/alpha", alpha.item(), step)
                writer.add_scalar("Values/q1", q1.mean().item(), step)
                writer.add_scalar("Values/q2", q2.mean().item(), step)
                writer.add_scalar("Values/q_mean", q.mean().item(), step)
                writer.add_scalar("Values/entropy", (-action_logp).mean().item(), step)
                writer.add_scalar("Diagnostic/Bellman_Error", td_error, step)
                writer.add_scalar("Diagnostic/critic_grad_norm", float(critic_grad_norm), step)
                writer.add_scalar("Diagnostic/actor_grad_norm", float(actor_grad_norm), step)

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


