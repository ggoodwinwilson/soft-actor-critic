import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from configuration import CriticModelConfig, ActorModelConfig

class ActorNetwork(nn.Module):
    def __init__(self, config:ActorModelConfig):
        super().__init__()
        self.optim = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.output_dim = config.d_out
        self.net = nn.Sequential(
            nn.Linear(config.d_in, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU()
        )
        self.mu = nn.Linear(config.d_model, config.d_out)
        self.log_std = nn.Linear(config.d_model, config.d_out)

    def forward(self, obs, deterministic=False):
        x = self.net(obs)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = torch.exp(log_std)

        # After independent is applied, dist.log_prob will sum the log probs
        # ex. AND the probabilities P(Total) = P(Motor 1) x P(Motor 2) x P(Motor 3)
        base = Independent(Normal(mu, std), 1)

        # squish the distribution into -1, +1 with the tanh transform
        dist = TransformedDistribution(base, [TanhTransform(cache_size=1)])

        if deterministic:
            z = mu
            action = torch.tanh(z)
            logp = dist.log_prob(action)
        else:
            # reparameterization trick with rsample to keep it differentiable
            action = dist.rsample()
            logp = dist.log_prob(action)

        return action, logp
    
        
class CriticNetwork(nn.Module):
    def __init__(self, config:CriticModelConfig):
        super().__init__()
        self.optim = optim.Adam(self.parameters(), lr=config.learning_rate)
        
        self.net1 = nn.Sequential(
            nn.Linear(config.action_dim + config.state_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_out)
        )
        self.net2 = nn.Sequential(
            nn.Linear(config.action_dim + config.state_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_out)
        )
        
    def forward(self, state, action):
        
        sa = torch.cat([state, action], dim=-1)
        q1 = self.net1(sa)
        q2 = self.net2(sa)
        
        return q1, q2
        
    def state_dict(self):

        return {
            "actor_model": self.actor.state_dict(),
            "critic_model": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_scheduler": self.actor_scheduler.state_dict() if getattr(self, "scheduler", None) else None,
            "critic_scheduler": self.critic_scheduler.state_dict() if getattr(self, "scheduler", None) else None,
            "scaler": self.scaler.state_dict() if getattr(self, "scaler", None) else None,
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor_model"])
        self.critic.load_state_dict(state["critic_model"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if getattr(self, "actor_scheduler", None) and state["actor_scheduler"] is not None:
            self.actor_scheduler.load_state_dict(state["actor_scheduler"])
        if getattr(self, "critic_scheduler", None) and state["critic_scheduler"] is not None:
            self.critic_scheduler.load_state_dict(state["critic_scheduler"])
        if getattr(self, "scaler", None) and state["scaler"] is not None:
            self.scaler.load_state_dict(state["scaler"])
