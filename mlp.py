import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from configuration import CriticModelConfig, ActorModelConfig

class ActorNetwork(nn.Module):
    def __init__(self, config:ActorModelConfig):
        super().__init__()
        self.output_dim = config.d_out
        
        self.net = nn.Sequential(
            nn.Linear(config.d_in, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU()
        )
        self.mu = nn.Linear(config.d_model, config.d_out)
        self.log_std = nn.Linear(config.d_model, config.d_out)
        
        self.optim = optim.Adam(self.parameters(), lr=config.learning_rate)

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
        
        self.optim = optim.Adam(self.parameters(), lr=config.learning_rate)
        
    def forward(self, state, action):
        
        sa = torch.cat([state, action], dim=-1)
        q1 = self.net1(sa)
        q2 = self.net2(sa)
        
        return q1, q2
        
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state, *args, **kwargs):
        return super().load_state_dict(state, *args, **kwargs)
