import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from configuration import CriticModelConfig, ActorModelConfig

import matplotlib.pyplot as plt
import numpy as np

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
            logp = 0
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
            nn.Linear(config.d_model, config.d_model), # Added layer
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_out)
        )
        
        self.net2 = nn.Sequential(
            nn.Linear(config.action_dim + config.state_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model), # Added layer
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_out)
        )

        # 
        nn.init.uniform_(self.net1[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net1[-1].bias, -3e-3, 3e-3)
        nn.init.uniform_(self.net2[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net2[-1].bias, -3e-3, 3e-3)
        
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


def plot_policy_pdf(actor, obs_batch, device):
    # 1. Select ONLY the first state from the batch
    # obs_batch is [256, state_dim] -> single_obs is [1, state_dim]
    single_obs = obs_batch[0:1] 

    # 2. Create the action sweep range
    actions = torch.linspace(-0.999, 0.999, 100).reshape(-1, 1).to(device)
    
    with torch.no_grad():
        # Get distribution parameters for that ONE state
        x = actor.net(single_obs)
        mu = actor.mu(x)
        log_std = torch.clamp(actor.log_std(x), -20, 2)
        std = torch.exp(log_std)
        
        base = Independent(Normal(mu, std), 1)
        dist = TransformedDistribution(base, [TanhTransform(cache_size=1)])
        
        # 3. Repeat the distribution parameters to match the 100 actions
        # This makes the batch_shape 100 so it matches the actions sweep
        mu_expanded = mu.expand(100, -1)
        std_expanded = std.expand(100, -1)
        
        base_exp = Independent(Normal(mu_expanded, std_expanded), 1)
        dist_exp = TransformedDistribution(base_exp, [TanhTransform(cache_size=1)])
        
        # 4. Now shapes match: [100, 1] vs [100, 1]
        log_probs = dist_exp.log_prob(actions)
        probs = log_probs.exp().cpu().numpy()

    plt.plot(actions.cpu().numpy(), probs)
    plt.title(f"Action PDF (mu={mu.item():.2f}, std={std.item():.2f})")
    plt.show()