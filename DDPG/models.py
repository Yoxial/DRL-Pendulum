import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, action_size),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Critic, self).__init__()
        self.state_net = nn.Sequential(
            nn.Linear(state_size, hidden_size//2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size//2)
        )
        
        self.action_net = nn.Sequential(
            nn.Linear(action_size, hidden_size//2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size//2)
        )
        
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, state, action):
        state_features = self.state_net(state)
        action_features = self.action_net(action)
        combined = torch.cat([state_features, action_features], dim=1)
        return self.combined_net(combined)

class DoubleCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DoubleCritic, self).__init__()
        self.critic1 = Critic(state_size, action_size, hidden_size)
        self.critic2 = Critic(state_size, action_size, hidden_size)
        
    def forward(self, state, action):
        return self.critic1(state, action), self.critic2(state, action)
