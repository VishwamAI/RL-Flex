# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3):
        self.network = DQNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def get_action(self, state):
        with torch.no_grad():
            q_values = self.network(torch.FloatTensor(state))
        return torch.argmax(q_values).item()
    
    def update(self, state, action, reward, next_state, done):
        q_values = self.network(torch.FloatTensor(state))
        next_q_values = self.network(torch.FloatTensor(next_state))
        target = reward + (1 - done) * 0.99 * torch.max(next_q_values).item()
        target_f = q_values.clone()
        target_f[action] = target
        loss = self.criterion(q_values, target_f.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor_fc1 = nn.Linear(state_dim, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_out = nn.Linear(64, action_dim)
        
        self.critic_fc1 = nn.Linear(state_dim, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_out = nn.Linear(64, 1)
    
    def forward(self, x):
        actor_x = torch.relu(self.actor_fc1(x))
        actor_x = torch.relu(self.actor_fc2(actor_x))
        action_probs = torch.softmax(self.actor_out(actor_x), dim=-1)
        
        critic_x = torch.relu(self.critic_fc1(x))
        critic_x = torch.relu(self.critic_fc2(critic_x))
        value = self.critic_out(critic_x)
        
        return action_probs, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, learning_rate=3e-4):
        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.eps_clip = 0.2
    
    def get_action(self, state):
        state = torch.FloatTensor(state)
        action_probs, _ = self.network(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()
    
    def update(self, states, actions, rewards, dones, old_log_probs):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        old_log_probs = torch.FloatTensor(old_log_probs)
        
        action_probs, values = self.network(states)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        
        advantages = rewards - values.detach().squeeze()
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        critic_loss = nn.MSELoss()(values.squeeze(), rewards)
        entropy = dist.entropy().mean()
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()