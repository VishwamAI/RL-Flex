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
from typing import List, Tuple
import torch.nn as nn
import torch.optim as optim

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

# Define a simple value network for Q-function
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

class OfflineRL:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize policy and Q networks
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.q_network = QNetwork(state_dim, action_dim)

        # Set up optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def behavior_cloning_loss(self, states, actions):
        pred_actions = self.policy(states)
        return nn.MSELoss()(pred_actions, actions)

    def q_loss(self, states, actions, targets):
        q_values = self.q_network(states, actions)
        return nn.MSELoss()(q_values, targets)

    def update(self, states, actions, rewards, next_states, dones):
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Behavior Cloning update
        self.policy_optimizer.zero_grad()
        policy_loss = self.behavior_cloning_loss(states, actions)
        policy_loss.backward()
        self.policy_optimizer.step()

        # Q-value update
        self.q_optimizer.zero_grad()
        with torch.no_grad():
            next_actions = self.policy(next_states)
            target_q_values = rewards + 0.99 * self.q_network(next_states, next_actions) * (1 - dones)
        q_loss = self.q_loss(states, actions, target_q_values)
        q_loss.backward()
        self.q_optimizer.step()

        return policy_loss.item(), q_loss.item()

    def train(self, dataset: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]], epochs: int = 100):
        for epoch in range(epochs):
            # Sample data
            states, actions, rewards, next_states, dones = map(np.array, zip(*dataset))
            
            # Update step
            policy_loss, q_loss = self.update(states, actions, rewards, next_states, dones)
            
            print(f"Epoch {epoch+1}/{epochs}, Policy Loss: {policy_loss}, Q Loss: {q_loss}")

    def select_action(self, state: np.ndarray):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.policy(state).detach().numpy().flatten()
        return action