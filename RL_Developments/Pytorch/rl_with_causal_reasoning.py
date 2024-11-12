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

# Define the Policy Network with Causal Reasoning
class CausalPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, causal_attention_dim=64):
        super(CausalPolicyNetwork, self).__init__()
        self.causal_attention_dim = causal_attention_dim
        self.query = nn.Linear(state_dim, causal_attention_dim)
        self.key = nn.Linear(state_dim, causal_attention_dim)
        self.value = nn.Linear(state_dim, causal_attention_dim)
        self.fc1 = nn.Linear(state_dim + causal_attention_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.T) / np.sqrt(self.causal_attention_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        causal_context = torch.matmul(attention_weights, value)
        combined_input = torch.cat([x, causal_context], dim=-1)
        x = torch.relu(self.fc1(combined_input))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

# Define the Value Network with Causal Reasoning
class CausalValueNetwork(nn.Module):
    def __init__(self, state_dim, causal_attention_dim=64):
        super(CausalValueNetwork, self).__init__()
        self.causal_attention_dim = causal_attention_dim
        self.query = nn.Linear(state_dim, causal_attention_dim)
        self.key = nn.Linear(state_dim, causal_attention_dim)
        self.value = nn.Linear(state_dim, causal_attention_dim)
        self.fc1 = nn.Linear(state_dim + causal_attention_dim + 1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, action):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.T) / np.sqrt(self.causal_attention_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        causal_context = torch.matmul(attention_weights, value)
        combined_input = torch.cat([x, action, causal_context], dim=-1)
        x = torch.relu(self.fc1(combined_input))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

# RL Agent with Causal Reasoning
class RLWithCausalReasoning:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_network = CausalPolicyNetwork(state_dim, action_dim)
        self.value_network = CausalValueNetwork(state_dim)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = self.policy_network(state)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Update policy network
        self.policy_optimizer.zero_grad()
        action_probs = self.policy_network(states)
        log_probs = torch.log(action_probs[range(len(actions)), actions])
        values = self.value_network(states, actions.float())
        next_values = self.value_network(next_states, actions.float())
        advantages = rewards + 0.99 * next_values * (1 - dones) - values
        policy_loss = -torch.mean(log_probs * advantages.detach())
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update value network
        self.value_optimizer.zero_grad()
        values = self.value_network(states, actions.float())
        targets = rewards + 0.99 * next_values * (1 - dones)
        value_loss = torch.mean((targets.detach() - values) ** 2)
        value_loss.backward()
        self.value_optimizer.step()

        return policy_loss.item(), value_loss.item()

    def train(self, dataset, epochs=100):
        for epoch in range(epochs):
            states, actions, rewards, next_states, dones = zip(*dataset)
            policy_loss, value_loss = self.update(states, actions, rewards, next_states, dones)
            print(f"Epoch {epoch + 1}/{epochs}, Policy Loss: {policy_loss}, Value Loss: {value_loss}")

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = self.policy_network(state)
        action = torch.multinomial(action_probs, 1).item()
        return action