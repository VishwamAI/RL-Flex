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
import torch.nn as nn
import torch.optim as optim

class Environment:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = torch.zeros(state_dim)
    
    def reset(self):
        self.state = torch.randn(self.state_dim)
        return self.state

    def step(self, action):
        noise = torch.randn(self.state_dim) * 0.1
        self.state = torch.clamp(self.state + action + noise, -1, 1)
        reward = -torch.norm(self.state)
        done = torch.norm(self.state) < 0.1
        return self.state, reward.item(), done.item(), {}

class Agent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

def train_agent(env, state_dim, action_dim, num_episodes, max_steps, batch_size, learning_rate):
    agent = Agent(state_dim, action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        states = []
        actions = []
        rewards = []

        for _ in range(max_steps):
            state_input = state.unsqueeze(0)
            action = agent(state_input).squeeze(0)
            next_state, reward, done, _ = env.step(action.detach())
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            episode_reward += reward

            if done:
                break

        if len(states) >= batch_size:
            batch_states = torch.stack(states[-batch_size:])
            batch_rewards = torch.tensor(rewards[-batch_size:])

            predicted_actions = agent(batch_states)
            loss = -torch.mean(batch_rewards * predicted_actions.sum(dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")

    return agent

# Usage
state_dim = 4
action_dim = 2
env = Environment(state_dim, action_dim)
trained_agent = train_agent(env, state_dim, action_dim, num_episodes=1000, max_steps=200, batch_size=32, learning_rate=1e-3)