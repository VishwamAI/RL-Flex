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

class ImitationLearningModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ImitationLearningModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class BehavioralCloning:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3):
        self.model = ImitationLearningModel(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def get_action(self, state):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(state))
            return torch.argmax(logits, dim=-1).item()

    def update(self, states, actions):
        self.model.train()
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        logits = self.model(states)
        loss = self.criterion(logits, actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

class DAgger:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3):
        self.model = ImitationLearningModel(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def get_action(self, state):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(state))
            return torch.argmax(logits, dim=-1).item()

    def update(self, states, actions):
        self.model.train()
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        logits = self.model(states)
        loss = self.criterion(logits, actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, env, expert_policy, n_iterations, n_episodes):
        for iteration in range(n_iterations):
            states = []
            actions = []
            for _ in range(n_episodes):
                state = env.reset()
                done = False
                while not done:
                    action = self.get_action(state)
                    expert_action = expert_policy(state)
                    states.append(state)
                    actions.append(expert_action)
                    state, _, done, _ = env.step(action)
            loss = self.update(states, actions)
            print(f"Iteration {iteration + 1}/{n_iterations}, Loss: {loss}")