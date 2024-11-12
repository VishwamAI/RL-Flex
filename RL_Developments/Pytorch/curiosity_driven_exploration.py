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
import gym

import torch.nn as nn
import torch.optim as optim

class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, beta=0.2):
        super(ICM, self).__init__()
        self.beta = beta

        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, state, next_state, action):
        state_feat = self.feature_encoder(state)
        next_state_feat = self.feature_encoder(next_state)

        action_one_hot = nn.functional.one_hot(action, num_classes=self.action_dim).float()

        pred_next_state_feat = self.forward_model(torch.cat([state_feat, action_one_hot], dim=1))
        pred_action = self.inverse_model(torch.cat([state_feat, next_state_feat], dim=1))

        return pred_action, pred_next_state_feat, next_state_feat

    def compute_intrinsic_reward(self, state, next_state, action):
        _, pred_next_state_feat, next_state_feat = self.forward(state, next_state, action)
        intrinsic_reward = self.beta * 0.5 * ((pred_next_state_feat - next_state_feat) ** 2).mean(dim=1)
        return intrinsic_reward

class NoveltyDetector:
    def __init__(self, state_dim, memory_size=1000, novelty_threshold=0.1):
        self.memory = np.zeros((memory_size, state_dim))
        self.memory_index = 0
        self.memory_size = memory_size
        self.novelty_threshold = novelty_threshold

    def compute_novelty(self, state):
        if self.memory_index < self.memory_size:
            distances = np.mean(np.abs(self.memory[:self.memory_index] - state.cpu().numpy()), axis=1)
        else:
            distances = np.mean(np.abs(self.memory - state.cpu().numpy()), axis=1)
        novelty = np.min(distances)
        return novelty

    def update_memory(self, state):
        self.memory[self.memory_index % self.memory_size] = state.cpu().numpy()
        self.memory_index += 1

    def is_novel(self, state):
        novelty = self.compute_novelty(state)
        return novelty > self.novelty_threshold

class CuriosityDrivenAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, learning_rate=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.icm = ICM(state_dim, action_dim, hidden_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=learning_rate)

        self.novelty_detector = NoveltyDetector(state_dim)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs = self.actor(state)
        action = torch.argmax(action_probs, dim=1)
        action_one_hot = nn.functional.one_hot(action, num_classes=self.action_dim).float()
        return action.item(), action_one_hot.squeeze(0)

    def update(self, state, action_one_hot, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action = torch.argmax(action_one_hot).unsqueeze(0)

        # ICM update
        pred_action, pred_next_state_feat, next_state_feat = self.icm(state, next_state, action)
        inverse_loss = nn.functional.cross_entropy(pred_action, action)
        forward_loss = 0.5 * ((pred_next_state_feat - next_state_feat) ** 2).mean()
        icm_loss = inverse_loss + forward_loss

        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_optimizer.step()

        self.novelty_detector.update_memory(next_state.squeeze(0))

        # Critic update
        value = self.critic(state)
        next_value = self.critic(next_state)
        td_error = reward + (1 - done) * 0.99 * next_value - value
        critic_loss = td_error.pow(2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        self.actor_optimizer.zero_grad()
        actor_loss = -(td_error.detach() * self.actor(state).gather(1, action.unsqueeze(1))).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        return icm_loss.item(), critic_loss.item()

def train_curiosity_agent(env, agent, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, action_one_hot = agent.act(state)
            next_state, extrinsic_reward, done, _ = env.step(action)
            total_reward += extrinsic_reward

            icm_loss, critic_loss = agent.update(state, action_one_hot, extrinsic_reward, next_state, done)
            state = next_state

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}, ICM Loss: {icm_loss:.4f}, Critic Loss: {critic_loss:.4f}")

def main():
    env = gym.make('MountainCar-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = CuriosityDrivenAgent(state_dim, action_dim)
    train_curiosity_agent(env, agent)

if __name__ == "__main__":
    main()