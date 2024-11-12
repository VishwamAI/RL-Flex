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
import numpy as np
import gym
import math

# Define the Environment Model in PyTorch
class EnvironmentModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

    def update(self, states, actions, next_states):
        self.optimizer.zero_grad()
        pred_next_states = self(states, actions)
        loss = nn.MSELoss()(pred_next_states, next_states)
        loss.backward()
        self.optimizer.step()
        return loss.item()

# Define the Actor and Critic Networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # For bounded actions
        )

    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

# Define the MBPO Agent
class MBPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.env_model = EnvironmentModel(state_dim, action_dim, hidden_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action = self.actor(state)
        return action.cpu().numpy()

    def update(self, states, actions, rewards, next_states, dones):
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Update Critic
        self.critic_optimizer.zero_grad()
        current_Q = self.critic(states, actions)
        next_actions = self.actor(next_states)
        next_Q = self.critic(next_states, next_actions)
        target_Q = rewards + (1.0 - dones) * 0.99 * next_Q
        critic_loss = nn.MSELoss()(current_Q, target_Q.detach())
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss = -self.critic(states, self.actor(states)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self, exploration_constant):
        return max(self.children, key=lambda c: c.ucb_score(exploration_constant))

    def ucb_score(self, exploration_constant):
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)

    def expand(self, action, next_state):
        child = MCTSNode(next_state, parent=self, action=action)
        self.children.append(child)
        return child

    def backpropagate(self, value):
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.backpropagate(value)

    def best_child(self):
        return max(self.children, key=lambda c: c.visits)

def train_agent(env, agent, num_episodes=1000, planning_horizon=5, num_simulations=10):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Update agent
            agent.update(
                np.array([state]),
                np.array([action]),
                np.array([reward]),
                np.array([next_state]),
                np.array([float(done)])
            )

            # Update environment model
            agent.env_model.update(
                torch.FloatTensor([state]).to(agent.device),
                torch.FloatTensor([action]).to(agent.device),
                torch.FloatTensor([next_state]).to(agent.device)
            )

            state = next_state

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}")

    return agent

def main():
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = MBPOAgent(state_dim, action_dim)
    trained_agent = train_agent(env, agent)

    # Test the trained agent
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = trained_agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Test episode reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()