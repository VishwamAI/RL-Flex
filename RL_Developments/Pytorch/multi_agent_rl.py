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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action = self.tanh(self.fc3(x))
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, learning_rate=1e-3):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.gamma = 0.99

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state)
        return action.detach().numpy().flatten()

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Update critic
        next_actions = self.actor(next_states)
        next_q_values = self.critic(next_states, next_actions).squeeze()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        q_values = self.critic(states, actions).squeeze()
        critic_loss = nn.MSELoss()(q_values, target_q_values.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()

class MultiAgentEnvironment:
    def __init__(self, num_agents, state_dim, action_dim):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

    def reset(self):
        return [np.random.rand(self.state_dim) for _ in range(self.num_agents)]

    def step(self, actions):
        next_states = [np.random.rand(self.state_dim) for _ in range(self.num_agents)]
        rewards = [np.random.rand() for _ in range(self.num_agents)]
        dones = [False for _ in range(self.num_agents)]
        return next_states, rewards, dones, {}

def train_maddpg(num_agents, state_dim, action_dim, num_episodes=1000):
    env = MultiAgentEnvironment(num_agents, state_dim, action_dim)
    agents = [MADDPGAgent(state_dim, action_dim) for _ in range(num_agents)]

    for episode in range(num_episodes):
        states = env.reset()
        episode_reward = 0

        while True:
            actions = [agent.act(state) for agent, state in zip(agents, states)]
            next_states, rewards, dones, _ = env.step(actions)
            episode_reward += sum(rewards)

            for i, agent in enumerate(agents):
                agent.update([states[i]], [actions[i]], [rewards[i]], [next_states[i]], [dones[i]])

            states = next_states

            if any(dones):
                break

        if episode % 100 == 0:
            print(f"Episode {episode}, Average Reward: {episode_reward / num_agents:.2f}")

    return agents

def inter_agent_communication(agents, messages):
    for i, agent in enumerate(agents):
        received_messages = [msg for j, msg in enumerate(messages) if j != i]
        # Process received messages (placeholder)
        pass  # In a real scenario, add agent message handling logic here

class MultiAgentRL:
    def __init__(self, num_agents, state_dim, action_dim):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agents = [MADDPGAgent(state_dim, action_dim) for _ in range(num_agents)]

    def train(self, env, num_episodes=1000):
        for episode in range(num_episodes):
            states = env.reset()
            episode_reward = 0

            while True:
                actions = [agent.act(state) for agent, state in zip(self.agents, states)]
                next_states, rewards, dones, _ = env.step(actions)
                episode_reward += sum(rewards)

                for i, agent in enumerate(self.agents):
                    agent.update([states[i]], [actions[i]], [rewards[i]], [next_states[i]], [dones[i]])

                states = next_states

                if any(dones):
                    break

            if episode % 100 == 0:
                print(f"Episode {episode}, Average Reward: {episode_reward / self.num_agents:.2f}")

        return self.agents

    def communicate(self, messages):
        for i, agent in enumerate(self.agents):
            received_messages = [msg for j, msg in enumerate(messages) if j != i]
            # Process received messages (placeholder)
            # In a real scenario, add agent message handling logic here

    def act(self, states):
        return [agent.act(state) for agent, state in zip(self.agents, states)]

def main():
    num_agents = 3
    state_dim = 5
    action_dim = 2

    trained_agents = train_maddpg(num_agents, state_dim, action_dim)

    # Simulate inter-agent communication
    messages = [f"Message from Agent {i}" for i in range(num_agents)]
    inter_agent_communication(trained_agents, messages)

if __name__ == "__main__":
    main()