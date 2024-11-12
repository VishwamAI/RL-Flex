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
import logging
import time
from typing import Dict, Any, List, Tuple
from collections import deque
import random

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, state_shape: tuple, action_shape: tuple, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.batch_size = 32
        self.state_shape = state_shape
        self.action_shape = action_shape

    def add(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.buffer) < batch_size:
            return None

        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        states, actions, rewards, next_states, dones = zip(*samples)
        
        return {
            'observations': torch.FloatTensor(np.array(states)),
            'actions': torch.LongTensor(np.array(actions)),
            'rewards': torch.FloatTensor(np.array(rewards)),
            'next_observations': torch.FloatTensor(np.array(next_states)),
            'dones': torch.FloatTensor(np.array(dones))
        }

    def clear(self, fraction: float = 1.0):
        num_to_clear = int(len(self.buffer) * fraction)
        self.buffer = self.buffer[num_to_clear:]
        self.priorities = self.priorities[num_to_clear:]
        self.position = len(self.buffer)

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, features: List[int]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for f in features:
            layers.extend([
                nn.Linear(prev_dim, f),
                nn.ReLU()
            ])
            prev_dim = f
            
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class SelfCuringRLAgent:
    def __init__(self, 
                 features: List[int], 
                 action_dim: int,
                 state_dim: int,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 performance_threshold: float = 0.8,
                 update_interval: int = 86400,  # 24 hours in seconds
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.features = features
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.performance_threshold = performance_threshold
        self.update_interval = update_interval

        # Initialize network and optimizer
        self.q_network = QNetwork(state_dim, action_dim, features).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(100000, (state_dim,), (action_dim,))
        
        # Initialize training variables
        self.epsilon = self.epsilon_start
        self.is_trained = False
        self.performance = 0.0
        self.last_update = time.time()

    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def update(self, batch: Dict[str, torch.Tensor]) -> float:
        if batch is None:
            return 0.0

        states = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_observations'].to(self.device)
        dones = batch['dones'].to(self.device)

        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.q_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, env, num_episodes: int, max_steps: int) -> Dict[str, Any]:
        episode_rewards = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                action = self.select_action(state, training=True)
                next_state, reward, done, truncated, _ = env.step(action)
                
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                if len(self.replay_buffer) > self.replay_buffer.batch_size:
                    batch = self.replay_buffer.sample()
                    loss = self.update(batch)

                if done or truncated:
                    break

            episode_rewards.append(episode_reward)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            if (episode + 1) % 10 == 0:
                avg_reward = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
                logging.info(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.2f}")

        self.is_trained = True
        self.performance = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
        self.last_update = time.time()

        return {
            "final_reward": self.performance,
            "episode_rewards": episode_rewards
        }

    def diagnose(self) -> List[str]:
        issues = []
        if not self.is_trained:
            issues.append("Model is not trained")
        if self.performance < self.performance_threshold:
            issues.append("Model performance is below threshold")
        if time.time() - self.last_update > self.update_interval:
            issues.append("Model hasn't been updated in 24 hours")
        return issues

    def heal(self, env, num_episodes: int, max_steps: int):
        issues = self.diagnose()
        min_episodes = max(5, num_episodes // 10)
        
        for issue in issues:
            if issue == "Model is not trained" or issue == "Model performance is below threshold":
                logging.info(f"Healing issue: {issue}")
                self.train(env, num_episodes, max_steps)
            elif issue == "Model hasn't been updated in 24 hours":
                logging.info(f"Healing issue: {issue}")
                self.update_model(env, min_episodes, max_steps)

    def update_model(self, env, num_episodes: int, max_steps: int):
        num_episodes = max(1, num_episodes)
        training_info = self.train(env, num_episodes, max_steps)
        self.performance = training_info['final_reward']
        self.last_update = time.time()

class SelfCuringRL:
    def __init__(self, env, agent: SelfCuringRLAgent):
        self.env = env
        self.agent = agent
        self.logger = logging.getLogger(__name__)

    def train(self, num_episodes: int, max_steps: int) -> None:
        for episode in range(num_episodes):
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]  # Handle new gym API
            total_reward = 0

            for step in range(max_steps):
                action = self.agent.select_action(state, training=True)
                next_state, reward, done, info = self.env.step(action)
                if isinstance(info, dict) and 'truncated' in info:
                    done = done or info['truncated']  # Handle new gym API
                total_reward += reward

                self.agent.replay_buffer.add(state, action, reward, next_state, done)

                if len(self.agent.replay_buffer) > self.agent.replay_buffer.batch_size:
                    batch = self.agent.replay_buffer.sample()
                    loss = self.agent.update(batch)

                state = next_state

                if done:
                    break

            self.agent.epsilon = max(
                self.agent.epsilon_end,
                self.agent.epsilon * self.agent.epsilon_decay
            )

            if episode % 10 == 0:
                self.logger.info(
                    f"Episode {episode}, Total Reward: {total_reward}, "
                    f"Epsilon: {self.agent.epsilon:.4f}"
                )

            self._check_and_update_agent()

    def _check_and_update_agent(self) -> None:
        current_time = time.time()
        if current_time - self.agent.last_update >= self.agent.update_interval:
            self.agent.performance = self._evaluate_agent()
            if self.agent.performance < self.agent.performance_threshold:
                self._self_cure()
            self.agent.last_update = current_time

    def _evaluate_agent(self, num_eval_episodes: int = 10) -> float:
        total_rewards = []
        
        for _ in range(num_eval_episodes):
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]
            episode_reward = 0
            done = False

            while not done:
                action = self.agent.select_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                if isinstance(info, dict) and 'truncated' in info:
                    done = done or info['truncated']
                episode_reward += reward
                state = next_state

            total_rewards.append(episode_reward)

        return np.mean(total_rewards)

    def _self_cure(self) -> None:
        self.logger.warning("Performance below threshold. Initiating self-curing process.")
        
        # Reset epsilon for more exploration
        self.agent.epsilon = self.agent.epsilon_start
        
        # Reinitialize the Q-network
        state_dim = self.agent.q_network.network[0].in_features
        self.agent.q_network = QNetwork(
            state_dim, 
            self.agent.action_dim, 
            self.agent.features
        ).to(self.agent.device)
        self.agent.optimizer = optim.Adam(
            self.agent.q_network.parameters(),
            lr=self.agent.learning_rate
        )
        
        # Clear a portion of the replay buffer
        self.agent.replay_buffer.clear(fraction=0.5)
        
        self.logger.info("Self-curing process completed. Resuming training with reset parameters.")

    def run(self, num_episodes: int, max_steps: int) -> None:
        self.train(num_episodes, max_steps)
        final_performance = self._evaluate_agent()
        self.logger.info(f"Training completed. Final performance: {final_performance:.2f}")

def create_self_curing_rl_agent(features: List[int], action_dim: int, state_dim: int) -> SelfCuringRLAgent:
    return SelfCuringRLAgent(
        features=features,
        action_dim=action_dim,
        state_dim=state_dim
    )

if __name__ == "__main__":
    import gym
    
    logging.basicConfig(level=logging.INFO)
    env = gym.make("CartPole-v1")
    
    agent = create_self_curing_rl_agent(
        features=[64, 64],
        action_dim=env.action_space.n,
        state_dim=env.observation_space.shape[0]
    )

    # Initial training
    training_info = agent.train(env, num_episodes=1000, max_steps=500)
    logging.info(f"Initial training completed. Final reward: {training_info['final_reward']}")

    # Simulate some time passing and performance degradation
    agent.last_update -= 100000  # Simulate 27+ hours passing
    agent.performance = 0.7  # Simulate performance drop

    # Diagnose and heal
    issues = agent.diagnose()
    if issues:
        logging.info(f"Detected issues: {issues}")
        agent.heal(env, num_episodes=500, max_steps=500)
        logging.info(f"Healing completed. New performance: {agent.performance}")