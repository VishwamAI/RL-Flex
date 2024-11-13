import tensorflow as tf
import numpy as np
from typing import Dict, Tuple
from .utils import get_device_strategy, create_optimizer

class QLearningAgent:
    """Tabular Q-Learning Agent implementation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        """Initialize Q-Learning Agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize Q-table
        self.q_table = {}

    def get_action(self, state: tf.Tensor) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        state_key = tuple(state.numpy().flatten())

        # Initialize state if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)

        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            return int(np.argmax(self.q_table[state_key]))

    def update(
        self,
        state: tf.Tensor,
        action: int,
        reward: float,
        next_state: tf.Tensor,
        done: bool
    ) -> Dict[str, float]:
        """Update Q-table.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Terminal flag

        Returns:
            Dictionary of metrics
        """
        state_key = tuple(state.numpy().flatten())
        next_state_key = tuple(next_state.numpy().flatten())

        # Initialize states if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_dim)

        # Q-learning update
        current_q = self.q_table[state_key][action]
        next_q = np.max(self.q_table[next_state_key]) if not done else 0
        target_q = reward + self.gamma * next_q
        td_error = target_q - current_q

        # Update Q-value
        self.q_table[state_key][action] += self.learning_rate * td_error

        return {
            "td_error": float(td_error),
            "q_value": float(current_q)
        }

class SARSAgent:
    """SARSA (State-Action-Reward-State-Action) Agent implementation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        """Initialize SARSA Agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize Q-table
        self.q_table = {}

    def get_action(self, state: tf.Tensor) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        state_key = tuple(state.numpy().flatten())

        # Initialize state if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)

        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            return int(np.argmax(self.q_table[state_key]))

    def update(
        self,
        state: tf.Tensor,
        action: int,
        reward: float,
        next_state: tf.Tensor,
        next_action: int,
        done: bool
    ) -> Dict[str, float]:
        """Update Q-table.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action
            done: Terminal flag

        Returns:
            Dictionary of metrics
        """
        state_key = tuple(state.numpy().flatten())
        next_state_key = tuple(next_state.numpy().flatten())

        # Initialize states if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_dim)

        # SARSA update
        current_q = self.q_table[state_key][action]
        next_q = self.q_table[next_state_key][next_action] if not done else 0
        target_q = reward + self.gamma * next_q
        td_error = target_q - current_q

        # Update Q-value
        self.q_table[state_key][action] += self.learning_rate * td_error

        return {
            "td_error": float(td_error),
            "q_value": float(current_q)
        }
