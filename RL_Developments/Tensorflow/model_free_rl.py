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
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional
from .utils import get_device_strategy, create_optimizer

class QLearningAgent(tf.keras.Model):
    """Q-Learning agent implementation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [64, 64],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        **kwargs
    ):
        """Initialize Q-Learning agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for Q-network
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            **kwargs: Additional arguments
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Get device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create Q-network
            self.q_network = self._create_network(hidden_dims)
            self.optimizer = create_optimizer(learning_rate)

    def _create_network(self, hidden_dims: List[int]) -> tf.keras.Model:
        """Create Q-network.

        Args:
            hidden_dims: List of hidden layer dimensions

        Returns:
            Q-network model
        """
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = inputs

        # Hidden layers
        for dim in hidden_dims:
            x = tf.keras.layers.Dense(dim, activation='relu')(x)

        # Output Q-values for each action
        outputs = tf.keras.layers.Dense(self.action_dim)(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def get_action(self, state: tf.Tensor, training: bool = True) -> int:
        """Get action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        q_values = self.q_network(tf.expand_dims(state, 0), training=False)
        return int(tf.argmax(q_values[0]))

    def update(
        self,
        state: tf.Tensor,
        action: int,
        reward: float,
        next_state: tf.Tensor,
        done: bool
    ) -> Dict[str, float]:
        """Update Q-Learning agent.

        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
            done: Whether episode is done

        Returns:
            Dictionary of metrics
        """
        with tf.GradientTape() as tape:
            # Current Q-value
            current_q = self.q_network(tf.expand_dims(state, 0))[0, action]

            # Target Q-value
            next_q = self.q_network(tf.expand_dims(next_state, 0))
            max_next_q = tf.reduce_max(next_q)
            target = reward + (1 - float(done)) * self.gamma * max_next_q

            # Compute loss
            loss = tf.square(target - current_q)

        # Update network
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables)
        )


        # Update exploration rate
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return {
            'loss': float(loss),
            'epsilon': self.epsilon
        }

class SARSAgent(tf.keras.Model):
    """SARSA (State-Action-Reward-State-Action) agent implementation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [64, 64],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        **kwargs
    ):
        """Initialize SARSA agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for Q-network
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            **kwargs: Additional arguments
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Get device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create Q-network
            self.q_network = self._create_network(hidden_dims)
            self.optimizer = create_optimizer(learning_rate)

    def _create_network(self, hidden_dims: List[int]) -> tf.keras.Model:
        """Create Q-network.

        Args:
            hidden_dims: List of hidden layer dimensions

        Returns:
            Q-network model
        """
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = inputs

        # Hidden layers
        for dim in hidden_dims:
            x = tf.keras.layers.Dense(dim, activation='relu')(x)

        # Output Q-values for each action
        outputs = tf.keras.layers.Dense(self.action_dim)(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def get_action(self, state: tf.Tensor, training: bool = True) -> int:
        """Get action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        q_values = self.q_network(tf.expand_dims(state, 0), training=False)
        return int(tf.argmax(q_values[0]))

    def update(
        self,
        state: tf.Tensor,
        action: int,
        reward: float,
        next_state: tf.Tensor,
        next_action: int,
        done: bool
    ) -> Dict[str, float]:
        """Update SARSA agent.

        Args:
            state: Current state
            action: Current action
            reward: Received reward
            next_state: Next state
            next_action: Next action
            done: Whether episode is done

        Returns:
            Dictionary of metrics
        """
        with tf.GradientTape() as tape:
            # Current Q-value
            current_q = self.q_network(tf.expand_dims(state, 0))[0, action]

            # Target Q-value (using next state-action pair)
            next_q = self.q_network(tf.expand_dims(next_state, 0))[0, next_action]
            target = reward + (1 - float(done)) * self.gamma * next_q

            # Compute loss
            loss = tf.square(target - current_q)

        # Update network
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables)
        )

        # Update exploration rate
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return {
            'loss': float(loss),
            'epsilon': self.epsilon
        }
