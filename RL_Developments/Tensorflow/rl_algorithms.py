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
import tensorflow_probability as tfp
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .utils import get_device_strategy, create_optimizer

class QLearning(tf.keras.Model):
    """Q-Learning implementation with neural network function approximation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
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
            self.q_network = self._create_q_network(hidden_dims)
            self.optimizer = create_optimizer(learning_rate)

    def _create_q_network(self, hidden_dims: List[int]) -> tf.keras.Model:
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
        q_values = tf.keras.layers.Dense(self.action_dim)(x)

        return tf.keras.Model(inputs=inputs, outputs=q_values)

    def get_action(self, state: tf.Tensor) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected action index
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        q_values = self.q_network(tf.expand_dims(state, 0), training=False)
        return int(tf.argmax(q_values[0]))

    def update(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor
    ) -> Dict[str, float]:
        """Update Q-Learning agent.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        with tf.GradientTape() as tape:
            # Current Q-values
            current_q = self.q_network(states)
            current_q_selected = tf.reduce_sum(
                current_q * tf.one_hot(actions, self.action_dim),
                axis=1,
                keepdims=True
            )

            # Target Q-values
            next_q = self.q_network(next_states)
            next_q_max = tf.reduce_max(next_q, axis=1, keepdims=True)
            target_q = rewards + self.gamma * (1 - dones) * next_q_max

            # Q-learning loss
            q_loss = tf.reduce_mean(tf.square(target_q - current_q_selected))

        # Update Q-network
        gradients = tape.gradient(q_loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables)
        )

        # Store metrics
        metrics.update({
            'q_loss': float(q_loss),
            'mean_q_value': float(tf.reduce_mean(current_q))
        })

        return metrics

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

class ActorCritic(tf.keras.Model):
    """Actor-Critic implementation with continuous action space."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 3e-4,
        gamma: float = 0.99,
        **kwargs
    ):
        """Initialize Actor-Critic agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            actor_learning_rate: Learning rate for actor network
            critic_learning_rate: Learning rate for critic network
            gamma: Discount factor
            **kwargs: Additional arguments
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Get device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create actor network
            self.actor = self._create_actor(hidden_dims)
            self.actor_optimizer = create_optimizer(actor_learning_rate)

            # Create critic network
            self.critic = self._create_critic(hidden_dims)
            self.critic_optimizer = create_optimizer(critic_learning_rate)

    def _create_actor(self, hidden_dims: List[int]) -> tf.keras.Model:
        """Create actor network.

        Args:
            hidden_dims: List of hidden layer dimensions

        Returns:
            Actor network model
        """
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = inputs

        # Hidden layers
        for dim in hidden_dims:
            x = tf.keras.layers.Dense(dim, activation='relu')(x)

        # Output mean and log std
        mean = tf.keras.layers.Dense(self.action_dim, activation='tanh')(x)
        log_std = tf.keras.layers.Dense(self.action_dim)(x)
        log_std = tf.clip_by_value(log_std, -20, 2)

        return tf.keras.Model(inputs=inputs, outputs=[mean, log_std])

    def _create_critic(self, hidden_dims: List[int]) -> tf.keras.Model:
        """Create critic network.

        Args:
            hidden_dims: List of hidden layer dimensions

        Returns:
            Critic network model
        """
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = inputs

        # Hidden layers
        for dim in hidden_dims:
            x = tf.keras.layers.Dense(dim, activation='relu')(x)

        # Output state value
        value = tf.keras.layers.Dense(1)(x)

        return tf.keras.Model(inputs=inputs, outputs=value)

    def get_action(
        self,
        state: tf.Tensor,
        deterministic: bool = False
    ) -> tf.Tensor:
        """Sample action from policy.

        Args:
            state: Current state
            deterministic: Whether to return deterministic action

        Returns:
            Action tensor
        """
        mean, log_std = self.actor(tf.expand_dims(state, 0), training=False)

        if deterministic:
            return mean[0]

        std = tf.exp(log_std)
        normal = tfp.distributions.Normal(mean, std)
        action = normal.sample()
        return tf.clip_by_value(action[0], -1.0, 1.0)

    def update(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor
    ) -> Dict[str, float]:
        """Update Actor-Critic agent.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Update critic
        with tf.GradientTape() as tape:
            values = self.critic(states)
            next_values = self.critic(next_states)
            target_values = rewards + self.gamma * (1 - dones) * next_values
            critic_loss = tf.reduce_mean(tf.square(target_values - values))

        critic_gradients = tape.gradient(
            critic_loss,
            self.critic.trainable_variables
        )
        self.critic_optimizer.apply_gradients(
            zip(critic_gradients, self.critic.trainable_variables)
        )

        # Update actor
        with tf.GradientTape() as tape:
            mean, log_std = self.actor(states)
            std = tf.exp(log_std)
            normal = tfp.distributions.Normal(mean, std)
            log_probs = tf.reduce_sum(
                normal.log_prob(actions),
                axis=-1,
                keepdims=True
            )
            advantages = target_values - values
            actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))

        actor_gradients = tape.gradient(
            actor_loss,
            self.actor.trainable_variables
        )
        self.actor_optimizer.apply_gradients(
            zip(actor_gradients, self.actor.trainable_variables)
        )

        # Store metrics
        metrics.update({
            'actor_loss': float(actor_loss),
            'critic_loss': float(critic_loss),
            'value_estimate': float(tf.reduce_mean(values))
        })

        return metrics

    def save_weights(self, filepath: str):
        """Save model weights.

        Args:
            filepath: Path to save weights
        """
        self.actor.save_weights(filepath + "_actor")
        self.critic.save_weights(filepath + "_critic")

    def load_weights(self, filepath: str):
        """Load model weights.

        Args:
            filepath: Path to load weights
        """
        self.actor.load_weights(filepath + "_actor")
        self.critic.load_weights(filepath + "_critic")

class PolicyGradient(tf.keras.Model):
    """Policy Gradient implementation with continuous action space."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        **kwargs
    ):
        """Initialize Policy Gradient agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for policy network
            gamma: Discount factor
            **kwargs: Additional arguments
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Get device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create policy network
            self.policy_network = self._create_policy_network(hidden_dims)
            self.optimizer = create_optimizer(learning_rate)

    def _create_policy_network(self, hidden_dims: List[int]) -> tf.keras.Model:
        """Create policy network.

        Args:
            hidden_dims: List of hidden layer dimensions

        Returns:
            Policy network model
        """
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = inputs

        # Hidden layers
        for dim in hidden_dims:
            x = tf.keras.layers.Dense(dim, activation='relu')(x)

        # Output mean and log std
        mean = tf.keras.layers.Dense(self.action_dim, activation='tanh')(x)
        log_std = tf.keras.layers.Dense(self.action_dim)(x)
        log_std = tf.clip_by_value(log_std, -20, 2)

        return tf.keras.Model(inputs=inputs, outputs=[mean, log_std])

    def get_action(self, state: tf.Tensor) -> tf.Tensor:
        """Sample action from policy.

        Args:
            state: Current state

        Returns:
            Action tensor
        """
        mean, log_std = self.policy_network(tf.expand_dims(state, 0), training=False)
        std = tf.exp(log_std)
        normal = tfp.distributions.Normal(mean, std)
        action = normal.sample()
        return tf.clip_by_value(action[0], -1.0, 1.0)

    def update(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor
    ) -> Dict[str, float]:
        """Update Policy Gradient agent.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        with tf.GradientTape() as tape:
            mean, log_std = self.policy_network(states)
            std = tf.exp(log_std)
            normal = tfp.distributions.Normal(mean, std)
            log_probs = tf.reduce_sum(
                normal.log_prob(actions),
                axis=-1,
                keepdims=True
            )
            entropy = tf.reduce_mean(normal.entropy())
            policy_loss = -tf.reduce_mean(log_probs * rewards)

        gradients = tape.gradient(
            policy_loss,
            self.policy_network.trainable_variables
        )
        self.optimizer.apply_gradients(
            zip(gradients, self.policy_network.trainable_variables)
        )

        # Store metrics
        metrics.update({
            'policy_loss': float(policy_loss),
            'entropy': float(entropy)
        })

        return metrics
