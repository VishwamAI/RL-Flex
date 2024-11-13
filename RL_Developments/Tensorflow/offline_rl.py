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
from typing import Dict, List, Tuple, Optional
from .utils import get_device_strategy, create_optimizer, update_target_network

class OfflineRL(tf.keras.Model):
    """Offline Reinforcement Learning with Conservative Q-Learning (CQL)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        cql_alpha: float = 1.0,
        **kwargs
    ):
        """Initialize Offline RL agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            actor_learning_rate: Learning rate for actor network
            critic_learning_rate: Learning rate for critic network
            gamma: Discount factor
            tau: Target network update rate
            cql_alpha: CQL regularization coefficient
            **kwargs: Additional arguments
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.cql_alpha = cql_alpha

        # Get device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create actor network
            self.actor = self._create_actor(hidden_dims)
            self.actor_optimizer = create_optimizer(actor_learning_rate)

            # Create critic network and target
            self.critic = self._create_critic(hidden_dims)
            self.target_critic = self._create_critic(hidden_dims)
            self.critic_optimizer = create_optimizer(critic_learning_rate)

            # Initialize target network
            update_target_network(self.critic, self.target_critic, tau=1.0)

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
        state_inputs = tf.keras.Input(shape=(self.state_dim,))
        action_inputs = tf.keras.Input(shape=(self.action_dim,))
        x = tf.keras.layers.Concatenate()([state_inputs, action_inputs])

        # Hidden layers
        for dim in hidden_dims:
            x = tf.keras.layers.Dense(dim, activation='relu')(x)

        # Output Q-value
        q_value = tf.keras.layers.Dense(1)(x)

        return tf.keras.Model(inputs=[state_inputs, action_inputs], outputs=q_value)

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
        """Update Offline RL agent.

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
            # Current Q-values
            current_q = self.critic([states, actions])

            # Target Q-values
            next_actions = self.get_action(next_states[0], deterministic=True)
            next_actions = tf.repeat(
                tf.expand_dims(next_actions, 0),
                states.shape[0],
                axis=0
            )
            target_q = self.target_critic([next_states, next_actions])
            td_target = rewards + self.gamma * (1 - dones) * target_q

            # Standard TD error
            td_error = tf.square(current_q - td_target)

            # CQL regularization
            random_actions = tf.random.uniform(
                actions.shape,
                -1.0,
                1.0,
                dtype=tf.float32
            )
            random_q = self.critic([states, random_actions])
            cql_loss = tf.reduce_logsumexp(random_q) - current_q

            # Total critic loss
            critic_loss = tf.reduce_mean(td_error + self.cql_alpha * cql_loss)

        # Update critic
        critic_gradients = tape.gradient(
            critic_loss,
            self.critic.trainable_variables
        )
        self.critic_optimizer.apply_gradients(
            zip(critic_gradients, self.critic.trainable_variables)
        )

        # Update actor
        with tf.GradientTape() as tape:
            # Sample actions from current policy
            mean, log_std = self.actor(states)
            std = tf.exp(log_std)
            normal = tfp.distributions.Normal(mean, std)
            sampled_actions = normal.sample()
            sampled_actions = tf.clip_by_value(sampled_actions, -1.0, 1.0)

            # Compute actor loss
            q_values = self.critic([states, sampled_actions])
            actor_loss = -tf.reduce_mean(q_values)

        # Update actor
        actor_gradients = tape.gradient(
            actor_loss,
            self.actor.trainable_variables
        )
        self.actor_optimizer.apply_gradients(
            zip(actor_gradients, self.actor.trainable_variables)
        )

        # Update target network
        update_target_network(self.critic, self.target_critic, self.tau)

        # Store metrics
        metrics.update({
            'critic_loss': float(critic_loss),
            'cql_loss': float(tf.reduce_mean(cql_loss)),
            'actor_loss': float(actor_loss)
        })

        return metrics

    def save_weights(self, filepath: str):
        """Save model weights.

        Args:
            filepath: Path to save weights
        """
        self.actor.save_weights(filepath + "_actor")
        self.critic.save_weights(filepath + "_critic")
        self.target_critic.save_weights(filepath + "_target_critic")

    def load_weights(self, filepath: str):
        """Load model weights.

        Args:
            filepath: Path to load weights
        """
        self.actor.load_weights(filepath + "_actor")
        self.critic.load_weights(filepath + "_critic")
        self.target_critic.load_weights(filepath + "_target_critic")
