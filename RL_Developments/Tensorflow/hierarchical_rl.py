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

class HierarchicalRL(tf.keras.Model):
    """Hierarchical Reinforcement Learning implementation using the Options framework."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_options: int = 4,
        hidden_dims: List[int] = [256, 256],
        high_level_learning_rate: float = 1e-4,
        low_level_learning_rate: float = 3e-4,
        gamma: float = 0.99,
        option_epsilon: float = 0.01,
        termination_reg: float = 0.01,
        **kwargs
    ):
        """Initialize Hierarchical RL agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            num_options: Number of options (sub-policies)
            hidden_dims: List of hidden layer dimensions
            high_level_learning_rate: Learning rate for high-level policy
            low_level_learning_rate: Learning rate for low-level policies
            gamma: Discount factor
            option_epsilon: Exploration rate for option selection
            termination_reg: Regularization for option termination
            **kwargs: Additional arguments
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_options = num_options
        self.gamma = gamma
        self.option_epsilon = option_epsilon
        self.termination_reg = termination_reg

        # Get device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create high-level policy (option selector)
            self.high_level_policy = self._create_high_level_policy(hidden_dims)
            self.high_level_optimizer = create_optimizer(high_level_learning_rate)

            # Create low-level policies (one for each option)
            self.low_level_policies = [
                self._create_low_level_policy(hidden_dims)
                for _ in range(num_options)
            ]
            self.low_level_optimizers = [
                create_optimizer(low_level_learning_rate)
                for _ in range(num_options)
            ]

            # Create termination networks (one for each option)
            self.termination_networks = [
                self._create_termination_network(hidden_dims)
                for _ in range(num_options)
            ]
            self.termination_optimizers = [
                create_optimizer(low_level_learning_rate)
                for _ in range(num_options)
            ]

    def _create_high_level_policy(self, hidden_dims: List[int]) -> tf.keras.Model:
        """Create high-level policy network.

        Args:
            hidden_dims: List of hidden layer dimensions

        Returns:
            High-level policy model
        """
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = inputs

        # Hidden layers
        for dim in hidden_dims:
            x = tf.keras.layers.Dense(dim, activation='relu')(x)

        # Output option logits
        option_logits = tf.keras.layers.Dense(self.num_options)(x)

        return tf.keras.Model(inputs=inputs, outputs=option_logits)

    def _create_low_level_policy(self, hidden_dims: List[int]) -> tf.keras.Model:
        """Create low-level policy network.

        Args:
            hidden_dims: List of hidden layer dimensions

        Returns:
            Low-level policy model
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

    def _create_termination_network(self, hidden_dims: List[int]) -> tf.keras.Model:
        """Create termination network.

        Args:
            hidden_dims: List of hidden layer dimensions

        Returns:
            Termination network model
        """
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = inputs

        # Hidden layers
        for dim in hidden_dims:
            x = tf.keras.layers.Dense(dim, activation='relu')(x)

        # Output termination probability
        termination_prob = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        return tf.keras.Model(inputs=inputs, outputs=termination_prob)

    def get_option(
        self,
        state: tf.Tensor,
        deterministic: bool = False
    ) -> int:
        """Select an option using the high-level policy.

        Args:
            state: Current state
            deterministic: Whether to select option deterministically

        Returns:
            Selected option index
        """
        if not deterministic and np.random.random() < self.option_epsilon:
            return np.random.randint(self.num_options)

        option_logits = self.high_level_policy(tf.expand_dims(state, 0))[0]
        return int(tf.argmax(option_logits))

    def get_action(
        self,
        state: tf.Tensor,
        option: int,
        deterministic: bool = False
    ) -> tf.Tensor:
        """Get action from the selected option's policy.

        Args:
            state: Current state
            option: Current option index
            deterministic: Whether to select action deterministically

        Returns:
            Action tensor
        """
        mean, log_std = self.low_level_policies[option](
            tf.expand_dims(state, 0),
            training=False
        )

        if deterministic:
            return mean[0]

        std = tf.exp(log_std)
        normal = tfp.distributions.Normal(mean, std)
        action = normal.sample()
        return tf.clip_by_value(action[0], -1.0, 1.0)

    def should_terminate_option(
        self,
        state: tf.Tensor,
        option: int
    ) -> bool:
        """Determine if current option should terminate.

        Args:
            state: Current state
            option: Current option index

        Returns:
            Whether to terminate the current option
        """
        termination_prob = self.termination_networks[option](
            tf.expand_dims(state, 0)
        )[0, 0]
        return float(termination_prob) > 0.5

    def update(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        options: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor
    ) -> Dict[str, float]:
        """Update hierarchical agent.

        Args:
            states: Batch of states
            actions: Batch of actions
            options: Batch of options
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Update high-level policy
        with tf.GradientTape() as tape:
            option_logits = self.high_level_policy(states)
            high_level_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=options,
                    logits=option_logits
                )
            )

        # Update high-level policy
        high_level_gradients = tape.gradient(
            high_level_loss,
            self.high_level_policy.trainable_variables
        )
        self.high_level_optimizer.apply_gradients(
            zip(high_level_gradients, self.high_level_policy.trainable_variables)
        )

        # Update low-level policies and termination networks
        total_low_level_loss = 0
        total_termination_loss = 0

        for option_idx in range(self.num_options):
            # Get option-specific data
            option_mask = tf.cast(options == option_idx, tf.float32)
            if tf.reduce_sum(option_mask) == 0:
                continue

            # Update low-level policy
            with tf.GradientTape() as tape:
                mean, log_std = self.low_level_policies[option_idx](states)
                std = tf.exp(log_std)
                normal = tfp.distributions.Normal(mean, std)
                log_probs = tf.reduce_sum(
                    normal.log_prob(actions) - tf.math.log(1 - tf.tanh(actions)**2 + 1e-6),
                    axis=-1,
                    keepdims=True
                )
                low_level_loss = -tf.reduce_mean(log_probs * option_mask)

            low_level_gradients = tape.gradient(
                low_level_loss,
                self.low_level_policies[option_idx].trainable_variables
            )
            self.low_level_optimizers[option_idx].apply_gradients(
                zip(
                    low_level_gradients,
                    self.low_level_policies[option_idx].trainable_variables
                )
            )

            # Update termination network
            with tf.GradientTape() as tape:
                termination_probs = self.termination_networks[option_idx](states)
                next_termination_probs = self.termination_networks[option_idx](next_states)

                # Termination loss with regularization
                termination_loss = tf.reduce_mean(
                    -(1 - dones) * tf.math.log(1 - termination_probs + 1e-6) * option_mask -
                    dones * tf.math.log(termination_probs + 1e-6) * option_mask +
                    self.termination_reg * termination_probs * option_mask
                )

            termination_gradients = tape.gradient(
                termination_loss,
                self.termination_networks[option_idx].trainable_variables
            )
            self.termination_optimizers[option_idx].apply_gradients(
                zip(
                    termination_gradients,
                    self.termination_networks[option_idx].trainable_variables
                )
            )

            total_low_level_loss += float(low_level_loss)
            total_termination_loss += float(termination_loss)

        # Store metrics
        metrics.update({
            'high_level_loss': float(high_level_loss),
            'low_level_loss': total_low_level_loss / self.num_options,
            'option_termination_loss': total_termination_loss / self.num_options
        })

        return metrics

    def save_weights(self, filepath: str):
        """Save model weights.

        Args:
            filepath: Path to save weights
        """
        self.high_level_policy.save_weights(filepath + "_high_level")
        for i, policy in enumerate(self.low_level_policies):
            policy.save_weights(f"{filepath}_low_level_{i}")
        for i, network in enumerate(self.termination_networks):
            network.save_weights(f"{filepath}_termination_{i}")

    def load_weights(self, filepath: str):
        """Load model weights.

        Args:
            filepath: Path to load weights
        """
        self.high_level_policy.load_weights(filepath + "_high_level")
        for i, policy in enumerate(self.low_level_policies):
            policy.load_weights(f"{filepath}_low_level_{i}")
        for i, network in enumerate(self.termination_networks):
            network.load_weights(f"{filepath}_termination_{i}")
