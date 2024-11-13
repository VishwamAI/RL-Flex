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

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional
from .utils import get_device_strategy, create_optimizer

class BehavioralCloning(tf.keras.Model):
    """Behavioral Cloning implementation for imitation learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
        **kwargs
    ):
        """Initialize BehavioralCloning.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate
            **kwargs: Additional arguments
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Get device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create policy network
            self.policy_network = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    hidden_dims[0],
                    activation='relu',
                    input_shape=(state_dim,)
                ),
                tf.keras.layers.Dense(
                    hidden_dims[1],
                    activation='relu'
                ),
                tf.keras.layers.Dense(action_dim)
            ])

            # Create optimizer
            self.optimizer = create_optimizer(learning_rate)

    def call(
        self,
        states: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        """Forward pass through the network.

        Args:
            states: Batch of states
            training: Whether in training mode

        Returns:
            Predicted actions
        """
        return self.policy_network(states, training=training)

    def update(
        self,
        expert_states: tf.Tensor,
        expert_actions: tf.Tensor
    ) -> Dict[str, float]:
        """Update policy using behavioral cloning.

        Args:
            expert_states: Batch of expert states
            expert_actions: Batch of expert actions

        Returns:
            Dictionary of metrics
        """
        with tf.GradientTape() as tape:
            predicted_actions = self(expert_states, training=True)
            loss = tf.reduce_mean(tf.square(predicted_actions - expert_actions))

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": float(loss)}

class DAgger(tf.keras.Model):
    """Dataset Aggregation (DAgger) implementation for imitation learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
        beta: float = 0.9,
        **kwargs
    ):
        """Initialize DAgger.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate
            beta: Mixture coefficient for expert policy
            **kwargs: Additional arguments
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.beta = beta

        # Get device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create policy network
            self.policy_network = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    hidden_dims[0],
                    activation='relu',
                    input_shape=(state_dim,)
                ),
                tf.keras.layers.Dense(
                    hidden_dims[1],
                    activation='relu'
                ),
                tf.keras.layers.Dense(action_dim)
            ])

            # Create optimizer
            self.optimizer = create_optimizer(learning_rate)

    def call(
        self,
        states: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        """Forward pass through the network.

        Args:
            states: Batch of states
            training: Whether in training mode

        Returns:
            Predicted actions
        """
        return self.policy_network(states, training=training)

    def get_action(
        self,
        states: tf.Tensor,
        expert_actions: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> tf.Tensor:
        """Get actions using DAgger policy.

        Args:
            states: Batch of states
            expert_actions: Optional batch of expert actions
            training: Whether in training mode

        Returns:
            Selected actions
        """
        predicted_actions = self(states, training=training)

        if training and expert_actions is not None:
            # During training, randomly choose between expert and learned policy
            use_expert = tf.random.uniform(()) < self.beta
            actions = tf.cond(
                use_expert,
                lambda: expert_actions,
                lambda: predicted_actions
            )
            return actions
        else:
            return predicted_actions

    def update(
        self,
        states: tf.Tensor,
        expert_actions: tf.Tensor
    ) -> Dict[str, float]:
        """Update policy using DAgger.

        Args:
            states: Batch of states
            expert_actions: Batch of expert actions

        Returns:
            Dictionary of metrics
        """
        with tf.GradientTape() as tape:
            predicted_actions = self(states, training=True)
            loss = tf.reduce_mean(tf.square(predicted_actions - expert_actions))

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {
            "loss": float(loss),
            "beta": float(self.beta)
        }

    def update_beta(self, decay: float = 0.95):
        """Update beta parameter for expert policy mixture.

        Args:
            decay: Decay factor for beta
        """
        self.beta *= decay
