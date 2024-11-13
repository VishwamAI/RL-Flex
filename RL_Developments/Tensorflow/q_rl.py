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
from typing import Dict, List, Tuple, Any, Optional
from .utils import get_device_strategy, create_optimizer

class QuantumLayer(tf.keras.layers.Layer):
    """Quantum-inspired neural network layer."""

    def __init__(
        self,
        units: int,
        activation: str = 'tanh',
        **kwargs
    ):
        """Initialize quantum layer.

        Args:
            units: Number of output units
            activation: Activation function
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.units = units
        self.activation_fn = tf.keras.activations.get(activation)

        # Initialize quantum-inspired parameters
        self.phase = self.add_weight(
            'phase',
            shape=[units],
            initializer='random_normal',
            trainable=True
        )
        self.amplitude = self.add_weight(
            'amplitude',
            shape=[units],
            initializer='random_normal',
            trainable=True
        )

    def build(self, input_shape):
        """Build layer.

        Args:
            input_shape: Shape of input tensor
        """
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            'kernel',
            shape=[input_dim, self.units],
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            'bias',
            shape=[self.units],
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        """Forward pass.

        Args:
            inputs: Input tensor

        Returns:
            Output tensor
        """
        # Quantum-inspired transformation
        quantum_state = tf.matmul(inputs, self.kernel) + self.bias
        quantum_phase = tf.complex(
            tf.cos(self.phase),
            tf.sin(self.phase)
        )
        quantum_amplitude = tf.complex(
            tf.cos(self.amplitude),
            tf.sin(self.amplitude)
        )
        quantum_output = tf.abs(quantum_state * quantum_phase * quantum_amplitude)
        return self.activation_fn(quantum_output)

class QuantumPolicyNetwork(tf.keras.Model):
    """Quantum-inspired policy network."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        """Initialize quantum policy network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Create quantum policy network
        self.quantum_layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            self.quantum_layers.append(QuantumLayer(dim))
            prev_dim = dim

        self.output_layer = tf.keras.layers.Dense(
            action_dim * 2,  # Mean and log_std
            activation=None
        )

    def call(self, inputs, training=False):
        """Forward pass.

        Args:
            inputs: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """
        x = inputs
        for layer in self.quantum_layers:
            x = layer(x, training=training)
        return self.output_layer(x, training=training)

class QuantumValueNetwork(tf.keras.Model):
    """Quantum-inspired value network."""

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        """Initialize quantum value network.

        Args:
            state_dim: Dimension of state space
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        self.state_dim = state_dim

        # Create quantum value network
        self.quantum_layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            self.quantum_layers.append(QuantumLayer(dim))
            prev_dim = dim

        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs, training=False):
        """Forward pass.

        Args:
            inputs: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """
        x = inputs
        for layer in self.quantum_layers:
            x = layer(x, training=training)
        return self.output_layer(x, training=training)

class QRLAgent(tf.keras.Model):
    """Quantum-inspired Reinforcement Learning Agent.

    This class implements quantum-inspired reinforcement learning
    with quantum-inspired neural networks for both policy and value
    functions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
        **kwargs
    ):
        """Initialize QRLAgent.

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
            # Create quantum networks
            self.policy = QuantumPolicyNetwork(
                state_dim,
                action_dim,
                hidden_dims
            )
            self.value = QuantumValueNetwork(state_dim, hidden_dims)

            # Create optimizers
            self.policy_optimizer = create_optimizer(learning_rate)
            self.value_optimizer = create_optimizer(learning_rate)

    def get_action(
        self,
        states: tf.Tensor,
        training: bool = False
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Get actions from policy network.

        Args:
            states: Batch of states
            training: Whether in training mode

        Returns:
            Tuple of (actions, additional info)
        """
        outputs = self.policy(states, training=training)
        means, log_stds = tf.split(outputs, 2, axis=-1)
        stds = tf.exp(tf.clip_by_value(log_stds, -20, 2))

        # Sample actions using quantum-inspired noise
        if training:
            phase = tf.random.uniform(tf.shape(means), 0, 2 * np.pi)
            quantum_noise = tf.cos(phase) + tf.sin(phase)
            actions = means + quantum_noise * stds
        else:
            actions = means

        actions = tf.clip_by_value(actions, -1, 1)

        return actions, {
            "means": means,
            "log_stds": log_stds,
            "stds": stds
        }

    def update(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor
    ) -> Dict[str, float]:
        """Update networks using quantum-inspired optimization.

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

        # Update policy
        with tf.GradientTape() as tape:
            actions, action_info = self.get_action(states, training=True)
            values = self.value(states, training=True)
            next_values = self.value(next_states, training=True)

            # Compute quantum-inspired advantages
            phase = tf.random.uniform(tf.shape(values), 0, 2 * np.pi)
            quantum_factor = tf.abs(tf.cos(phase) + tf.complex(0., 1.) * tf.sin(phase))
            advantages = (
                rewards +
                0.99 * (1 - dones) * next_values -
                values
            ) * quantum_factor

            # Compute policy loss
            means = action_info["means"]
            stds = action_info["stds"]
            log_probs = -0.5 * (
                tf.square((actions - means) / stds) +
                2 * tf.math.log(stds) +
                tf.math.log(2 * np.pi)
            )
            log_probs = tf.reduce_sum(log_probs, axis=-1)
            policy_loss = -tf.reduce_mean(log_probs * advantages)

        policy_grads = tape.gradient(
            policy_loss,
            self.policy.trainable_variables
        )
        self.policy_optimizer.apply_gradients(
            zip(policy_grads, self.policy.trainable_variables)
        )

        metrics["policy_loss"] = float(policy_loss)

        # Update value function
        with tf.GradientTape() as tape:
            values = self.value(states, training=True)
            value_targets = rewards + 0.99 * (1 - dones) * next_values
            value_loss = tf.reduce_mean(tf.square(value_targets - values))

        value_grads = tape.gradient(
            value_loss,
            self.value.trainable_variables
        )
        self.value_optimizer.apply_gradients(
            zip(value_grads, self.value.trainable_variables)
        )

        metrics["value_loss"] = float(value_loss)

        return metrics

    def save_weights(self, path: str):
        """Save network weights.

        Args:
            path: Path to save weights
        """
        self.policy.save_weights(f"{path}_policy")
        self.value.save_weights(f"{path}_value")

    def load_weights(self, path: str):
        """Load network weights.

        Args:
            path: Path to load weights
        """
        self.policy.load_weights(f"{path}_policy")
        self.value.load_weights(f"{path}_value")
