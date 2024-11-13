import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .utils import get_device_strategy, create_optimizer

class QNetwork(tf.keras.Model):
    """Advanced Q-Network implementation with various extensions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
        dueling: bool = True,
        noisy: bool = True,
        distributional: bool = True,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0
    ):
        """Initialize QNetwork.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate
            dueling: Whether to use dueling architecture
            noisy: Whether to use noisy networks
            distributional: Whether to use distributional RL
            num_atoms: Number of atoms for distributional RL
            v_min: Minimum value for distributional RL
            v_max: Maximum value for distributional RL
        """
        super().__init__()
        self.action_dim = action_dim
        self.dueling = dueling
        self.noisy = noisy
        self.distributional = distributional
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.support = tf.linspace(v_min, v_max, num_atoms)

        # Create feature extractor
        self.feature_layers = []
        current_dim = state_dim
        for dim in hidden_dims:
            if noisy:
                self.feature_layers.extend([
                    NoisyDense(dim),
                    tf.keras.layers.LayerNormalization(),
                    tf.keras.layers.ReLU()
                ])
            else:
                self.feature_layers.extend([
                    tf.keras.layers.Dense(dim),
                    tf.keras.layers.LayerNormalization(),
                    tf.keras.layers.ReLU()
                ])
            current_dim = dim

        if dueling:
            # Value stream
            if noisy:
                self.value_hidden = NoisyDense(256)
                self.value_out = NoisyDense(
                    num_atoms if distributional else 1
                )
            else:
                self.value_hidden = tf.keras.layers.Dense(256)
                self.value_out = tf.keras.layers.Dense(
                    num_atoms if distributional else 1
                )

            # Advantage stream
            if noisy:
                self.advantage_hidden = NoisyDense(256)
                self.advantage_out = NoisyDense(
                    action_dim * num_atoms if distributional else action_dim
                )
            else:
                self.advantage_hidden = tf.keras.layers.Dense(256)
                self.advantage_out = tf.keras.layers.Dense(
                    action_dim * num_atoms if distributional else action_dim
                )
        else:
            if noisy:
                self.q_out = NoisyDense(
                    action_dim * num_atoms if distributional else action_dim
                )
            else:
                self.q_out = tf.keras.layers.Dense(
                    action_dim * num_atoms if distributional else action_dim
                )

    def call(self, states: tf.Tensor) -> tf.Tensor:
        """Forward pass.

        Args:
            states: Batch of states

        Returns:
            Q-values or value distribution
        """
        features = states
        for layer in self.feature_layers:
            features = layer(features)

        if self.dueling:
            # Value stream
            value = tf.keras.activations.relu(self.value_hidden(features))
            value = self.value_out(value)

            # Advantage stream
            advantage = tf.keras.activations.relu(
                self.advantage_hidden(features)
            )
            advantage = self.advantage_out(advantage)

            if self.distributional:
                # Reshape for distributional RL
                value = tf.reshape(value, [-1, 1, self.num_atoms])
                advantage = tf.reshape(
                    advantage,
                    [-1, self.action_dim, self.num_atoms]
                )

                # Combine value and advantage
                q_dist = value + (
                    advantage - tf.reduce_mean(
                        advantage, axis=1, keepdims=True
                    )
                )
                # Apply softmax
                q_dist = tf.nn.softmax(q_dist, axis=-1)
                return q_dist
            else:
                # Regular dueling
                advantage = tf.reshape(
                    advantage,
                    [-1, self.action_dim]
                )
                value = tf.reshape(value, [-1, 1])
                return value + (
                    advantage - tf.reduce_mean(
                        advantage, axis=1, keepdims=True
                    )
                )
        else:
            q_values = self.q_out(features)
            if self.distributional:
                # Reshape and apply softmax for distributional RL
                q_dist = tf.reshape(
                    q_values,
                    [-1, self.action_dim, self.num_atoms]
                )
                return tf.nn.softmax(q_dist, axis=-1)
            else:
                return q_values

    def reset_noise(self):
        """Reset noise for all noisy layers."""
        if self.noisy:
            for layer in self.feature_layers:
                if isinstance(layer, NoisyDense):
                    layer.reset_noise()
            if self.dueling:
                self.value_hidden.reset_noise()
                self.value_out.reset_noise()
                self.advantage_hidden.reset_noise()
                self.advantage_out.reset_noise()
            else:
                self.q_out.reset_noise()

class NoisyDense(tf.keras.layers.Layer):
    """Noisy Dense Layer implementation."""

    def __init__(
        self,
        units: int,
        sigma_init: float = 0.5
    ):
        """Initialize NoisyDense.

        Args:
            units: Number of output units
            sigma_init: Initial noise standard deviation
        """
        super().__init__()
        self.units = units
        self.sigma_init = sigma_init

    def build(self, input_shape):
        """Build layer.

        Args:
            input_shape: Input shape
        """
        self.input_dim = int(input_shape[-1])

        # Initialize mu weights
        mu_range = 1 / np.sqrt(self.input_dim)
        self.weight_mu = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer=tf.random_uniform_initializer(-mu_range, mu_range),
            trainable=True,
            name='weight_mu'
        )
        self.bias_mu = self.add_weight(
            shape=(self.units,),
            initializer=tf.random_uniform_initializer(-mu_range, mu_range),
            trainable=True,
            name='bias_mu'
        )

        # Initialize sigma weights
        self.weight_sigma = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer=tf.constant_initializer(
                self.sigma_init / np.sqrt(self.input_dim)
            ),
            trainable=True,
            name='weight_sigma'
        )
        self.bias_sigma = self.add_weight(
            shape=(self.units,),
            initializer=tf.constant_initializer(
                self.sigma_init / np.sqrt(self.input_dim)
            ),
            trainable=True,
            name='bias_sigma'
        )

        self.reset_noise()

    def call(self, inputs):
        """Forward pass.

        Args:
            inputs: Input tensor

        Returns:
            Output tensor
        """
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return tf.matmul(inputs, weight) + bias

    def reset_noise(self):
        """Reset noise."""
        self.weight_epsilon = tf.random.normal((self.input_dim, self.units))
        self.bias_epsilon = tf.random.normal((self.units,))
