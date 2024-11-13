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
# OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
from typing import Dict, List, Tuple, Any, Optional
from .utils import get_device_strategy, create_optimizer

class CognitionAgent(tf.keras.Model):
    """Cognition Agent implementation for reinforcement learning.

    This class implements a cognitive architecture for reinforcement learning,
    incorporating perception, memory, and decision-making components.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        memory_size: int = 1000,
        attention_heads: int = 4,
        learning_rate: float = 3e-4,
        **kwargs
    ):
        """Initialize CognitionAgent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            memory_size: Size of episodic memory
            attention_heads: Number of attention heads
            learning_rate: Learning rate
            **kwargs: Additional arguments
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_size = memory_size
        self.attention_heads = attention_heads

        # Get device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create perception network
            self.perception_net = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    hidden_dims[0],
                    activation='relu',
                    input_shape=(state_dim,)
                ),
                tf.keras.layers.Dense(
                    hidden_dims[1],
                    activation='relu'
                )
            ])

            # Create memory network (transformer-based)
            self.memory_key = tf.keras.layers.Dense(hidden_dims[1])
            self.memory_query = tf.keras.layers.Dense(hidden_dims[1])
            self.memory_value = tf.keras.layers.Dense(hidden_dims[1])

            # Create decision network
            self.decision_net = tf.keras.Sequential([
                tf.keras.layers.Dense(hidden_dims[1], activation='relu'),
                tf.keras.layers.Dense(action_dim)
            ])

            # Initialize memory
            self.memory = tf.Variable(
                tf.zeros([memory_size, hidden_dims[1]]),
                trainable=False
            )
            self.memory_counter = tf.Variable(0, trainable=False)

            # Create optimizer
            self.optimizer = create_optimizer(learning_rate)

    def attention(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor
    ) -> tf.Tensor:
        """Multi-head attention mechanism.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor

        Returns:
            Attention output
        """
        # Split heads
        batch_size = tf.shape(query)[0]
        query = tf.reshape(
            query,
            [batch_size, -1, self.attention_heads, query.shape[-1] // self.attention_heads]
        )
        key = tf.reshape(
            key,
            [batch_size, -1, self.attention_heads, key.shape[-1] // self.attention_heads]
        )
        value = tf.reshape(
            value,
            [batch_size, -1, self.attention_heads, value.shape[-1] // self.attention_heads]
        )

        # Compute attention scores
        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / tf.math.sqrt(
            tf.cast(query.shape[-1], tf.float32)
        )
        weights = tf.nn.softmax(scores)

        # Apply attention
        output = tf.matmul(weights, value)
        output = tf.reshape(
            output,
            [batch_size, -1, output.shape[-2] * output.shape[-1]]
        )

        return output

    def update_memory(self, state_encoding: tf.Tensor):
        """Update episodic memory.

        Args:
            state_encoding: Encoded state to store
        """
        index = self.memory_counter % self.memory_size
        self.memory[index].assign(state_encoding[0])
        self.memory_counter.assign_add(1)

    def call(
        self,
        states: tf.Tensor,
        training: bool = False
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Forward pass through the network.

        Args:
            states: Batch of states
            training: Whether in training mode

        Returns:
            Tuple of (selected actions, additional info)
        """
        # Perception
        state_encoding = self.perception_net(states)

        # Memory retrieval
        query = self.memory_query(state_encoding)
        key = self.memory_key(self.memory)
        value = self.memory_value(self.memory)

        memory_output = self.attention(
            query[:, None],
            key[None],
            value[None]
        )[:, 0]

        # Combine current state with memory
        combined_features = tf.concat(
            [state_encoding, memory_output],
            axis=-1
        )

        # Decision making
        actions = self.decision_net(combined_features)

        # Update memory if training
        if training:
            self.update_memory(state_encoding)

        return actions, {
            "state_encoding": state_encoding,
            "memory_output": memory_output
        }

    def update(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor
    ) -> Dict[str, float]:
        """Update networks.

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
            # Forward pass
            predicted_actions, info = self(states, training=True)

            # Compute losses
            action_loss = tf.reduce_mean(
                tf.square(actions - predicted_actions)
            )

            # Add regularization for memory usage
            memory_sparsity = tf.reduce_mean(
                tf.abs(info["memory_output"])
            )
            total_loss = action_loss + 0.01 * memory_sparsity

        # Update networks
        gradients = tape.gradient(
            total_loss,
            self.trainable_variables
        )
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )

        metrics.update({
            "action_loss": float(action_loss),
            "memory_sparsity": float(memory_sparsity),
            "total_loss": float(total_loss)
        })

        return metrics

    def save_weights(self, path: str):
        """Save network weights.

        Args:
            path: Path to save weights
        """
        super().save_weights(path)
        # Save memory state
        np.save(f"{path}_memory", self.memory.numpy())
        np.save(f"{path}_counter", self.memory_counter.numpy())

    def load_weights(self, path: str):
        """Load network weights.

        Args:
            path: Path to load weights
        """
        super().load_weights(path)
        # Load memory state
        memory_state = np.load(f"{path}_memory.npy")
        counter_state = np.load(f"{path}_counter.npy")
        self.memory.assign(memory_state)
        self.memory_counter.assign(counter_state)
