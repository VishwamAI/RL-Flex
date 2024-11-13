import tensorflow as tf
from typing import Tuple, Dict, Any
from .utils import get_device_strategy, create_optimizer, update_target_network

class QNetwork(tf.keras.Model):
    """Deep Q-Network implementation in TensorFlow."""

    def __init__(self, state_dim: int, action_dim: int):
        """Initialize Q-Network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.q_values = tf.keras.layers.Dense(action_dim)

    def call(self, states: tf.Tensor) -> tf.Tensor:
        """Forward pass through the network.

        Args:
            states: Batch of states

        Returns:
            Q-values for each action
        """
        x = self.dense1(states)
        x = self.dense2(x)
        return self.q_values(x)

class DQNAgent:
    """Deep Q-Learning Agent implementation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256
    ):
        """Initialize DQN Agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            tau: Target network update rate
            batch_size: Size of training batch
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Get device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create networks
            self.q_network = QNetwork(state_dim, action_dim)
            self.target_network = QNetwork(state_dim, action_dim)

            # Create optimizer
            self.optimizer = create_optimizer(learning_rate)

            # Build networks
            dummy_state = tf.zeros((1, state_dim))
            self.q_network(dummy_state)
            self.target_network(dummy_state)

            # Copy weights to target network
            update_target_network(self.q_network, self.target_network, tau=1.0)

    def get_action(self, state: tf.Tensor, epsilon: float = 0.1) -> tf.Tensor:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state
            epsilon: Exploration rate

        Returns:
            Selected action
        """
        if tf.random.uniform(()) < epsilon:
            return tf.random.uniform((), 0, self.action_dim, dtype=tf.int32)
        else:
            state = tf.expand_dims(state, 0)
            q_values = self.q_network(state)
            return tf.argmax(q_values[0])

    def update(self, states: tf.Tensor, actions: tf.Tensor,
               rewards: tf.Tensor, next_states: tf.Tensor,
               dones: tf.Tensor) -> Dict[str, float]:
        """Update the agent's networks.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags

        Returns:
            Dictionary of training metrics
        """
        with self.strategy.scope():
            with tf.GradientTape() as tape:
                # Current Q-values
                current_q = self.q_network(states)
                current_q_actions = tf.gather(current_q,
                                           actions,
                                           batch_dims=1)

                # Target Q-values
                next_q = self.target_network(next_states)
                next_q_max = tf.reduce_max(next_q, axis=1)
                target_q = rewards + self.gamma * next_q_max * (1 - dones)

                # Compute loss
                loss = tf.reduce_mean(tf.square(target_q - current_q_actions))

            # Update Q-network
            gradients = tape.gradient(loss, self.q_network.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.q_network.trainable_variables)
            )

            # Update target network
            update_target_network(self.q_network, self.target_network, self.tau)

            return {
                "q_loss": float(loss),
                "q_value_mean": float(tf.reduce_mean(current_q))
            }
