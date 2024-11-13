import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, Any, Optional

class DQNetwork(tf.keras.Model):
    """Deep Q-Network implementation in TensorFlow.

    This network maps state observations to Q-values for each possible action.
    Architecture matches the JAX implementation for consistency.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu',
                                          kernel_initializer='glorot_uniform')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu',
                                          kernel_initializer='glorot_uniform')
        self.out = tf.keras.layers.Dense(action_dim,
                                       kernel_initializer='glorot_uniform')

        # Build model with sample input
        self.build((None, state_dim))

    def call(self, x, training=False):
        """Forward pass through the network.

        Args:
            x: Input tensor representing the state
            training: Boolean indicating training mode

        Returns:
            Q-values for each action
        """
        x = self.dense1(x)
        x = self.dense2(x)
        return self.out(x)

class ActorCritic(tf.keras.Model):
    """Actor-Critic network implementation in TensorFlow.

    Combined network that outputs both policy (actor) and value (critic).
    Architecture matches the JAX implementation for consistency.

    Args:
        action_dim: Dimension of the action space
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # Actor network
        self.actor_dense1 = tf.keras.layers.Dense(64, activation='relu',
                                               kernel_initializer='glorot_uniform')
        self.actor_dense2 = tf.keras.layers.Dense(64, activation='relu',
                                               kernel_initializer='glorot_uniform')
        self.actor_out = tf.keras.layers.Dense(action_dim, activation=None,
                                            kernel_initializer='glorot_uniform')

        # Critic network
        self.critic_dense1 = tf.keras.layers.Dense(64, activation='relu',
                                                kernel_initializer='glorot_uniform')
        self.critic_dense2 = tf.keras.layers.Dense(64, activation='relu',
                                                kernel_initializer='glorot_uniform')
        self.critic_out = tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform')

        # Build model with sample input
        self.build((None, state_dim))

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward pass through both actor and critic networks.

        Args:
            x: Input tensor representing the state
            training: Boolean indicating training mode (unused, kept for API consistency)

        Returns:
            Tuple of (action_probabilities, value_estimate)
        """
        # Actor forward pass
        actor_hidden = self.actor_dense1(x)
        actor_hidden = self.actor_dense2(actor_hidden)
        actor_output = self.actor_out(actor_hidden)
        action_probs = tf.nn.softmax(actor_output)

        # Critic forward pass
        critic_hidden = self.critic_dense1(x)
        critic_hidden = self.critic_dense2(critic_hidden)
        value = self.critic_out(critic_hidden)

        return action_probs, tf.squeeze(value, axis=-1)

# Device handling utility
def get_device_strategy():
    """Returns appropriate TensorFlow distribution strategy.

    Returns:
        tf.distribute.Strategy: Appropriate strategy for available hardware
    """
    if tf.config.list_physical_devices('GPU'):
        return tf.distribute.MirroredStrategy()
    return tf.distribute.OneDeviceStrategy("/cpu:0")

class DQNAgent:
    """Deep Q-Network Agent implementation in TensorFlow.

    Implements a DQN agent with experience replay and target network.
    Uses TensorFlow's distribution strategy for device handling.
    """
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """Initialize DQN Agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Get appropriate device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create online and target networks
            self.q_network = DQNetwork(state_dim, action_dim)
            self.target_network = DQNetwork(state_dim, action_dim)
            self.target_network.set_weights(self.q_network.get_weights())

            # Initialize optimizer
            self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def get_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.q_network(state)
        return tf.argmax(q_values[0]).numpy()

    def update(self, state, action, reward, next_state, done):
        """Update the network using a single transition."""
        with self.strategy.scope():
            with tf.GradientTape() as tape:
                loss = self._compute_loss(state, action, reward, next_state, done)
            
            # Compute and apply gradients
            gradients = tape.gradient(loss, self.q_network.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

            # Update exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            return loss.numpy()

    def _compute_loss(self, state, action, reward, next_state, done):
        """Compute TD loss for DQN update."""
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
        
        # Compute target Q-values
        next_q_values = self.target_network(next_state)
        max_next_q = tf.reduce_max(next_q_values, axis=1)
        target = reward + (1 - done) * self.gamma * max_next_q
        
        # Compute current Q-values and gather the Q-value for the action taken
        current_q_values = self.q_network(state)
        current_q = tf.gather(current_q_values[0], action)
        
        return tf.keras.losses.MSE(target, current_q)

    def update_target_network(self):
        """Synchronize target network with current network."""
        self.target_network.set_weights(self.q_network.get_weights())

# Device handling utility (already defined above)

class PPOAgent:
    """Proximal Policy Optimization (PPO) agent implementation in TensorFlow.

    Implements PPO algorithm with clipped objective and value function loss.
    Uses TensorFlow's distribution strategy for device handling.
    """
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 3e-4):
        """Initialize PPO Agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            learning_rate: Learning rate for the optimizer
        """
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Get appropriate device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create actor-critic network
            self.ac_network = ActorCritic(state_dim, action_dim)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def get_action(self, state: tf.Tensor) -> Tuple[int, tf.Tensor]:
        """Select action using current policy."""
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs, _ = self.ac_network(state)
        action = tf.random.categorical(tf.math.log(action_probs), 1)
        return int(action[0, 0]), action_probs[0]

    def update(self, states: tf.Tensor, actions: tf.Tensor, rewards: tf.Tensor,
               dones: tf.Tensor, old_log_probs: tf.Tensor) -> Dict[str, float]:
        """Update policy and value function using PPO objective."""
        with self.strategy.scope():
            with tf.GradientTape() as tape:
                # Forward pass
                action_probs, values = self.ac_network(states)

                # Compute log probabilities
                indices = tf.range(0, tf.shape(actions)[0])
                action_indices = tf.stack([indices, actions], axis=1)
                new_log_probs = tf.math.log(tf.gather_nd(action_probs, action_indices))

                # Compute advantages (simplified to match JAX)
                advantages = rewards - values

                # Compute ratio and clipped objective (matching JAX parameters)
                ratio = tf.exp(new_log_probs - old_log_probs)
                clip_adv = tf.clip_by_value(ratio, 0.8, 1.2) * advantages
                loss = -tf.minimum(ratio * advantages, clip_adv)

                # Compute value loss and entropy (matching JAX coefficients)
                value_loss = tf.square(rewards - values)
                entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=-1)

                # Compute total loss with matching coefficients
                total_loss = tf.reduce_mean(loss + 0.5 * value_loss - 0.01 * entropy)

            # Compute and apply gradients
            gradients = tape.gradient(total_loss, self.ac_network.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.ac_network.trainable_variables))

            return {
                'total_loss': float(total_loss.numpy()),
                'policy_loss': float(tf.reduce_mean(loss).numpy()),
                'value_loss': float(tf.reduce_mean(value_loss).numpy()),
                'entropy': float(tf.reduce_mean(entropy).numpy())
            }
