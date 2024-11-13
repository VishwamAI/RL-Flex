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

class SACNetwork(tf.keras.Model):
    """Soft Actor-Critic network implementation in TensorFlow.

    Implements the neural networks for SAC, including:
    - Policy network (actor)
    - Twin Q-networks (critics)

    Architecture matches the JAX implementation for consistency.
    Uses TensorFlow's distribution strategy for device handling.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """Initialize SAC networks.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        self.action_dim = action_dim

        # Policy network
        self.policy_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu',
                                kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(hidden_dim, activation='relu',
                                kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(action_dim * 2, kernel_initializer='glorot_uniform')
        ])

        # Twin Q-networks
        self.q1_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu',
                                kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(hidden_dim, activation='relu',
                                kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform')
        ])

        self.q2_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu',
                                kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(hidden_dim, activation='relu',
                                kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform')
        ])

        # Build networks
        self.build([(None, state_dim), (None, action_dim)])

    def build(self, input_shapes: list) -> None:
        """Build the model by running a forward pass with dummy inputs."""
        state_shape, action_shape = input_shapes
        dummy_state = tf.keras.Input(shape=state_shape[1:])
        dummy_action = tf.keras.Input(shape=action_shape[1:])

        self.policy_net(dummy_state)
        self.q1_net(tf.concat([dummy_state, dummy_action], axis=-1))
        self.q2_net(tf.concat([dummy_state, dummy_action], axis=-1))

        super().build(input_shapes)

    def get_action(self, state: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """Sample action from the policy network.

        Args:
            state: Current state observation
            training: Whether to sample (True) or use mean (False)

        Returns:
            Tuple of (sampled_action, log_probability)
        """
        mean, log_std = tf.split(self.policy_net(state), 2, axis=-1)
        log_std = tf.clip_by_value(log_std, -20, 2)
        std = tf.exp(log_std)

        # Sample action using reparameterization trick
        eps = tf.random.normal(tf.shape(mean)) if training else 0.0
        action = mean + eps * std

        # Compute log probability
        log_prob = -0.5 * (
            tf.math.log(2 * np.pi) +
            2 * log_std +
            tf.square((action - mean) / (std + 1e-6))
        )
        log_prob = tf.reduce_sum(log_prob, axis=-1, keepdims=True)

        # Apply tanh squashing
        action = tf.tanh(action)
        log_prob -= tf.reduce_sum(
            tf.math.log(1 - tf.square(action) + 1e-6),
            axis=-1, keepdims=True
        )

        return action, log_prob

    def q_values(self, state: tf.Tensor, action: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute Q-values from both Q-networks.

        Args:
            state: Current state observation
            action: Action to evaluate

        Returns:
            Tuple of (Q1_values, Q2_values)
        """
        inputs = tf.concat([state, action], axis=-1)
        return self.q1_net(inputs), self.q2_net(inputs)

class SACAgent:
    """Soft Actor-Critic agent implementation in TensorFlow.

    Implements the SAC algorithm with automatic temperature tuning.
    Uses TensorFlow's distribution strategy for device handling.
    """
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 3e-4,
                 gamma: float = 0.99, tau: float = 0.005):
        """Initialize SAC agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            learning_rate: Learning rate for all networks
            gamma: Discount factor
            tau: Soft update coefficient
        """
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = -action_dim  # Heuristic value

        # Get appropriate device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create networks
            self.actor_critic = SACNetwork(state_dim, action_dim)
            self.target_critic = SACNetwork(state_dim, action_dim)

            # Copy weights to target network
            for target, source in zip(self.target_critic.variables,
                                    self.actor_critic.variables):
                target.assign(source)

            # Create optimizers
            self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
            self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate)

            # Initialize log alpha (temperature parameter)
            self.log_alpha = tf.Variable(0.0)
            self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate)

    def get_action(self, state: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Select action using current policy.

        Args:
            state: Current state observation
            training: Whether to sample (True) or use mean (False)

        Returns:
            Selected action
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action, _ = self.actor_critic.get_action(state, training)
        return action[0]

    def update(self, states: tf.Tensor, actions: tf.Tensor, rewards: tf.Tensor,
               next_states: tf.Tensor, dones: tf.Tensor) -> Dict[str, float]:
        """Update networks using SAC objective.

        Args:
            states: Batch of state observations
            actions: Batch of actions taken
            rewards: Batch of rewards received
            next_states: Batch of next state observations
            dones: Batch of done flags

        Returns:
            Dictionary containing loss metrics
        """
        with self.strategy.scope():
            # Update critic
            with tf.GradientTape(persistent=True) as tape:
                # Sample actions and compute log probs for next states
                next_actions, next_log_probs = self.actor_critic.get_action(next_states, True)

                # Compute target Q-values
                next_q1, next_q2 = self.target_critic.q_values(next_states, next_actions)
                next_q = tf.minimum(next_q1, next_q2)

                # Compute target value with entropy
                alpha = tf.exp(self.log_alpha)
                target_q = rewards + self.gamma * (1 - dones) * (next_q - alpha * next_log_probs)

                # Compute current Q-values
                current_q1, current_q2 = self.actor_critic.q_values(states, actions)

                # Compute critic losses
                q1_loss = tf.reduce_mean(tf.square(current_q1 - target_q))
                q2_loss = tf.reduce_mean(tf.square(current_q2 - target_q))
                critic_loss = q1_loss + q2_loss

            # Update critics
            critic_vars = (self.actor_critic.q1_net.trainable_variables +
                         self.actor_critic.q2_net.trainable_variables)
            critic_grads = tape.gradient(critic_loss, critic_vars)
            self.critic_optimizer.apply_gradients(zip(critic_grads, critic_vars))

            # Update actor
            with tf.GradientTape(persistent=True) as tape:
                # Sample actions and compute log probs
                actions, log_probs = self.actor_critic.get_action(states, True)

                # Compute Q-values for sampled actions
                q1, q2 = self.actor_critic.q_values(states, actions)
                q = tf.minimum(q1, q2)

                # Compute actor loss with entropy
                alpha = tf.exp(self.log_alpha)
                actor_loss = tf.reduce_mean(alpha * log_probs - q)

                # Compute temperature loss
                alpha_loss = -tf.reduce_mean(
                    self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy)
                )

            # Update actor
            actor_grads = tape.gradient(
                actor_loss,
                self.actor_critic.policy_net.trainable_variables
            )
            self.actor_optimizer.apply_gradients(zip(
                actor_grads,
                self.actor_critic.policy_net.trainable_variables
            ))

            # Update temperature parameter
            alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

            # Soft update target network
            for target, source in zip(self.target_critic.variables,
                                    self.actor_critic.variables):
                target.assign(target * (1 - self.tau) + source * self.tau)

            return {
                'critic_loss': float(critic_loss.numpy()),
                'actor_loss': float(actor_loss.numpy()),
                'alpha_loss': float(alpha_loss.numpy()),
                'alpha': float(alpha.numpy())
            }
