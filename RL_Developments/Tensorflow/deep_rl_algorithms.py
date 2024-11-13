import tensorflow as tf
import numpy as np

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
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Actor network
        self.actor_dense1 = tf.keras.layers.Dense(64, activation='relu',
                                                kernel_initializer='glorot_uniform')
        self.actor_dense2 = tf.keras.layers.Dense(64, activation='relu',
                                                kernel_initializer='glorot_uniform')
        self.actor_out = tf.keras.layers.Dense(action_dim, activation='softmax',
                                             kernel_initializer='glorot_uniform')

        # Critic network
        self.critic_dense1 = tf.keras.layers.Dense(64, activation='relu',
                                                 kernel_initializer='glorot_uniform')
        self.critic_dense2 = tf.keras.layers.Dense(64, activation='relu',
                                                 kernel_initializer='glorot_uniform')
        self.critic_out = tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform')

        # Build model with sample input
        self.build((None, state_dim))

    def call(self, x, training=False):
        """Forward pass through both actor and critic networks.

        Args:
            x: Input tensor representing the state
            training: Boolean indicating training mode

        Returns:
            Tuple of (action_probabilities, value_estimate)
        """
        # Actor forward pass
        actor_x = self.actor_dense1(x)
        actor_x = self.actor_dense2(actor_x)
        action_probs = self.actor_out(actor_x)

        # Critic forward pass
        critic_x = self.critic_dense1(x)
        critic_x = self.critic_dense2(critic_x)
        value = self.critic_out(critic_x)

        return action_probs, value

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
