import tensorflow as tf
from typing import Tuple, Dict, Any, List
from .utils import get_device_strategy, create_optimizer

class Actor(tf.keras.Model):
    """Actor network for PPO."""

    def __init__(self, state_dim: int, action_dim: int):
        """Initialize actor network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.mean = tf.keras.layers.Dense(action_dim)
        self.log_std = tf.keras.layers.Dense(action_dim)

    def call(self, states: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward pass through the network.

        Args:
            states: Batch of states

        Returns:
            Tuple of (action means, log standard deviations)
        """
        x = self.dense1(states)
        x = self.dense2(x)
        return self.mean(x), self.log_std(x)

class Critic(tf.keras.Model):
    """Critic network for PPO."""

    def __init__(self, state_dim: int):
        """Initialize critic network.

        Args:
            state_dim: Dimension of state space
        """
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.value = tf.keras.layers.Dense(1)

    def call(self, states: tf.Tensor) -> tf.Tensor:
        """Forward pass through the network.

        Args:
            states: Batch of states

        Returns:
            State values
        """
        x = self.dense1(states)
        x = self.dense2(x)
        return self.value(x)

class PPOAgent:
    """Proximal Policy Optimization Agent implementation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        epsilon_clip: float = 0.2,
        c1: float = 1.0,
        c2: float = 0.01,
        batch_size: int = 64,
        n_epochs: int = 10
    ):
        """Initialize PPO Agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            lambda_gae: GAE parameter
            epsilon_clip: PPO clipping parameter
            c1: Value loss coefficient
            c2: Entropy coefficient
            batch_size: Size of training batch
            n_epochs: Number of epochs per update
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.epsilon_clip = epsilon_clip
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        # Get device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create networks
            self.actor = Actor(state_dim, action_dim)
            self.critic = Critic(state_dim)

            # Create optimizers
            self.actor_optimizer = create_optimizer(learning_rate)
            self.critic_optimizer = create_optimizer(learning_rate)

            # Build networks
            dummy_state = tf.zeros((1, state_dim))
            self.actor(dummy_state)
            self.critic(dummy_state)

    def get_action(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Sample action from policy.

        Args:
            state: Current state

        Returns:
            Tuple of (sampled action, log probability)
        """
        state = tf.expand_dims(state, 0)
        mean, log_std = self.actor(state)
        std = tf.exp(log_std)

        # Sample action from normal distribution
        action = mean + tf.random.normal(tf.shape(mean)) * std

        # Compute log probability
        log_prob = -0.5 * (
            tf.square((action - mean) / std) +
            2 * log_std +
            tf.math.log(2 * tf.constant(3.141592653589793))
        )
        log_prob = tf.reduce_sum(log_prob, axis=-1)

        return action[0], log_prob[0]

    def update(self, states: tf.Tensor, actions: tf.Tensor,
               old_log_probs: tf.Tensor, advantages: tf.Tensor,
               returns: tf.Tensor) -> Dict[str, float]:
        """Update the agent's networks.

        Args:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Batch of old log probabilities
            advantages: Batch of advantages
            returns: Batch of returns

        Returns:
            Dictionary of training metrics
        """
        metrics = {}

        with self.strategy.scope():
            for _ in range(self.n_epochs):
                with tf.GradientTape() as actor_tape:
                    # Get current policy distribution
                    mean, log_std = self.actor(states)
                    std = tf.exp(log_std)

                    # Compute log probabilities
                    log_probs = -0.5 * (
                        tf.square((actions - mean) / std) +
                        2 * log_std +
                        tf.math.log(2 * tf.constant(3.141592653589793))
                    )
                    log_probs = tf.reduce_sum(log_probs, axis=-1)

                    # Compute ratio and clipped ratio
                    ratio = tf.exp(log_probs - old_log_probs)
                    clipped_ratio = tf.clip_by_value(
                        ratio,
                        1 - self.epsilon_clip,
                        1 + self.epsilon_clip
                    )

                    # Compute losses
                    surrogate1 = ratio * advantages
                    surrogate2 = clipped_ratio * advantages
                    actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

                    # Add entropy bonus
                    entropy = tf.reduce_mean(
                        0.5 * (tf.math.log(2 * tf.constant(3.141592653589793)) +
                              2 * log_std + 1)
                    )
                    actor_loss = actor_loss - self.c2 * entropy

                # Update actor
                actor_grads = actor_tape.gradient(
                    actor_loss,
                    self.actor.trainable_variables
                )
                self.actor_optimizer.apply_gradients(
                    zip(actor_grads, self.actor.trainable_variables)
                )

                with tf.GradientTape() as critic_tape:
                    # Compute value loss
                    values = self.critic(states)
                    critic_loss = tf.reduce_mean(
                        tf.square(returns - tf.squeeze(values))
                    )

                # Update critic
                critic_grads = critic_tape.gradient(
                    critic_loss,
                    self.critic.trainable_variables
                )
                self.critic_optimizer.apply_gradients(
                    zip(critic_grads, self.critic.trainable_variables)
                )

                metrics.update({
                    "actor_loss": float(actor_loss),
                    "critic_loss": float(critic_loss),
                    "entropy": float(entropy)
                })

        return metrics
