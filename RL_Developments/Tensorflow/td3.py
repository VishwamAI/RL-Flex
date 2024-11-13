import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, Optional
from .utils import get_device_strategy


class TD3Network(tf.keras.Model):
    """Twin Delayed Deep Deterministic Policy Gradient network implementation in TensorFlow.

    Implements the neural networks for TD3, including:
    - Actor network (deterministic policy)
    - Twin critic networks (Q-functions)
    - Target networks for both actor and critics

    Architecture matches the JAX implementation for consistency.
    Uses TensorFlow's distribution strategy for device handling.
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0,
                 hidden_dim: int = 64):
        """Initialize TD3 networks.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            max_action: Maximum action value
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        self.max_action = max_action

        # Actor network (deterministic policy)
        self.actor = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu',
                                kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(hidden_dim, activation='relu',
                                kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(action_dim, activation='tanh',
                                kernel_initializer='glorot_uniform')
        ])

        # Twin critics
        self.critic1 = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu',
                                kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(hidden_dim, activation='relu',
                                kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform')
        ])

        self.critic2 = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu',
                                kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(hidden_dim, activation='relu',
                                kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform')
        ])

    def get_action(self, state: tf.Tensor, noise: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Get action from the actor network.

        Args:
            state: Current state observation
            noise: Optional exploration noise

        Returns:
            Action scaled by max_action
        """
        action = self.actor(state)
        if noise is not None:
            action += noise
        return tf.clip_by_value(action * self.max_action, -self.max_action, self.max_action)

    def q_values(self, state: tf.Tensor, action: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute Q-values from both critic networks.

        Args:
            state: Current state observation
            action: Action to evaluate

        Returns:
            Tuple of (Q1_values, Q2_values)
        """
        inputs = tf.concat([state, action], axis=-1)
        return self.critic1(inputs), self.critic2(inputs)


class TD3Agent:
    """Twin Delayed Deep Deterministic Policy Gradient agent implementation in TensorFlow.

    Implements the TD3 algorithm with:
    - Twin critics for reduced overestimation
    - Delayed policy updates
    - Target policy smoothing
    - Clipped double Q-learning

    Uses TensorFlow's distribution strategy for device handling.
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0,
                 learning_rate: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 policy_noise: float = 0.2, noise_clip: float = 0.5,
                 policy_delay: int = 2):
        """Initialize TD3 agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            max_action: Maximum action value
            learning_rate: Learning rate for all networks
            gamma: Discount factor
            tau: Soft update coefficient
            policy_noise: Standard deviation of target policy smoothing noise
            noise_clip: Maximum value of target policy smoothing noise
            policy_delay: Number of iterations between policy updates
        """
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_iterations = 0

        # Get appropriate device strategy
        self.strategy = get_device_strategy()

        # Initialize all components within strategy scope
        with self.strategy.scope():
            # Create networks
            self.actor_critic = TD3Network(state_dim, action_dim, max_action)
            self.target_network = TD3Network(state_dim, action_dim, max_action)

            # Copy weights to target network
            for target, source in zip(self.target_network.variables,
                                    self.actor_critic.variables):
                target.assign(source)

            # Create optimizers
            self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
            self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate)

    def get_action(self, state: tf.Tensor, noise_scale: float = 0.1) -> tf.Tensor:
        """Select action using current policy.

        Args:
            state: Current state observation
            noise_scale: Scale of exploration noise

        Returns:
            Selected action
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        noise = tf.random.normal(shape=(1, self.actor_critic.actor.output_shape[-1]),
                               stddev=noise_scale)
        return self.actor_critic.get_action(state, noise)[0]

    def update(self, states: tf.Tensor, actions: tf.Tensor, rewards: tf.Tensor,
               next_states: tf.Tensor, dones: tf.Tensor) -> Dict[str, float]:
        """Update networks using TD3 objective.

        Args:
            states: Batch of state observations
            actions: Batch of actions taken
            rewards: Batch of rewards received
            next_states: Batch of next state observations
            dones: Batch of done flags

        Returns:
            Dictionary containing loss metrics
        """
        # Ensure tensors are in the correct format
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Update critics
        with tf.GradientTape(persistent=True) as tape:
            # Select next actions with target policy smoothing
            noise = tf.clip_by_value(
                tf.random.normal(tf.shape(actions)) * self.policy_noise,
                -self.noise_clip, self.noise_clip
            )
            next_actions = self.target_network.get_action(next_states, noise)

            # Compute target Q-values
            target_q1, target_q2 = self.target_network.q_values(next_states, next_actions)
            target_q = tf.minimum(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * target_q

            # Compute current Q-values
            current_q1, current_q2 = self.actor_critic.q_values(states, actions)

            # Compute critic losses
            critic1_loss = tf.reduce_mean(tf.square(current_q1 - target_q))
            critic2_loss = tf.reduce_mean(tf.square(current_q2 - target_q))
            critic_loss = critic1_loss + critic2_loss

        # Update critics
        critic_vars = (self.actor_critic.critic1.trainable_variables +
                     self.actor_critic.critic2.trainable_variables)
        critic_grads = tape.gradient(critic_loss, critic_vars)
        self.critic_optimizer.apply_gradients(zip(critic_grads, critic_vars))

        actor_loss = 0.0
        # Delayed policy updates
        if self.total_iterations % self.policy_delay == 0:
            # Update actor
            with tf.GradientTape() as tape:
                # Compute actor loss
                actor_actions = self.actor_critic.get_action(states)
                actor_q1, _ = self.actor_critic.q_values(states, actor_actions)
                actor_loss = -tf.reduce_mean(actor_q1)

            # Update actor
            actor_grads = tape.gradient(
                actor_loss,
                self.actor_critic.actor.trainable_variables
            )
            self.actor_optimizer.apply_gradients(zip(
                actor_grads,
                self.actor_critic.actor.trainable_variables
            ))

            # Soft update target networks
            for target, source in zip(self.target_network.variables,
                                    self.actor_critic.variables):
                target.assign(target * (1 - self.tau) + source * self.tau)

        self.total_iterations += 1

        return {
            'critic_loss': float(critic_loss.numpy()),
            'actor_loss': float(actor_loss.numpy()) if isinstance(actor_loss, tf.Tensor) else actor_loss
        }
