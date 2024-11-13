import tensorflow as tf
from typing import Dict, Tuple
from .utils import get_device_strategy, create_optimizer
from .utils.rl_utils import discount_cumsum

class PolicyGradient:
    """REINFORCE (Monte Carlo Policy Gradient) implementation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99
    ):
        """Initialize Policy Gradient.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate
            gamma: Discount factor
        """
        self.gamma = gamma

        # Get device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create policy network
            self.policy_network = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(action_dim, activation='softmax')
            ])

            # Create optimizer
            self.optimizer = create_optimizer(learning_rate)

            # Build network
            dummy_state = tf.zeros((1, state_dim))
            self.policy_network(dummy_state)

    def get_action(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Sample action from policy.

        Args:
            state: Current state

        Returns:
            Tuple of (action, log probability)
        """
        state = tf.expand_dims(state, 0)
        probs = self.policy_network(state)[0]
        action = tf.random.categorical(tf.math.log(probs[None, :]), 1)[0, 0]
        log_prob = tf.math.log(probs[action])
        return action, log_prob

    def update(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor
    ) -> Dict[str, float]:
        """Update policy network.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards

        Returns:
            Dictionary of metrics
        """
        with self.strategy.scope():
            # Calculate returns
            returns = discount_cumsum(rewards, self.gamma)

            with tf.GradientTape() as tape:
                # Get action probabilities
                probs = self.policy_network(states)
                actions_one_hot = tf.one_hot(actions, probs.shape[-1])
                selected_probs = tf.reduce_sum(
                    probs * actions_one_hot, axis=-1
                )

                # Compute loss
                loss = -tf.reduce_mean(
                    tf.math.log(selected_probs) * returns
                )

            # Update policy
            gradients = tape.gradient(
                loss,
                self.policy_network.trainable_variables
            )
            self.optimizer.apply_gradients(
                zip(gradients, self.policy_network.trainable_variables)
            )

            return {
                "policy_loss": float(loss),
                "mean_return": float(tf.reduce_mean(returns))
            }

class ActorCritic:
    """Advantage Actor-Critic (A2C) implementation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99
    ):
        """Initialize Actor-Critic.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            gamma: Discount factor
        """
        self.gamma = gamma

        # Get device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create actor network
            self.actor = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(action_dim, activation='softmax')
            ])

            # Create critic network
            self.critic = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)
            ])

            # Create optimizers
            self.actor_optimizer = create_optimizer(actor_lr)
            self.critic_optimizer = create_optimizer(critic_lr)

            # Build networks
            dummy_state = tf.zeros((1, state_dim))
            self.actor(dummy_state)
            self.critic(dummy_state)

    def get_action(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Sample action from policy.

        Args:
            state: Current state

        Returns:
            Tuple of (action, log probability)
        """
        state = tf.expand_dims(state, 0)
        probs = self.actor(state)[0]
        action = tf.random.categorical(tf.math.log(probs[None, :]), 1)[0, 0]
        log_prob = tf.math.log(probs[action])
        return action, log_prob

    def update(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor
    ) -> Dict[str, float]:
        """Update actor and critic networks.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags

        Returns:
            Dictionary of metrics
        """
        with self.strategy.scope():
            # Compute returns and advantages
            next_values = self.critic(next_states)
            values = self.critic(states)
            td_targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = td_targets - values

            # Update critic
            with tf.GradientTape() as tape:
                value_pred = self.critic(states)
                critic_loss = tf.reduce_mean(
                    tf.square(td_targets - value_pred)
                )

            critic_grads = tape.gradient(
                critic_loss,
                self.critic.trainable_variables
            )
            self.critic_optimizer.apply_gradients(
                zip(critic_grads, self.critic.trainable_variables)
            )

            # Update actor
            with tf.GradientTape() as tape:
                probs = self.actor(states)
                actions_one_hot = tf.one_hot(actions, probs.shape[-1])
                selected_probs = tf.reduce_sum(
                    probs * actions_one_hot, axis=-1
                )

                # Compute loss
                actor_loss = -tf.reduce_mean(
                    tf.math.log(selected_probs) * tf.stop_gradient(advantages)
                )

            actor_grads = tape.gradient(
                actor_loss,
                self.actor.trainable_variables
            )
            self.actor_optimizer.apply_gradients(
                zip(actor_grads, self.actor.trainable_variables)
            )

            return {
                "actor_loss": float(actor_loss),
                "critic_loss": float(critic_loss),
                "mean_advantage": float(tf.reduce_mean(advantages))
            }
