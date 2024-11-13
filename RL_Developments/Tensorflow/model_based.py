import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, Optional
from .utils import get_device_strategy


class WorldModel(tf.keras.Model):
    """World model for model-based reinforcement learning.

    Implements a dynamics model that predicts next states and rewards
    given current states and actions. Used for planning and
    curiosity-driven exploration.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64,
                 ensemble_size: int = 5):
        """Initialize world model.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
            ensemble_size: Number of ensemble models for uncertainty estimation
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size

        # Create ensemble of dynamics models
        self.dynamics_models = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(hidden_dim, activation='relu',
                                    kernel_initializer='glorot_uniform'),
                tf.keras.layers.Dense(hidden_dim, activation='relu',
                                    kernel_initializer='glorot_uniform'),
                tf.keras.layers.Dense(state_dim + 1)  # predict next state and reward
            ]) for _ in range(ensemble_size)
        ]

    def call(self, states: tf.Tensor, actions: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Predict next states and rewards with uncertainty estimation.

        Args:
            states: Current state observations
            actions: Actions to take

        Returns:
            Tuple of (predicted_next_states, predicted_rewards, prediction_uncertainty)
        """
        inputs = tf.concat([states, actions], axis=-1)

        # Get predictions from all ensemble models
        predictions = tf.stack([model(inputs) for model in self.dynamics_models])

        # Split predictions into next states and rewards
        next_states_ensemble = predictions[..., :self.state_dim]
        rewards_ensemble = predictions[..., -1:]

        # Compute means and uncertainties
        next_states = tf.reduce_mean(next_states_ensemble, axis=0)
        rewards = tf.reduce_mean(rewards_ensemble, axis=0)
        uncertainties = tf.math.reduce_std(next_states_ensemble, axis=0)

        return next_states, rewards, uncertainties

    def compute_curiosity_reward(self, states: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
        """Compute curiosity reward based on prediction uncertainty.

        Args:
            states: Current state observations
            actions: Actions to take

        Returns:
            Curiosity rewards based on prediction uncertainty
        """
        _, _, uncertainties = self(states, actions)
        return tf.reduce_mean(uncertainties, axis=-1, keepdims=True)


class ModelBasedAgent:
    """Model-based reinforcement learning agent with curiosity-driven exploration.

    Combines a world model for planning and curiosity-driven exploration
    with a base policy (e.g., SAC or TD3) for action selection.
    """
    def __init__(self, state_dim: int, action_dim: int, base_agent: tf.keras.Model,
                 learning_rate: float = 1e-3, curiosity_weight: float = 0.1):
        """Initialize model-based agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            base_agent: Base policy agent (SAC or TD3)
            learning_rate: Learning rate for world model
            curiosity_weight: Weight for curiosity reward
        """
        self.curiosity_weight = curiosity_weight
        self.base_agent = base_agent

        # Use the same strategy as the base agent
        self.strategy = base_agent.strategy

        # Initialize world model and optimizer using the same strategy
        with self.strategy.scope():
            self.world_model = WorldModel(state_dim, action_dim)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def get_action(self, state: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Select action using base policy.

        Args:
            state: Current state observation
            training: Whether to add exploration noise

        Returns:
            Selected action
        """
        return self.base_agent.get_action(state, training)

    def update_world_model(self, states: tf.Tensor, actions: tf.Tensor,
                          next_states: tf.Tensor, rewards: tf.Tensor) -> Dict[str, float]:
        """Update world model using observed transitions.

        Args:
            states: Batch of state observations
            actions: Batch of actions taken
            next_states: Batch of next state observations
            rewards: Batch of rewards received

        Returns:
            Dictionary containing loss metrics
        """
        with self.strategy.scope():
            with tf.GradientTape() as tape:
                # Predict next states and rewards
                pred_next_states, pred_rewards, _ = self.world_model(states, actions)

                # Compute prediction losses
                state_loss = tf.reduce_mean(tf.square(pred_next_states - next_states))
                reward_loss = tf.reduce_mean(tf.square(pred_rewards - rewards))
                total_loss = state_loss + reward_loss

            # Update world model
            grads = tape.gradient(total_loss, self.world_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.world_model.trainable_variables))

            return {
                'state_loss': float(state_loss.numpy()),
                'reward_loss': float(reward_loss.numpy()),
                'total_loss': float(total_loss.numpy())
            }

    def update(self, states: tf.Tensor, actions: tf.Tensor, rewards: tf.Tensor,
               next_states: tf.Tensor, dones: tf.Tensor) -> Dict[str, float]:
        """Update both world model and base policy.

        Args:
            states: Batch of state observations
            actions: Batch of actions taken
            rewards: Batch of rewards received
            next_states: Batch of next state observations
            dones: Batch of done flags

        Returns:
            Dictionary containing all loss metrics
        """
        # Update world model
        world_model_losses = self.update_world_model(states, actions, next_states, rewards)

        # Compute curiosity rewards
        curiosity_rewards = self.world_model.compute_curiosity_reward(states, actions)
        augmented_rewards = rewards + self.curiosity_weight * curiosity_rewards

        # Update base policy with augmented rewards
        policy_losses = self.base_agent.update(states, actions, augmented_rewards,
                                             next_states, dones)

        # Combine and return all losses
        return {**world_model_losses, **policy_losses}
