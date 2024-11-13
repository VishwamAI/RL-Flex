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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .utils import get_device_strategy, create_optimizer

class AgenticBehavior(tf.keras.Model):
    """Agentic Behavior implementation for reinforcement learning.

    This class implements agentic behavior modeling with goal-directed
    decision making and adaptive behavior patterns.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
        **kwargs
    ):
        """Initialize AgenticBehavior.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate
            **kwargs: Additional arguments
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Get device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create behavior networks
            self.goal_network = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    hidden_dims[0],
                    activation='relu',
                    input_shape=(state_dim,)
                ),
                tf.keras.layers.Dense(
                    hidden_dims[1],
                    activation='relu'
                ),
                tf.keras.layers.Dense(state_dim)  # Goal state prediction
            ])

            self.policy_network = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    hidden_dims[0],
                    activation='relu',
                    input_shape=(state_dim * 2,)  # State + Goal
                ),
                tf.keras.layers.Dense(
                    hidden_dims[1],
                    activation='relu'
                ),
                tf.keras.layers.Dense(action_dim * 2)  # Mean and log_std
            ])

            self.value_network = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    hidden_dims[0],
                    activation='relu',
                    input_shape=(state_dim * 2,)  # State + Goal
                ),
                tf.keras.layers.Dense(
                    hidden_dims[1],
                    activation='relu'
                ),
                tf.keras.layers.Dense(1)
            ])

            # Create optimizers
            self.goal_optimizer = create_optimizer(learning_rate)
            self.policy_optimizer = create_optimizer(learning_rate)
            self.value_optimizer = create_optimizer(learning_rate)

    def predict_goal(
        self,
        states: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        """Predict goal states.

        Args:
            states: Batch of states
            training: Whether in training mode

        Returns:
            Predicted goal states
        """
        return self.goal_network(states, training=training)

    def get_action(
        self,
        states: tf.Tensor,
        goals: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Get actions from policy network.

        Args:
            states: Batch of states
            goals: Optional batch of goal states
            training: Whether in training mode

        Returns:
            Tuple of (actions, additional info)
        """
        if goals is None:
            goals = self.predict_goal(states, training=training)

        # Combine state and goal information
        inputs = tf.concat([states, goals], axis=-1)
        outputs = self.policy_network(inputs, training=training)
        means, log_stds = tf.split(outputs, 2, axis=-1)
        stds = tf.exp(tf.clip_by_value(log_stds, -20, 2))

        # Sample actions
        if training:
            actions = means + tf.random.normal(tf.shape(means)) * stds
        else:
            actions = means

        actions = tf.clip_by_value(actions, -1, 1)

        return actions, {
            "means": means,
            "log_stds": log_stds,
            "stds": stds,
            "goals": goals
        }

    def update(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor,
        target_goals: Optional[tf.Tensor] = None
    ) -> Dict[str, float]:
        """Update networks using agentic behavior learning.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            target_goals: Optional batch of target goal states

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Update goal network
        with tf.GradientTape() as tape:
            predicted_goals = self.predict_goal(states, training=True)
            if target_goals is None:
                # If no target goals provided, use next states as targets
                target_goals = next_states
            goal_loss = tf.reduce_mean(tf.square(predicted_goals - target_goals))

        goal_grads = tape.gradient(
            goal_loss,
            self.goal_network.trainable_variables
        )
        self.goal_optimizer.apply_gradients(
            zip(goal_grads, self.goal_network.trainable_variables)
        )

        metrics["goal_loss"] = float(goal_loss)

        # Update policy and value networks
        with tf.GradientTape(persistent=True) as tape:
            # Get current actions and values
            current_goals = self.predict_goal(states, training=True)
            actions, action_info = self.get_action(
                states,
                current_goals,
                training=True
            )
            state_goal_inputs = tf.concat([states, current_goals], axis=-1)
            values = self.value_network(state_goal_inputs, training=True)

            # Get next values
            next_goals = self.predict_goal(next_states, training=True)
            next_state_goal_inputs = tf.concat([next_states, next_goals], axis=-1)
            next_values = self.value_network(next_state_goal_inputs, training=True)

            # Compute advantages
            advantages = (
                rewards +
                0.99 * (1 - dones) * next_values -
                values
            )

            # Compute policy loss
            means = action_info["means"]
            stds = action_info["stds"]
            log_probs = -0.5 * (
                tf.square((actions - means) / stds) +
                2 * tf.math.log(stds) +
                tf.math.log(2 * np.pi)
            )
            log_probs = tf.reduce_sum(log_probs, axis=-1)
            policy_loss = -tf.reduce_mean(log_probs * advantages)

            # Compute value loss
            value_targets = rewards + 0.99 * (1 - dones) * next_values
            value_loss = tf.reduce_mean(tf.square(value_targets - values))

        # Update policy network
        policy_grads = tape.gradient(
            policy_loss,
            self.policy_network.trainable_variables
        )
        self.policy_optimizer.apply_gradients(
            zip(policy_grads, self.policy_network.trainable_variables)
        )

        # Update value network
        value_grads = tape.gradient(
            value_loss,
            self.value_network.trainable_variables
        )
        self.value_optimizer.apply_gradients(
            zip(value_grads, self.value_network.trainable_variables)
        )

        metrics.update({
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss)
        })

        return metrics

    def save_weights(self, path: str):
        """Save network weights.

        Args:
            path: Path to save weights
        """
        self.goal_network.save_weights(f"{path}_goal")
        self.policy_network.save_weights(f"{path}_policy")
        self.value_network.save_weights(f"{path}_value")

    def load_weights(self, path: str):
        """Load network weights.

        Args:
            path: Path to load weights
        """
        self.goal_network.load_weights(f"{path}_goal")
        self.policy_network.load_weights(f"{path}_policy")
        self.value_network.load_weights(f"{path}_value")
