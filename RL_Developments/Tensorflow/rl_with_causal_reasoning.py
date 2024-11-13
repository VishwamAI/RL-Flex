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

class CausalGraph(tf.keras.Model):
    """Causal graph implementation for reinforcement learning."""

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        """Initialize causal graph.

        Args:
            state_dim: Dimension of state space
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        self.state_dim = state_dim

        # Create structural equation model
        self.structural_model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                hidden_dims[0],
                activation='relu',
                input_shape=(state_dim,)
            ),
            tf.keras.layers.Dense(
                hidden_dims[1],
                activation='relu'
            ),
            tf.keras.layers.Dense(state_dim)
        ])

        # Create adjacency matrix for causal relationships
        self.adjacency = tf.Variable(
            tf.random.uniform([state_dim, state_dim], 0, 0.1),
            trainable=True
        )

    def get_causal_effects(
        self,
        states: tf.Tensor,
        training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get causal effects between state variables.

        Args:
            states: Batch of states
            training: Whether in training mode

        Returns:
            Tuple of (predicted next states, adjacency matrix)
        """
        # Apply structural equations
        next_states = self.structural_model(states, training=training)

        # Apply causal relationships
        adjacency = tf.nn.sigmoid(self.adjacency)
        causal_effects = tf.matmul(next_states, adjacency)

        return causal_effects, adjacency

class InterventionModel(tf.keras.Model):
    """Model for performing interventions in causal reasoning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        """Initialize intervention model.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Create intervention network
        self.intervention_network = tf.keras.Sequential([
            tf.keras.layers.Dense(
                hidden_dims[0],
                activation='relu',
                input_shape=(state_dim + action_dim,)
            ),
            tf.keras.layers.Dense(
                hidden_dims[1],
                activation='relu'
            ),
            tf.keras.layers.Dense(state_dim)
        ])

    def intervene(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        """Perform intervention.

        Args:
            states: Batch of states
            actions: Batch of actions
            training: Whether in training mode

        Returns:
            Intervened states
        """
        inputs = tf.concat([states, actions], axis=-1)
        interventions = self.intervention_network(inputs, training=training)
        return states + interventions

class CounterfactualModel(tf.keras.Model):
    """Model for counterfactual reasoning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        """Initialize counterfactual model.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Create counterfactual network
        self.counterfactual_network = tf.keras.Sequential([
            tf.keras.layers.Dense(
                hidden_dims[0],
                activation='relu',
                input_shape=(state_dim * 2 + action_dim,)
            ),
            tf.keras.layers.Dense(
                hidden_dims[1],
                activation='relu'
            ),
            tf.keras.layers.Dense(state_dim)
        ])

    def reason(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        factual_next_states: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        """Perform counterfactual reasoning.

        Args:
            states: Batch of states
            actions: Batch of actions
            factual_next_states: Batch of actual next states
            training: Whether in training mode

        Returns:
            Counterfactual states
        """
        inputs = tf.concat([states, actions, factual_next_states], axis=-1)
        return self.counterfactual_network(inputs, training=training)

class RLWithCausalReasoning(tf.keras.Model):
    """Reinforcement Learning with Causal Reasoning implementation.

    This class implements RL with causal reasoning capabilities,
    including causal graph learning, interventions, and
    counterfactual reasoning.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
        **kwargs
    ):
        """Initialize RLWithCausalReasoning.

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
            # Create causal reasoning components
            self.causal_graph = CausalGraph(state_dim, hidden_dims)
            self.intervention_model = InterventionModel(
                state_dim,
                action_dim,
                hidden_dims
            )
            self.counterfactual_model = CounterfactualModel(
                state_dim,
                action_dim,
                hidden_dims
            )

            # Create policy network
            self.policy = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    hidden_dims[0],
                    activation='relu',
                    input_shape=(state_dim,)
                ),
                tf.keras.layers.Dense(
                    hidden_dims[1],
                    activation='relu'
                ),
                tf.keras.layers.Dense(action_dim * 2)  # Mean and log_std
            ])

            # Create value network
            self.value = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    hidden_dims[0],
                    activation='relu',
                    input_shape=(state_dim,)
                ),
                tf.keras.layers.Dense(
                    hidden_dims[1],
                    activation='relu'
                ),
                tf.keras.layers.Dense(1)
            ])

            # Create optimizers
            self.causal_optimizer = create_optimizer(learning_rate)
            self.intervention_optimizer = create_optimizer(learning_rate)
            self.counterfactual_optimizer = create_optimizer(learning_rate)
            self.policy_optimizer = create_optimizer(learning_rate)
            self.value_optimizer = create_optimizer(learning_rate)

    def get_action(
        self,
        states: tf.Tensor,
        training: bool = False
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Get actions from policy network.

        Args:
            states: Batch of states
            training: Whether in training mode

        Returns:
            Tuple of (actions, additional info)
        """
        outputs = self.policy(states, training=training)
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
            "stds": stds
        }

    def update(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor
    ) -> Dict[str, float]:
        """Update networks using causal reasoning.

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

        # Update causal graph
        with tf.GradientTape() as tape:
            causal_effects, adjacency = self.causal_graph.get_causal_effects(
                states,
                training=True
            )
            causal_loss = tf.reduce_mean(
                tf.square(causal_effects - next_states)
            )
            # Add sparsity regularization for adjacency matrix
            sparsity_reg = tf.reduce_mean(tf.abs(adjacency))
            total_causal_loss = causal_loss + 0.01 * sparsity_reg

        causal_grads = tape.gradient(
            total_causal_loss,
            self.causal_graph.trainable_variables
        )
        self.causal_optimizer.apply_gradients(
            zip(causal_grads, self.causal_graph.trainable_variables)
        )

        metrics["causal_loss"] = float(causal_loss)
        metrics["sparsity_reg"] = float(sparsity_reg)

        # Update intervention model
        with tf.GradientTape() as tape:
            intervened_states = self.intervention_model.intervene(
                states,
                actions,
                training=True
            )
            intervention_loss = tf.reduce_mean(
                tf.square(intervened_states - next_states)
            )

        intervention_grads = tape.gradient(
            intervention_loss,
            self.intervention_model.trainable_variables
        )
        self.intervention_optimizer.apply_gradients(
            zip(
                intervention_grads,
                self.intervention_model.trainable_variables
            )
        )


        metrics["intervention_loss"] = float(intervention_loss)

        # Update counterfactual model
        with tf.GradientTape() as tape:
            counterfactual_states = self.counterfactual_model.reason(
                states,
                actions,
                next_states,
                training=True
            )
            # Use causal effects as target for counterfactual reasoning
            causal_effects, _ = self.causal_graph.get_causal_effects(
                states,
                training=False
            )
            counterfactual_loss = tf.reduce_mean(
                tf.square(counterfactual_states - causal_effects)
            )

        counterfactual_grads = tape.gradient(
            counterfactual_loss,
            self.counterfactual_model.trainable_variables
        )
        self.counterfactual_optimizer.apply_gradients(
            zip(
                counterfactual_grads,
                self.counterfactual_model.trainable_variables
            )
        )

        metrics["counterfactual_loss"] = float(counterfactual_loss)

        # Update policy
        with tf.GradientTape() as tape:
            actions, action_info = self.get_action(states, training=True)
            values = self.value(states, training=True)
            next_values = self.value(next_states, training=True)

            # Compute advantages using causal effects
            causal_effects, _ = self.causal_graph.get_causal_effects(
                states,
                training=False
            )
            causal_values = self.value(causal_effects, training=False)
            advantages = rewards + 0.99 * (1 - dones) * next_values - values
            causal_advantages = (
                rewards +
                0.99 * (1 - dones) * causal_values -
                values
            )

            # Compute policy loss with causal advantages
            means = action_info["means"]
            stds = action_info["stds"]
            log_probs = -0.5 * (
                tf.square((actions - means) / stds) +
                2 * tf.math.log(stds) +
                tf.math.log(2 * np.pi)
            )
            log_probs = tf.reduce_sum(log_probs, axis=-1)
            policy_loss = -tf.reduce_mean(
                log_probs * (advantages + 0.5 * causal_advantages)
            )

        policy_grads = tape.gradient(
            policy_loss,
            self.policy.trainable_variables
        )
        self.policy_optimizer.apply_gradients(
            zip(policy_grads, self.policy.trainable_variables)
        )

        metrics["policy_loss"] = float(policy_loss)

        # Update value function
        with tf.GradientTape() as tape:
            values = self.value(states, training=True)
            value_targets = rewards + 0.99 * (1 - dones) * next_values
            value_loss = tf.reduce_mean(tf.square(value_targets - values))

        value_grads = tape.gradient(
            value_loss,
            self.value.trainable_variables
        )
        self.value_optimizer.apply_gradients(
            zip(value_grads, self.value.trainable_variables)
        )

        metrics["value_loss"] = float(value_loss)

        return metrics

    def save_weights(self, path: str):
        """Save network weights.

        Args:
            path: Path to save weights
        """
        self.causal_graph.save_weights(f"{path}_causal_graph")
        self.intervention_model.save_weights(f"{path}_intervention")
        self.counterfactual_model.save_weights(f"{path}_counterfactual")
        self.policy.save_weights(f"{path}_policy")
        self.value.save_weights(f"{path}_value")

    def load_weights(self, path: str):
        """Load network weights.

        Args:
            path: Path to load weights
        """
        self.causal_graph.load_weights(f"{path}_causal_graph")
        self.intervention_model.load_weights(f"{path}_intervention")
        self.counterfactual_model.load_weights(f"{path}_counterfactual")
        self.policy.load_weights(f"{path}_policy")
        self.value.load_weights(f"{path}_value")
