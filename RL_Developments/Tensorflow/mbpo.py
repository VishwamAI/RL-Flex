import tensorflow as tf
from typing import Dict, List, Tuple, Any
from .utils import get_device_strategy, create_optimizer
from .model_based import WorldModel

class MBPOAgent:
    """Model-Based Policy Optimization Agent implementation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        horizon: int = 1,
        num_policy_updates: int = 20,
        policy_learning_rate: float = 3e-4,
        value_learning_rate: float = 3e-4
    ):
        """Initialize MBPO Agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            ensemble_size: Number of models in ensemble
            horizon: Planning horizon
            num_policy_updates: Number of policy updates per iteration
            policy_learning_rate: Learning rate for policy network
            value_learning_rate: Learning rate for value network
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_policy_updates = num_policy_updates

        # Get device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create world model
            self.world_model = WorldModel(
                state_dim,
                action_dim,
                ensemble_size
            )

            # Create policy network
            self.policy_network = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(action_dim, activation='tanh')
            ])

            # Create value network
            self.value_network = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)
            ])

            # Create optimizers
            self.policy_optimizer = create_optimizer(policy_learning_rate)
            self.value_optimizer = create_optimizer(value_learning_rate)

            # Build networks
            dummy_state = tf.zeros((1, state_dim))
            self.policy_network(dummy_state)
            self.value_network(dummy_state)

    def get_action(self, state: tf.Tensor) -> tf.Tensor:
        """Get action from policy network.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        state = tf.expand_dims(state, 0)
        return self.policy_network(state)[0]

    def imagine_trajectories(
        self,
        initial_states: tf.Tensor,
        num_trajectories: int
    ) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
        """Imagine trajectories using world model.

        Args:
            initial_states: Initial states
            num_trajectories: Number of trajectories to imagine

        Returns:
            Tuple of (states, actions, rewards)
        """
        batch_size = tf.shape(initial_states)[0]
        states = [initial_states]
        actions = []
        rewards = []

        for _ in range(self.horizon):
            # Get actions from policy
            current_states = states[-1]
            current_actions = self.policy_network(current_states)
            actions.append(current_actions)

            # Predict next states and rewards
            next_states, step_rewards, _ = self.world_model(
                current_states,
                current_actions
            )

            states.append(next_states)
            rewards.append(step_rewards)

        return states, actions, rewards

    def update(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor
    ) -> Dict[str, float]:
        """Update agent's networks.

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

        with self.strategy.scope():
            # Update world model
            model_metrics = self.world_model.update(
                states, actions, next_states, rewards
            )
            metrics.update(model_metrics)

            # Imagine trajectories
            imagined_states, imagined_actions, imagined_rewards = \
                self.imagine_trajectories(states, num_trajectories=10)

            # Update policy
            for _ in range(self.num_policy_updates):
                with tf.GradientTape() as tape:
                    # Compute predicted values
                    values = []
                    for s in imagined_states:
                        v = self.value_network(s)
                        values.append(v)

                    # Compute returns
                    returns = []
                    next_value = values[-1]
                    for r in reversed(imagined_rewards):
                        next_value = r + 0.99 * next_value
                        returns.insert(0, next_value)

                    # Compute policy loss
                    policy_loss = 0.0
                    for s, a, r in zip(
                        imagined_states[:-1],
                        imagined_actions,
                        returns
                    ):
                        pred_actions = self.policy_network(s)
                        policy_loss += tf.reduce_mean(
                            tf.square(pred_actions - a)
                        )

                    policy_loss /= len(imagined_states) - 1

                # Update policy
                policy_grads = tape.gradient(
                    policy_loss,
                    self.policy_network.trainable_variables
                )
                self.policy_optimizer.apply_gradients(
                    zip(
                        policy_grads,
                        self.policy_network.trainable_variables
                    )
                )

                metrics['policy_loss'] = float(policy_loss)

            # Update value network
            with tf.GradientTape() as tape:
                # Compute value predictions
                value_preds = self.value_network(states)
                next_values = self.value_network(next_states)

                # Compute targets
                targets = rewards + 0.99 * next_values * (1 - dones)

                # Compute value loss
                value_loss = tf.reduce_mean(
                    tf.square(targets - value_preds)
                )

            # Update value network
            value_grads = tape.gradient(
                value_loss,
                self.value_network.trainable_variables
            )
            self.value_optimizer.apply_gradients(
                zip(
                    value_grads,
                    self.value_network.trainable_variables
                )
            )

            metrics['value_loss'] = float(value_loss)

        return metrics
