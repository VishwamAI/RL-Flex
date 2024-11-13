import tensorflow as tf
from typing import Dict, List, Tuple, Any, Optional
from .utils import get_device_strategy, create_optimizer
from .q_network import QNetwork

class AdvancedRLAgent:
    """Advanced Reinforcement Learning Agent with multiple enhancements."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        use_double_q: bool = True,
        use_dueling: bool = True,
        use_noisy: bool = True,
        use_per: bool = True,
        use_n_step: bool = True,
        n_step: int = 3,
        use_distributional: bool = True,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0
    ):
        """Initialize AdvancedRLAgent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            tau: Target network update rate
            use_double_q: Whether to use double Q-learning
            use_dueling: Whether to use dueling architecture
            use_noisy: Whether to use noisy networks
            use_per: Whether to use prioritized experience replay
            use_n_step: Whether to use n-step returns
            n_step: Number of steps for n-step returns
            use_distributional: Whether to use distributional RL
            num_atoms: Number of atoms for distributional RL
            v_min: Minimum value for distributional RL
            v_max: Maximum value for distributional RL
        """
        self.gamma = gamma
        self.tau = tau
        self.use_double_q = use_double_q
        self.use_per = use_per
        self.use_n_step = use_n_step
        self.n_step = n_step
        self.use_distributional = use_distributional

        # Get device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create online and target Q-networks
            self.online_q = QNetwork(
                state_dim,
                action_dim,
                hidden_dims,
                learning_rate,
                dueling=use_dueling,
                noisy=use_noisy,
                distributional=use_distributional,
                num_atoms=num_atoms,
                v_min=v_min,
                v_max=v_max
            )
            self.target_q = QNetwork(
                state_dim,
                action_dim,
                hidden_dims,
                learning_rate,
                dueling=use_dueling,
                noisy=use_noisy,
                distributional=use_distributional,
                num_atoms=num_atoms,
                v_min=v_min,
                v_max=v_max
            )

            # Create optimizer
            self.optimizer = create_optimizer(learning_rate)

            # Initialize target network
            self.update_target(tau=1.0)

    def get_action(
        self,
        state: tf.Tensor,
        deterministic: bool = False
    ) -> tf.Tensor:
        """Get action from policy.

        Args:
            state: Current state
            deterministic: Whether to use deterministic action

        Returns:
            Selected action
        """
        state = tf.expand_dims(state, 0)
        q_values = self.online_q(state)

        if self.use_distributional:
            # For distributional RL, compute expected values
            q_values = tf.reduce_sum(
                q_values * self.online_q.support,
                axis=-1
            )

        if deterministic:
            return tf.argmax(q_values[0])
        else:
            if self.online_q.noisy:
                # For noisy nets, no need for epsilon-greedy
                return tf.argmax(q_values[0])
            else:
                # Epsilon-greedy exploration
                if tf.random.uniform(()) < 0.05:
                    return tf.random.uniform(
                        (),
                        0,
                        q_values.shape[-1],
                        dtype=tf.int64
                    )
                else:
                    return tf.argmax(q_values[0])

    def update(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor,
        weights: Optional[tf.Tensor] = None
    ) -> Dict[str, float]:
        """Update networks.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            weights: Optional importance sampling weights

        Returns:
            Dictionary of metrics
        """
        with self.strategy.scope():
            with tf.GradientTape() as tape:
                if self.use_distributional:
                    # Distributional Q-learning update
                    current_dist = self.online_q(states)
                    current_dist = tf.gather(
                        current_dist,
                        actions,
                        batch_dims=1
                    )

                    if self.use_double_q:
                        next_actions = tf.argmax(
                            tf.reduce_sum(
                                self.online_q(next_states) *
                                self.online_q.support,
                                axis=-1
                            ),
                            axis=-1
                        )
                    else:
                        next_actions = tf.argmax(
                            tf.reduce_sum(
                                self.target_q(next_states) *
                                self.target_q.support,
                                axis=-1
                            ),
                            axis=-1
                        )

                    next_dist = self.target_q(next_states)
                    next_dist = tf.gather(
                        next_dist,
                        next_actions,
                        batch_dims=1
                    )

                    # Compute projected distribution
                    tz = rewards[:, None] + (
                        1.0 - dones[:, None]
                    ) * (self.gamma ** self.n_step) * self.target_q.support
                    tz = tf.clip_by_value(
                        tz,
                        self.target_q.v_min,
                        self.target_q.v_max
                    )
                    b = (tz - self.target_q.v_min) / self.target_q.delta_z
                    l = tf.floor(b)
                    u = l + 1
                    l_idx = tf.clip_by_value(
                        l,
                        0,
                        self.target_q.num_atoms - 1
                    )
                    u_idx = tf.clip_by_value(
                        u,
                        0,
                        self.target_q.num_atoms - 1
                    )

                    target_dist = tf.zeros_like(current_dist)
                    for i in range(target_dist.shape[0]):
                        target_dist = tf.tensor_scatter_nd_add(
                            target_dist,
                            tf.stack(
                                [
                                    tf.ones_like(l_idx[i]) * i,
                                    tf.cast(l_idx[i], tf.int32)
                                ],
                                axis=1
                            ),
                            next_dist[i] * (u[i] - b[i])
                        )
                        target_dist = tf.tensor_scatter_nd_add(
                            target_dist,
                            tf.stack(
                                [
                                    tf.ones_like(u_idx[i]) * i,
                                    tf.cast(u_idx[i], tf.int32)
                                ],
                                axis=1
                            ),
                            next_dist[i] * (b[i] - l[i])
                        )

                    # Compute cross-entropy loss
                    loss = -tf.reduce_sum(
                        target_dist * tf.math.log(current_dist + 1e-8),
                        axis=-1
                    )
                else:
                    # Regular Q-learning update
                    current_q = self.online_q(states)
                    current_q = tf.gather(
                        current_q,
                        actions,
                        batch_dims=1
                    )

                    if self.use_double_q:
                        next_actions = tf.argmax(
                            self.online_q(next_states),
                            axis=-1
                        )
                        next_q = tf.gather(
                            self.target_q(next_states),
                            next_actions,
                            batch_dims=1
                        )
                    else:
                        next_q = tf.reduce_max(
                            self.target_q(next_states),
                            axis=-1
                        )

                    target_q = rewards + (
                        1.0 - dones
                    ) * (self.gamma ** self.n_step) * next_q
                    loss = tf.square(target_q - current_q)

                if weights is not None:
                    loss = loss * weights
                loss = tf.reduce_mean(loss)

            # Update online network
            gradients = tape.gradient(
                loss,
                self.online_q.trainable_variables
            )
            self.optimizer.apply_gradients(
                zip(gradients, self.online_q.trainable_variables)
            )

            # Update target network
            self.update_target()

            if self.online_q.noisy:
                self.online_q.reset_noise()
                self.target_q.reset_noise()

            return {
                "loss": float(loss),
                "q_mean": float(tf.reduce_mean(current_q))
            }

    def update_target(self, tau: Optional[float] = None):
        """Update target network parameters.

        Args:
            tau: Optional update rate (if None, use self.tau)
        """
        tau = tau if tau is not None else self.tau
        for target, online in zip(
            self.target_q.trainable_variables,
            self.online_q.trainable_variables
        ):
            target.assign(
                target * (1.0 - tau) + online * tau
            )

    def save_weights(self, path: str):
        """Save network weights.

        Args:
            path: Path to save weights
        """
        self.online_q.save_weights(f"{path}_online")
        self.target_q.save_weights(f"{path}_target")

    def load_weights(self, path: str):
        """Load network weights.

        Args:
            path: Path to load weights
        """
        self.online_q.load_weights(f"{path}_online")
        self.target_q.load_weights(f"{path}_target")
