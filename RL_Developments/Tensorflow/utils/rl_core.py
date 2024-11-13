import tensorflow as tf
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from .device_utils import get_device_strategy

class Actor(tf.keras.Model):
    """Generic Actor network for policy-based methods."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = 'relu',
        continuous: bool = False
    ):
        """Initialize Actor.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            continuous: Whether action space is continuous
        """
        super().__init__()

        layers = []
        current_dim = state_dim
        for dim in hidden_dims:
            layers.extend([
                tf.keras.layers.Dense(dim),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Activation(activation)
            ])
            current_dim = dim

        if continuous:
            # For continuous actions, output mean and log_std
            self.mean_layer = tf.keras.layers.Dense(action_dim)
            self.log_std_layer = tf.keras.layers.Dense(action_dim)
        else:
            # For discrete actions, output logits
            layers.append(tf.keras.layers.Dense(action_dim))

        self.network = tf.keras.Sequential(layers)
        self.continuous = continuous

    def call(self, states: tf.Tensor) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """Forward pass.

        Args:
            states: Batch of states

        Returns:
            Action logits or (mean, log_std) for continuous actions
        """
        if self.continuous:
            features = self.network(states)
            mean = self.mean_layer(features)
            log_std = self.log_std_layer(features)
            return mean, log_std
        else:
            return self.network(states)

class Critic(tf.keras.Model):
    """Generic Critic network for value-based methods."""

    def __init__(
        self,
        state_dim: int,
        action_dim: Optional[int] = None,
        hidden_dims: List[int] = [256, 256],
        activation: str = 'relu'
    ):
        """Initialize Critic.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (if action-value function)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
        """
        super().__init__()

        layers = []
        current_dim = state_dim + (action_dim if action_dim else 0)
        for dim in hidden_dims:
            layers.extend([
                tf.keras.layers.Dense(dim),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Activation(activation)
            ])
            current_dim = dim

        layers.append(tf.keras.layers.Dense(1))
        self.network = tf.keras.Sequential(layers)

    def call(
        self,
        states: tf.Tensor,
        actions: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """Forward pass.

        Args:
            states: Batch of states
            actions: Optional batch of actions

        Returns:
            Value estimates
        """
        if actions is not None:
            x = tf.concat([states, actions], axis=-1)
        else:
            x = states
        return self.network(x)

class RLAgent:
    """Base class for RL agents."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99
    ):
        """Initialize RLAgent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate
            gamma: Discount factor
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Get device strategy
        self.strategy = get_device_strategy()

    def get_action(self, state: tf.Tensor) -> tf.Tensor:
        """Get action from policy.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        raise NotImplementedError

    def update(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor
    ) -> Dict[str, float]:
        """Update agent.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags

        Returns:
            Dictionary of metrics
        """
        raise NotImplementedError

def select_action(
    policy: tf.keras.Model,
    state: tf.Tensor,
    deterministic: bool = False
) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
    """Select action from policy.

    Args:
        policy: Policy network
        state: Current state
        deterministic: Whether to use deterministic action

    Returns:
        Tuple of (action, log probability)
    """
    state = tf.expand_dims(state, 0)
    policy_output = policy(state)

    if isinstance(policy_output, tuple):
        # Continuous actions
        mean, log_std = policy_output
        if deterministic:
            return mean[0], None
        else:
            std = tf.exp(log_std)
            action = mean + tf.random.normal(tf.shape(mean)) * std
            log_prob = -0.5 * (
                tf.square((action - mean) / std) +
                2 * log_std +
                tf.math.log(2 * np.pi)
            )
            return action[0], log_prob[0]
    else:
        # Discrete actions
        logits = policy_output[0]
        if deterministic:
            return tf.argmax(logits), None
        else:
            action = tf.random.categorical(
                tf.expand_dims(logits, 0),
                1
            )[0, 0]
            log_prob = tf.nn.log_softmax(logits)[action]
            return action, log_prob

def update_ppo(
    policy: tf.keras.Model,
    value: tf.keras.Model,
    policy_optimizer: tf.keras.optimizers.Optimizer,
    value_optimizer: tf.keras.optimizers.Optimizer,
    states: tf.Tensor,
    actions: tf.Tensor,
    advantages: tf.Tensor,
    returns: tf.Tensor,
    old_log_probs: tf.Tensor,
    clip_ratio: float = 0.2
) -> Dict[str, float]:
    """Update policy and value networks using PPO.

    Args:
        policy: Policy network
        value: Value network
        policy_optimizer: Policy optimizer
        value_optimizer: Value optimizer
        states: Batch of states
        actions: Batch of actions
        advantages: Batch of advantages
        returns: Batch of returns
        old_log_probs: Batch of old log probabilities
        clip_ratio: PPO clip ratio

    Returns:
        Dictionary of metrics
    """
    with tf.GradientTape() as tape:
        # Get current policy distribution
        policy_output = policy(states)
        if isinstance(policy_output, tuple):
            # Continuous actions
            mean, log_std = policy_output
            std = tf.exp(log_std)
            dist = tfp.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions)
        else:
            # Discrete actions
            logits = policy_output
            log_probs = tf.reduce_sum(
                tf.one_hot(actions, logits.shape[-1]) *
                tf.nn.log_softmax(logits),
                axis=-1
            )

        # Compute ratio and clipped ratio
        ratio = tf.exp(log_probs - old_log_probs)
        clipped_ratio = tf.clip_by_value(
            ratio,
            1 - clip_ratio,
            1 + clip_ratio
        )

        # Compute policy loss
        policy_loss = -tf.reduce_mean(
            tf.minimum(
                ratio * advantages,
                clipped_ratio * advantages
            )
        )

    # Update policy
    policy_grads = tape.gradient(
        policy_loss,
        policy.trainable_variables
    )
    policy_optimizer.apply_gradients(
        zip(policy_grads, policy.trainable_variables)
    )

    # Update value function
    with tf.GradientTape() as tape:
        values = value(states)
        value_loss = tf.reduce_mean(tf.square(returns - values))

    value_grads = tape.gradient(
        value_loss,
        value.trainable_variables
    )
    value_optimizer.apply_gradients(
        zip(value_grads, value.trainable_variables)
    )

    return {
        "policy_loss": float(policy_loss),
        "value_loss": float(value_loss),
        "ratio_mean": float(tf.reduce_mean(ratio))
    }

def train_rl_agent(
    agent: RLAgent,
    env: Any,
    num_episodes: int,
    max_steps: int = 1000,
    eval_interval: int = 10
) -> Dict[str, List[float]]:
    """Train RL agent.

    Args:
        agent: RL agent
        env: Environment
        num_episodes: Number of episodes
        max_steps: Maximum steps per episode
        eval_interval: Evaluation interval

    Returns:
        Dictionary of training metrics
    """
    metrics = {
        "episode_returns": [],
        "episode_lengths": []
    }

    for episode in range(num_episodes):
        state = env.reset()
        episode_return = 0
        episode_length = 0

        for step in range(max_steps):
            # Get action
            action = agent.get_action(state)

            # Take step
            next_state, reward, done, _ = env.step(action)

            # Update agent
            update_metrics = agent.update(
                tf.expand_dims(state, 0),
                tf.expand_dims(action, 0),
                tf.expand_dims(reward, 0),
                tf.expand_dims(next_state, 0),
                tf.expand_dims(done, 0)
            )

            episode_return += reward
            episode_length += 1
            state = next_state

            if done:
                break


        metrics["episode_returns"].append(episode_return)
        metrics["episode_lengths"].append(episode_length)

        # Evaluate agent
        if (episode + 1) % eval_interval == 0:
            eval_return = 0
            state = env.reset()
            for _ in range(max_steps):
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                eval_return += reward
                state = next_state
                if done:
                    break
            metrics.setdefault("eval_returns", []).append(eval_return)

    return metrics
