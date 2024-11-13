# Advanced Reinforcement Learning Algorithms Module

This module provides implementations of advanced reinforcement learning algorithms, specifically **Soft Actor-Critic (SAC)** and **Twin Delayed DDPG (TD3)**. These algorithms are designed for continuous action spaces and are implemented using **JAX** and **Flax** for efficient computation.

## Key Classes and Functions

### 1. **SACAgent**

The `SACAgent` class represents an agent that follows the **Soft Actor-Critic (SAC)** algorithm. SAC is a popular algorithm for reinforcement learning in continuous action spaces, known for its stability and sample efficiency.

- **Attributes**:
  - `state_dim`: The size of the state space.
  - `action_dim`: The size of the action space.
  - `gamma`: The discount factor used to prioritize future rewards.
  - `tau`: The soft update factor for updating target networks.
  - `alpha`: Coefficient for entropy regularization, encouraging exploration.
  - `actor`: The policy network that suggests actions based on states.
  - `critic1`, `critic2`: Two Q-value networks to estimate the value of state-action pairs.
  - `target_critic1`, `target_critic2`: Target networks for `critic1` and `critic2`, used for stability.
  - Optimizer and parameter attributes to handle model updates effectively.

- **Methods**:
  - `select_action(state)`: Chooses an action based on the current policy.
  - `update(batch)`: Updates the agent's parameters based on experience tuples (state, action, reward, next state, done).
  - `soft_update(source_params, target_params)`: Gradually updates target network parameters to match the main networks.

### 2. **TD3Agent**

The `TD3Agent` class represents an agent following the **Twin Delayed DDPG (TD3)** algorithm. TD3 is an improvement over DDPG, incorporating tricks to reduce overestimation bias and stabilize training in continuous action spaces.

- **Attributes**:
  - Similar to `SACAgent`, with additional attributes:
    - `policy_noise`: Controls the noise added to actions during training.
    - `noise_clip`: Limits the value range for noise, ensuring stability.
    - `policy_freq`: Specifies how often the policy (actor) network is updated.
  - `actor`, `critic1`, `critic2`, and their target counterparts for managing the policy and value estimation.
  - Optimizers for handling model updates.

- **Methods**:
  - `select_action(state)`: Chooses an action based on the current policy.
  - `update(batch, step)`: Updates the agent's parameters based on experience tuples and training step.
  - `soft_update(source_params, target_params)`: Gradually updates target network parameters to match the main networks.

### 3. **Actor Network**

The `Actor` class represents the policy network that outputs actions given states. This is used by both the SAC and TD3 agents.

- **Attributes**:
  - `action_dim`: The size of the action space.
  - `hidden_dim`: Number of units in the hidden layers.
  
- **Methods**:
  - `__call__(x)`: The forward pass, transforming the input state into an action after applying a `tanh` activation for bounding.

### 4. **Critic Network**

The `Critic` class represents the value network that outputs Q-values given states and actions, for use in SAC and TD3 agents.

- **Attributes**:
  - `hidden_dim`: Number of units in the hidden layers.
  
- **Methods**:
  - `__call__(state, action)`: The forward pass, which estimates the Q-value for a given state-action pair.

### 5. **Helper Functions**

- `create_sac_agent(state_dim, action_dim, **kwargs)`: Initializes and returns an `SACAgent`.
- `create_td3_agent(state_dim, action_dim, **kwargs)`: Initializes and returns a `TD3Agent`.

### Summary

This module simplifies the process of creating and managing advanced RL agents for continuous action tasks. Both SAC and TD3 are designed to be effective in high-dimensional and complex environments, making them suitable for a wide range of reinforcement learning applications.



code explanation

"""
Advanced Reinforcement Learning Algorithms Module

This module implements advanced reinforcement learning algorithms,
including Soft Actor-Critic (SAC) and Twin Delayed DDPG (TD3).
"""

class SACAgent:
    """Soft Actor-Critic (SAC) agent implementation.

    Implements the SAC algorithm for continuous action spaces using JAX and Flax.

    Attributes:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        gamma (float): Discount factor for future rewards.
        tau (float): Soft update coefficient for target networks.
        alpha (float): Entropy regularization coefficient.
        actor (Actor): Policy network.
        critic1 (Critic): First Q-value network.
        critic2 (Critic): Second Q-value network.
        target_critic1 (Critic): Target network for the first critic.
        target_critic2 (Critic): Target network for the second critic.
        actor_params (dict): Parameters of the actor network.
        critic1_params (dict): Parameters of the first critic network.
        critic2_params (dict): Parameters of the second critic network.
        target_critic1_params (dict): Parameters of the target first critic network.
        target_critic2_params (dict): Parameters of the target second critic network.
        actor_optimizer (optax.GradientTransformation): Optimizer for the actor network.
        critic_optimizer (optax.GradientTransformation): Optimizer for the critic networks.
        actor_opt_state (optax.OptState): Optimizer state for the actor network.
        critic1_opt_state (optax.OptState): Optimizer state for the first critic network.
        critic2_opt_state (optax.OptState): Optimizer state for the second critic network.
    """

    def select_action(self, state: jnp.ndarray) -> jnp.ndarray:
        """Select an action using the current policy.

        Args:
            state (jnp.ndarray): Current state.

        Returns:
            jnp.ndarray: Action to be taken.
        """

    def update(self, batch: Tuple[jnp.ndarray, ...]) -> Tuple[float, float]:
        """Update the SAC agent using a batch of experiences.

        Args:
            batch (Tuple[jnp.ndarray, ...]): Batch of transitions (states, actions, rewards, next_states, dones).

        Returns:
            Tuple[float, float]: Tuple containing critic loss and actor loss.
        """

    def soft_update(self, source_params: dict, target_params: dict) -> dict:
        """Soft update of target network parameters.

        Args:
            source_params (dict): Source network parameters.
            target_params (dict): Target network parameters.

        Returns:
            dict: Updated target network parameters.
        """

class TD3Agent:
    """Twin Delayed DDPG (TD3) agent implementation.

    Implements the TD3 algorithm for continuous action spaces using JAX and Flax.

    Attributes:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        gamma (float): Discount factor for future rewards.
        tau (float): Soft update coefficient for target networks.
        policy_noise (float): Standard deviation of the noise added to target actions.
        noise_clip (float): Maximum absolute value for the target action noise.
        policy_freq (int): Frequency of policy updates.
        actor (Actor): Policy network.
        critic1 (Critic): First Q-value network.
        critic2 (Critic): Second Q-value network.
        target_actor (Actor): Target policy network.
        target_critic1 (Critic): Target network for the first critic.
        target_critic2 (Critic): Target network for the second critic.
        actor_params (dict): Parameters of the actor network.
        critic1_params (dict): Parameters of the first critic network.
        critic2_params (dict): Parameters of the second critic network.
        target_actor_params (dict): Parameters of the target actor network.
        target_critic1_params (dict): Parameters of the target first critic network.
        target_critic2_params (dict): Parameters of the target second critic network.
        actor_optimizer (optax.GradientTransformation): Optimizer for the actor network.
        critic_optimizer (optax.GradientTransformation): Optimizer for the critic networks.
        actor_opt_state (optax.OptState): Optimizer state for the actor network.
        critic1_opt_state (optax.OptState): Optimizer state for the first critic network.
        critic2_opt_state (optax.OptState): Optimizer state for the second critic network.
    """

    def select_action(self, state: jnp.ndarray) -> jnp.ndarray:
        """Select an action using the current policy.

        Args:
            state (jnp.ndarray): Current state.

        Returns:
            jnp.ndarray: Action to be taken.
        """

    def update(self, batch: Tuple[jnp.ndarray, ...], step: int) -> Tuple[float, float]:
        """Update the TD3 agent using a batch of experiences.

        Args:
            batch (Tuple[jnp.ndarray, ...]): Batch of transitions (states, actions, rewards, next_states, dones).
            step (int): Current training step.

        Returns:
            Tuple[float, float]: Tuple containing critic loss and actor loss.
        """

    def soft_update(self, source_params: dict, target_params: dict) -> dict:
        """Soft update of target network parameters.

        Args:
            source_params (dict): Source network parameters.
            target_params (dict): Target network parameters.

        Returns:
            dict: Updated target network parameters.
        """

class Actor(nn.Module):
    """Actor network for both SAC and TD3 agents.

    Neural network that outputs actions given states.

    Attributes:
        action_dim (int): Dimension of the action space.
        hidden_dim (int): Dimension of the hidden layers.
    """

    def __call__(self, x):
        """Forward pass of the actor network.

        Args:
            x (jnp.ndarray): Input state.

        Returns:
            jnp.ndarray: Action output after applying a tanh activation.
        """

class Critic(nn.Module):
    """Critic network for both SAC and TD3 agents.

    Neural network that outputs Q-values given states and actions.

    Attributes:
        hidden_dim (int): Dimension of the hidden layers.
    """

    def __call__(self, state, action):
        """Forward pass of the critic network.

        Args:
            state (jnp.ndarray): Input state.
            action (jnp.ndarray): Input action.

        Returns:
            jnp.ndarray: Estimated Q-value.
        """

def create_sac_agent(state_dim: int, action_dim: int, **kwargs) -> SACAgent:
    """Create a Soft Actor-Critic (SAC) agent.

    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        **kwargs: Additional keyword arguments for SACAgent.

    Returns:
        SACAgent: Initialized SAC agent.
    """

def create_td3_agent(state_dim: int, action_dim: int, **kwargs) -> TD3Agent:
    """Create a Twin Delayed DDPG (TD3) agent.

    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        **kwargs: Additional keyword arguments for TD3Agent.

    Returns:
        TD3Agent: Initialized TD3 agent.
    """