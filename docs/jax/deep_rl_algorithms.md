"""
Module implementing Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) agents using JAX, Flax, and Optax.

Classes:
- DQNAgent: Agent that uses a DQN for learning optimal policies in a reinforcement learning environment.
- DQNetwork: Neural network model used by DQNAgent to approximate Q-values.
- PPOAgent: Agent that implements the PPO algorithm for learning policies.
- ActorCritic: Neural network model used by PPOAgent, combining actor and critic networks.
"""

# Documentation for DQNAgent
class DQNAgent:
    """
    Deep Q-Network Agent.

    Attributes:
    - state_dim (int): Dimension of the input state space.
    - action_dim (int): Number of possible actions.
    - network: Instance of DQNetwork for approximating Q-values.
    - params: Parameters of the neural network.
    - optimizer: Optimization algorithm (Optax optimizer).
    - opt_state: Current state of the optimizer.

    Methods:
    - get_action(state): Returns the action with the highest Q-value for a given state.
    - update(state, action, reward, next_state, done): Updates the network parameters based on the observed transition.
    - _loss_fn(params, state, action, reward, next_state, done): Computes the loss function for training.
    """

# Documentation for DQNetwork
class DQNetwork(nn.Module):
    """
    Neural network model for approximating Q-values in DQN.

    Attributes:
    - action_dim (int): Number of possible actions.

    Methods:
    - __call__(x): Defines the forward pass of the network.
    """

# Documentation for PPOAgent
class PPOAgent:
    """
    Proximal Policy Optimization Agent.

    Attributes:
    - state_dim (int): Dimension of the input state space.
    - action_dim (int): Number of possible actions.
    - network: Instance of ActorCritic model combining policy and value networks.
    - params: Parameters of the neural network.
    - optimizer: Optimization algorithm (Optax optimizer).
    - opt_state: Current state of the optimizer.

    Methods:
    - get_action(state): Samples an action based on the current policy.
    - update(states, actions, rewards, dones, old_log_probs): Updates the network parameters using PPO loss.
    - _loss_fn(params, states, actions, rewards, dones, old_log_probs): Computes the PPO loss function.
    """

# Documentation for ActorCritic
class ActorCritic(nn.Module):
    """
    Actor-Critic neural network model.

    Attributes:
    - action_dim (int): Number of possible actions.

    Methods:
    - __call__(x): Performs a forward pass returning action probabilities and state value estimate.
    """