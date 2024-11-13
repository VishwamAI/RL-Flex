import pytest
import tensorflow as tf
import numpy as np
from RL_Developments.Tensorflow.deep_rl_algorithms import DQNAgent, PPOAgent, ActorCritic

def test_actor_critic_network():
    """Test ActorCritic network architecture and outputs."""
    state_dim = 4
    action_dim = 2
    batch_size = 32

    # Initialize network
    ac_network = ActorCritic(state_dim, action_dim)

    # Create sample input
    states = tf.random.normal((batch_size, state_dim))

    # Get outputs
    action_probs, values = ac_network(states)

    # Check shapes and values
    assert action_probs.shape == (batch_size, action_dim)
    assert values.shape == (batch_size,)
    assert tf.reduce_all(tf.greater_equal(action_probs, 0.0))
    assert tf.reduce_all(tf.less_equal(action_probs, 1.0))
    assert tf.reduce_all(tf.abs(tf.reduce_sum(action_probs, axis=1) - 1.0) < 1e-6)

def test_ppo_agent():
    """Test PPO agent initialization and basic operations."""
    state_dim = 4
    action_dim = 2

    # Initialize agent
    agent = PPOAgent(state_dim, action_dim)

    # Test action selection
    state = tf.random.normal((state_dim,))
    action, probs = agent.get_action(state)

    # Check action and probabilities
    assert isinstance(action, int)
    assert 0 <= action < action_dim
    assert probs.shape == (action_dim,)
    assert tf.reduce_all(tf.greater_equal(probs, 0.0))
    assert tf.reduce_all(tf.less_equal(probs, 1.0))

def test_dqn_agent():
    """Test DQN agent initialization and basic operations."""
    state_dim = 4
    action_dim = 2

    # Initialize agent
    agent = DQNAgent(state_dim, action_dim)

    # Test action selection
    state = tf.random.normal((state_dim,))
    action = agent.get_action(state, training=False)

    # Check action
    assert isinstance(action, int)
    assert 0 <= action < action_dim
