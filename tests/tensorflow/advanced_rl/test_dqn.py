import tensorflow as tf
import numpy as np
from RL_Developments.Tensorflow.dqn import DQNAgent, QNetwork

def test_qnetwork_initialization():
    """Test Q-Network initialization and forward pass."""
    state_dim = 4
    action_dim = 2

    network = QNetwork(state_dim, action_dim)

    # Test forward pass
    batch_size = 32
    states = tf.random.normal((batch_size, state_dim))
    q_values = network(states)

    assert q_values.shape == (batch_size, action_dim)

def test_dqn_agent_initialization():
    """Test DQN Agent initialization."""
    state_dim = 4
    action_dim = 2

    agent = DQNAgent(state_dim, action_dim)

    # Verify network initialization
    assert isinstance(agent.q_network, QNetwork)
    assert isinstance(agent.target_network, QNetwork)

def test_dqn_agent_update():
    """Test DQN Agent update."""
    state_dim = 4
    action_dim = 2
    batch_size = 32

    agent = DQNAgent(state_dim, action_dim)

    # Create dummy batch
    states = tf.random.normal((batch_size, state_dim))
    actions = tf.random.uniform((batch_size,), 0, action_dim, dtype=tf.int32)
    rewards = tf.random.normal((batch_size,))
    next_states = tf.random.normal((batch_size, state_dim))
    dones = tf.zeros((batch_size,))

    # Update agent
    metrics = agent.update(states, actions, rewards, next_states, dones)

    assert "q_loss" in metrics
    assert "q_value_mean" in metrics
