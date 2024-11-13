import tensorflow as tf
import numpy as np
from RL_Developments.Tensorflow.mbpo import MBPOAgent

def test_mbpo_agent_initialization():
    """Test MBPO Agent initialization."""
    state_dim = 4
    action_dim = 2

    agent = MBPOAgent(state_dim, action_dim)

    # Verify network initialization
    assert isinstance(agent.policy_network, tf.keras.Sequential)
    assert isinstance(agent.value_network, tf.keras.Sequential)
    assert isinstance(agent.world_model.dynamics_models[0], tf.keras.Sequential)

def test_mbpo_agent_action_selection():
    """Test MBPO Agent action selection."""
    state_dim = 4
    action_dim = 2

    agent = MBPOAgent(state_dim, action_dim)

    # Test action selection
    state = tf.zeros(state_dim)
    action = agent.get_action(state)

    assert isinstance(action, tf.Tensor)
    assert action.shape == (action_dim,)

def test_mbpo_agent_update():
    """Test MBPO Agent update."""
    state_dim = 4
    action_dim = 2
    batch_size = 32

    agent = MBPOAgent(state_dim, action_dim)

    # Create dummy batch
    states = tf.random.normal((batch_size, state_dim))
    actions = tf.random.normal((batch_size, action_dim))
    rewards = tf.random.normal((batch_size, 1))
    next_states = tf.random.normal((batch_size, state_dim))
    dones = tf.zeros((batch_size, 1))

    # Update agent
    metrics = agent.update(states, actions, rewards, next_states, dones)

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
