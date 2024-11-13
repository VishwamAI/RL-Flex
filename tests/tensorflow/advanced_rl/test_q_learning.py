import tensorflow as tf
import numpy as np
from RL_Developments.Tensorflow.q_learning import QLearningAgent, SARSAgent

def test_qlearning_agent_initialization():
    """Test Q-Learning Agent initialization."""
    state_dim = 4
    action_dim = 2

    agent = QLearningAgent(state_dim, action_dim)

    # Verify initialization
    assert agent.state_dim == state_dim
    assert agent.action_dim == action_dim
    assert isinstance(agent.q_table, dict)

def test_qlearning_agent_update():
    """Test Q-Learning Agent update."""
    state_dim = 4
    action_dim = 2

    agent = QLearningAgent(state_dim, action_dim)

    # Test update
    state = tf.zeros(state_dim)
    action = 0
    reward = 1.0
    next_state = tf.ones(state_dim)
    done = False

    metrics = agent.update(state, action, reward, next_state, done)

    assert "td_error" in metrics
    assert "q_value" in metrics

def test_sarsa_agent_initialization():
    """Test SARSA Agent initialization."""
    state_dim = 4
    action_dim = 2

    agent = SARSAgent(state_dim, action_dim)

    # Verify initialization
    assert agent.state_dim == state_dim
    assert agent.action_dim == action_dim
    assert isinstance(agent.q_table, dict)

def test_sarsa_agent_update():
    """Test SARSA Agent update."""
    state_dim = 4
    action_dim = 2

    agent = SARSAgent(state_dim, action_dim)

    # Test update
    state = tf.zeros(state_dim)
    action = 0
    reward = 1.0
    next_state = tf.ones(state_dim)
    next_action = 1
    done = False

    metrics = agent.update(state, action, reward, next_state, next_action, done)

    assert "td_error" in metrics
    assert "q_value" in metrics
