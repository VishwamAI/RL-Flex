import tensorflow as tf
import numpy as np
import pytest
from RL_Developments.Tensorflow.model_free_rl import QLearningAgent, SARSAgent

@pytest.fixture
def state_dim():
    return 4

@pytest.fixture
def action_dim():
    return 2

@pytest.fixture
def q_learning_agent(state_dim, action_dim):
    return QLearningAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[32, 32],
        learning_rate=0.001
    )

@pytest.fixture
def sarsa_agent(state_dim, action_dim):
    return SARSAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[32, 32],
        learning_rate=0.001
    )

def test_q_learning_initialization(q_learning_agent, state_dim, action_dim):
    assert q_learning_agent.state_dim == state_dim
    assert q_learning_agent.action_dim == action_dim
    assert isinstance(q_learning_agent.q_network, tf.keras.Model)

def test_sarsa_initialization(sarsa_agent, state_dim, action_dim):
    assert sarsa_agent.state_dim == state_dim
    assert sarsa_agent.action_dim == action_dim
    assert isinstance(sarsa_agent.q_network, tf.keras.Model)

def test_q_learning_get_action(q_learning_agent, state_dim):
    state = tf.random.normal((state_dim,))
    action = q_learning_agent.get_action(state, training=False)
    assert isinstance(action, int)
    assert 0 <= action < q_learning_agent.action_dim

def test_sarsa_get_action(sarsa_agent, state_dim):
    state = tf.random.normal((state_dim,))
    action = sarsa_agent.get_action(state, training=False)
    assert isinstance(action, int)
    assert 0 <= action < sarsa_agent.action_dim

def test_q_learning_update(q_learning_agent, state_dim):
    state = tf.random.normal((state_dim,))
    next_state = tf.random.normal((state_dim,))
    action = 0
    reward = 1.0
    done = False

    metrics = q_learning_agent.update(state, action, reward, next_state, done)
    assert isinstance(metrics, dict)
    assert 'loss' in metrics
    assert 'epsilon' in metrics

def test_sarsa_update(sarsa_agent, state_dim):
    state = tf.random.normal((state_dim,))
    next_state = tf.random.normal((state_dim,))
    action = 0
    next_action = 1
    reward = 1.0
    done = False

    metrics = sarsa_agent.update(state, action, reward, next_state, next_action, done)
    assert isinstance(metrics, dict)
    assert 'loss' in metrics
    assert 'epsilon' in metrics

def test_q_learning_device_strategy(q_learning_agent):
    assert hasattr(q_learning_agent, 'strategy')
    assert isinstance(q_learning_agent.strategy, tf.distribute.Strategy)

def test_sarsa_device_strategy(sarsa_agent):
    assert hasattr(sarsa_agent, 'strategy')
    assert isinstance(sarsa_agent.strategy, tf.distribute.Strategy)

def test_q_learning_epsilon_decay(q_learning_agent, state_dim):
    state = tf.random.normal((state_dim,))
    next_state = tf.random.normal((state_dim,))
    initial_epsilon = q_learning_agent.epsilon

    for _ in range(5):
        q_learning_agent.update(state, 0, 1.0, next_state, False)

    assert q_learning_agent.epsilon < initial_epsilon

def test_sarsa_epsilon_decay(sarsa_agent, state_dim):
    state = tf.random.normal((state_dim,))
    next_state = tf.random.normal((state_dim,))
    initial_epsilon = sarsa_agent.epsilon

    for _ in range(5):
        sarsa_agent.update(state, 0, 1.0, next_state, 1, False)

    assert sarsa_agent.epsilon < initial_epsilon
