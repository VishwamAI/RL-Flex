import tensorflow as tf
import numpy as np
import pytest
from RL_Developments.Tensorflow.offline_rl import OfflineRL

@pytest.fixture
def state_dim():
    return 4

@pytest.fixture
def action_dim():
    return 2

@pytest.fixture
def offline_rl_agent(state_dim, action_dim):
    return OfflineRL(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[32, 32],
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4
    )

def test_offline_rl_initialization(offline_rl_agent, state_dim, action_dim):
    assert offline_rl_agent.state_dim == state_dim
    assert offline_rl_agent.action_dim == action_dim
    assert isinstance(offline_rl_agent.actor, tf.keras.Model)
    assert isinstance(offline_rl_agent.critic, tf.keras.Model)
    assert isinstance(offline_rl_agent.target_critic, tf.keras.Model)

def test_get_action(offline_rl_agent, state_dim):
    state = tf.random.normal((state_dim,))
    action = offline_rl_agent.get_action(state)
    assert isinstance(action, tf.Tensor)
    assert action.shape == (offline_rl_agent.action_dim,)
    assert tf.reduce_all(tf.abs(action) <= 1.0)

def test_update(offline_rl_agent, state_dim, action_dim):
    batch_size = 32
    states = tf.random.normal((batch_size, state_dim))
    actions = tf.random.uniform((batch_size, action_dim), -1, 1)
    rewards = tf.random.normal((batch_size, 1))
    next_states = tf.random.normal((batch_size, state_dim))
    dones = tf.zeros((batch_size, 1))

    metrics = offline_rl_agent.update(states, actions, rewards, next_states, dones)
    assert isinstance(metrics, dict)
    assert 'critic_loss' in metrics
    assert 'cql_loss' in metrics
    assert 'actor_loss' in metrics

def test_device_strategy(offline_rl_agent):
    assert hasattr(offline_rl_agent, 'strategy')
    assert isinstance(offline_rl_agent.strategy, tf.distribute.Strategy)

def test_save_load_weights(offline_rl_agent, tmp_path):
    filepath = str(tmp_path / "test_model")

    # Save weights
    offline_rl_agent.save_weights(filepath)

    # Create new agent with same architecture
    new_agent = OfflineRL(
        state_dim=offline_rl_agent.state_dim,
        action_dim=offline_rl_agent.action_dim,
        hidden_dims=[32, 32]
    )

    # Load weights
    new_agent.load_weights(filepath)

    # Compare weights
    for orig_var, new_var in zip(
        offline_rl_agent.actor.variables,
        new_agent.actor.variables
    ):
        assert tf.reduce_all(orig_var == new_var)

    for orig_var, new_var in zip(
        offline_rl_agent.critic.variables,
        new_agent.critic.variables
    ):
        assert tf.reduce_all(orig_var == new_var)

def test_deterministic_vs_stochastic_actions(offline_rl_agent, state_dim):
    state = tf.random.normal((state_dim,))

    # Test deterministic actions
    det_action1 = offline_rl_agent.get_action(state, deterministic=True)
    det_action2 = offline_rl_agent.get_action(state, deterministic=True)
    assert tf.reduce_all(det_action1 == det_action2)

    # Test stochastic actions
    stoch_action1 = offline_rl_agent.get_action(state, deterministic=False)
    stoch_action2 = offline_rl_agent.get_action(state, deterministic=False)
    assert not tf.reduce_all(stoch_action1 == stoch_action2)

def test_cql_regularization(offline_rl_agent, state_dim, action_dim):
    batch_size = 32
    states = tf.random.normal((batch_size, state_dim))
    actions = tf.random.uniform((batch_size, action_dim), -1, 1)
    rewards = tf.random.normal((batch_size, 1))
    next_states = tf.random.normal((batch_size, state_dim))
    dones = tf.zeros((batch_size, 1))

    # Test with different CQL alpha values
    offline_rl_agent.cql_alpha = 0.0
    metrics_no_cql = offline_rl_agent.update(states, actions, rewards, next_states, dones)

    offline_rl_agent.cql_alpha = 1.0
    metrics_with_cql = offline_rl_agent.update(states, actions, rewards, next_states, dones)

    assert metrics_with_cql['cql_loss'] > 0
    assert metrics_with_cql['critic_loss'] != metrics_no_cql['critic_loss']
