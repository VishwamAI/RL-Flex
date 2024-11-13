import tensorflow as tf
import numpy as np
import pytest
from RL_Developments.Tensorflow.rl_algorithms import QLearning, ActorCritic, PolicyGradient

@pytest.fixture
def state_dim():
    return 4

@pytest.fixture
def action_dim():
    return 2

@pytest.fixture
def q_learning_agent(state_dim, action_dim):
    return QLearning(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[32, 32],
        learning_rate=3e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )

@pytest.fixture
def actor_critic_agent(state_dim, action_dim):
    return ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[32, 32],
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        gamma=0.99
    )

@pytest.fixture
def policy_gradient_agent(state_dim, action_dim):
    return PolicyGradient(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[32, 32],
        learning_rate=3e-4,
        gamma=0.99
    )

# Q-Learning Tests
def test_q_learning_initialization(q_learning_agent, state_dim, action_dim):
    assert q_learning_agent.state_dim == state_dim
    assert q_learning_agent.action_dim == action_dim
    assert isinstance(q_learning_agent.q_network, tf.keras.Model)

def test_q_learning_get_action(q_learning_agent, state_dim):
    state = tf.random.normal((state_dim,))
    action = q_learning_agent.get_action(state)
    assert isinstance(action, int)
    assert 0 <= action < q_learning_agent.action_dim

def test_q_learning_update(q_learning_agent, state_dim, action_dim):
    batch_size = 32
    states = tf.random.normal((batch_size, state_dim))
    actions = tf.random.uniform((batch_size,), 0, action_dim, dtype=tf.int32)
    rewards = tf.random.normal((batch_size, 1))
    next_states = tf.random.normal((batch_size, state_dim))
    dones = tf.zeros((batch_size, 1))

    metrics = q_learning_agent.update(states, actions, rewards, next_states, dones)
    assert isinstance(metrics, dict)
    assert 'q_loss' in metrics
    assert 'mean_q_value' in metrics

def test_q_learning_epsilon_decay(q_learning_agent):
    initial_epsilon = q_learning_agent.epsilon
    q_learning_agent.decay_epsilon()
    assert q_learning_agent.epsilon == max(
        q_learning_agent.epsilon_end,
        initial_epsilon * q_learning_agent.epsilon_decay
    )

# Actor-Critic Tests
def test_actor_critic_initialization(actor_critic_agent, state_dim, action_dim):
    assert actor_critic_agent.state_dim == state_dim
    assert actor_critic_agent.action_dim == action_dim
    assert isinstance(actor_critic_agent.actor, tf.keras.Model)
    assert isinstance(actor_critic_agent.critic, tf.keras.Model)

def test_actor_critic_get_action(actor_critic_agent, state_dim):
    state = tf.random.normal((state_dim,))
    action = actor_critic_agent.get_action(state)
    assert isinstance(action, tf.Tensor)
    assert action.shape == (actor_critic_agent.action_dim,)
    assert tf.reduce_all(tf.abs(action) <= 1.0)

def test_actor_critic_update(actor_critic_agent, state_dim, action_dim):
    batch_size = 32
    states = tf.random.normal((batch_size, state_dim))
    actions = tf.random.uniform((batch_size, action_dim), -1, 1)
    rewards = tf.random.normal((batch_size, 1))
    next_states = tf.random.normal((batch_size, state_dim))
    dones = tf.zeros((batch_size, 1))

    metrics = actor_critic_agent.update(states, actions, rewards, next_states, dones)
    assert isinstance(metrics, dict)
    assert 'actor_loss' in metrics
    assert 'critic_loss' in metrics
    assert 'value_estimate' in metrics

# Policy Gradient Tests
def test_policy_gradient_initialization(policy_gradient_agent, state_dim, action_dim):
    assert policy_gradient_agent.state_dim == state_dim
    assert policy_gradient_agent.action_dim == action_dim
    assert isinstance(policy_gradient_agent.policy_network, tf.keras.Model)

def test_policy_gradient_get_action(policy_gradient_agent, state_dim):
    state = tf.random.normal((state_dim,))
    action = policy_gradient_agent.get_action(state)
    assert isinstance(action, tf.Tensor)
    assert action.shape == (policy_gradient_agent.action_dim,)
    assert tf.reduce_all(tf.abs(action) <= 1.0)

def test_policy_gradient_update(policy_gradient_agent, state_dim, action_dim):
    batch_size = 32
    states = tf.random.normal((batch_size, state_dim))
    actions = tf.random.uniform((batch_size, action_dim), -1, 1)
    rewards = tf.random.normal((batch_size, 1))

    metrics = policy_gradient_agent.update(states, actions, rewards)
    assert isinstance(metrics, dict)
    assert 'policy_loss' in metrics
    assert 'entropy' in metrics

def test_device_strategy(q_learning_agent, actor_critic_agent, policy_gradient_agent):
    assert hasattr(q_learning_agent, 'strategy')
    assert isinstance(q_learning_agent.strategy, tf.distribute.Strategy)
    assert hasattr(actor_critic_agent, 'strategy')
    assert isinstance(actor_critic_agent.strategy, tf.distribute.Strategy)
    assert hasattr(policy_gradient_agent, 'strategy')
    assert isinstance(policy_gradient_agent.strategy, tf.distribute.Strategy)

def test_deterministic_vs_stochastic_actions(actor_critic_agent, state_dim):
    state = tf.random.normal((state_dim,))

    # Test deterministic actions
    det_action1 = actor_critic_agent.get_action(state, deterministic=True)
    det_action2 = actor_critic_agent.get_action(state, deterministic=True)
    assert tf.reduce_all(det_action1 == det_action2)

    # Test stochastic actions
    stoch_action1 = actor_critic_agent.get_action(state, deterministic=False)
    stoch_action2 = actor_critic_agent.get_action(state, deterministic=False)
    assert not tf.reduce_all(stoch_action1 == stoch_action2)

def test_save_load_weights(actor_critic_agent, tmp_path):
    filepath = str(tmp_path / "test_model")

    # Save weights
    actor_critic_agent.save_weights(filepath)

    # Create new agent with same architecture
    new_agent = ActorCritic(
        state_dim=actor_critic_agent.state_dim,
        action_dim=actor_critic_agent.action_dim,
        hidden_dims=[32, 32]
    )

    # Load weights
    new_agent.load_weights(filepath)

    # Compare weights
    for orig_var, new_var in zip(
        actor_critic_agent.actor.variables,
        new_agent.actor.variables
    ):
        assert tf.reduce_all(orig_var == new_var)

    for orig_var, new_var in zip(
        actor_critic_agent.critic.variables,
        new_agent.critic.variables
    ):
        assert tf.reduce_all(orig_var == new_var)
