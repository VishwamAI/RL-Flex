import tensorflow as tf
import numpy as np
import pytest
from RL_Developments.Tensorflow.hierarchical_rl import HierarchicalRL

@pytest.fixture
def state_dim():
    return 8

@pytest.fixture
def action_dim():
    return 4

@pytest.fixture
def num_options():
    return 3

@pytest.fixture
def hierarchical_agent(state_dim, action_dim, num_options):
    return HierarchicalRL(
        state_dim=state_dim,
        action_dim=action_dim,
        num_options=num_options,
        hidden_dims=[32, 32],
        high_level_learning_rate=1e-4,
        low_level_learning_rate=3e-4
    )

def test_initialization(hierarchical_agent, state_dim, action_dim, num_options):
    assert hierarchical_agent.state_dim == state_dim
    assert hierarchical_agent.action_dim == action_dim
    assert hierarchical_agent.num_options == num_options
    assert isinstance(hierarchical_agent.high_level_policy, tf.keras.Model)
    assert isinstance(hierarchical_agent.low_level_policies, list)
    assert len(hierarchical_agent.low_level_policies) == num_options

def test_get_option(hierarchical_agent, state_dim):
    state = tf.random.normal((state_dim,))
    option = hierarchical_agent.get_option(state)
    assert isinstance(option, int)
    assert 0 <= option < hierarchical_agent.num_options

def test_get_action(hierarchical_agent, state_dim):
    state = tf.random.normal((state_dim,))
    option = 0
    action = hierarchical_agent.get_action(state, option)
    assert isinstance(action, tf.Tensor)
    assert action.shape == (hierarchical_agent.action_dim,)

def test_update(hierarchical_agent, state_dim, action_dim):
    batch_size = 32
    states = tf.random.normal((batch_size, state_dim))
    actions = tf.random.uniform((batch_size, action_dim), -1, 1)
    options = tf.random.uniform((batch_size,), 0, hierarchical_agent.num_options, dtype=tf.int32)
    rewards = tf.random.normal((batch_size, 1))
    next_states = tf.random.normal((batch_size, state_dim))
    dones = tf.zeros((batch_size, 1))

    metrics = hierarchical_agent.update(states, actions, options, rewards, next_states, dones)
    assert isinstance(metrics, dict)
    assert 'high_level_loss' in metrics
    assert 'low_level_loss' in metrics
    assert 'option_termination_loss' in metrics

def test_device_strategy(hierarchical_agent):
    assert hasattr(hierarchical_agent, 'strategy')
    assert isinstance(hierarchical_agent.strategy, tf.distribute.Strategy)

def test_option_termination(hierarchical_agent, state_dim):
    state = tf.random.normal((state_dim,))
    option = 0
    terminate = hierarchical_agent.should_terminate_option(state, option)
    assert isinstance(terminate, bool)

def test_save_load_weights(hierarchical_agent, tmp_path):
    filepath = str(tmp_path / "test_model")

    # Save weights
    hierarchical_agent.save_weights(filepath)

    # Create new agent with same architecture
    new_agent = HierarchicalRL(
        state_dim=hierarchical_agent.state_dim,
        action_dim=hierarchical_agent.action_dim,
        num_options=hierarchical_agent.num_options,
        hidden_dims=[32, 32]
    )

    # Load weights
    new_agent.load_weights(filepath)

    # Compare weights
    for orig_var, new_var in zip(
        hierarchical_agent.high_level_policy.variables,
        new_agent.high_level_policy.variables
    ):
        assert tf.reduce_all(orig_var == new_var)

    for i in range(hierarchical_agent.num_options):
        for orig_var, new_var in zip(
            hierarchical_agent.low_level_policies[i].variables,
            new_agent.low_level_policies[i].variables
        ):
            assert tf.reduce_all(orig_var == new_var)

def test_option_switching(hierarchical_agent, state_dim):
    state = tf.random.normal((state_dim,))

    # Get initial option
    option = hierarchical_agent.get_option(state)

    # Run multiple steps and check if option switching occurs
    num_steps = 50
    option_changes = 0

    for _ in range(num_steps):
        new_state = tf.random.normal((state_dim,))
        if hierarchical_agent.should_terminate_option(new_state, option):
            new_option = hierarchical_agent.get_option(new_state)
            if new_option != option:
                option_changes += 1
            option = new_option


    # There should be some option switches but not too many
    assert 0 < option_changes < num_steps

def test_deterministic_vs_stochastic_actions(hierarchical_agent, state_dim):
    state = tf.random.normal((state_dim,))
    option = 0

    # Test deterministic actions
    det_action1 = hierarchical_agent.get_action(state, option, deterministic=True)
    det_action2 = hierarchical_agent.get_action(state, option, deterministic=True)
    assert tf.reduce_all(det_action1 == det_action2)

    # Test stochastic actions
    stoch_action1 = hierarchical_agent.get_action(state, option, deterministic=False)
    stoch_action2 = hierarchical_agent.get_action(state, option, deterministic=False)
    assert not tf.reduce_all(stoch_action1 == stoch_action2)
