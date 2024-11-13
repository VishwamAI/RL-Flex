import tensorflow as tf
import numpy as np
from RL_Developments.Tensorflow.agentic_behavior import AgenticBehavior

def test_agentic_behavior_initialization():
    """Test AgenticBehavior initialization."""
    state_dim = 4
    action_dim = 2

    agent = AgenticBehavior(state_dim, action_dim)

    # Verify initialization
    assert isinstance(agent.goal_network, tf.keras.Sequential)
    assert isinstance(agent.policy_network, tf.keras.Sequential)
    assert isinstance(agent.value_network, tf.keras.Sequential)

def test_goal_prediction():
    """Test goal prediction."""
    state_dim = 4
    action_dim = 2
    batch_size = 32

    agent = AgenticBehavior(state_dim, action_dim)
    states = tf.random.normal((batch_size, state_dim))

    # Test goal prediction
    goals = agent.predict_goal(states)
    assert goals.shape == (batch_size, state_dim)

def test_action_generation():
    """Test action generation."""
    state_dim = 4
    action_dim = 2
    batch_size = 32

    agent = AgenticBehavior(state_dim, action_dim)
    states = tf.random.normal((batch_size, state_dim))

    # Test action generation without goals
    actions, info = agent.get_action(states)
    assert actions.shape == (batch_size, action_dim)
    assert tf.reduce_all(actions >= -1)
    assert tf.reduce_all(actions <= 1)
    assert all(key in info for key in ["means", "log_stds", "stds", "goals"])

    # Test action generation with goals
    goals = tf.random.normal((batch_size, state_dim))
    actions, info = agent.get_action(states, goals)
    assert actions.shape == (batch_size, action_dim)
    assert tf.reduce_all(actions >= -1)
    assert tf.reduce_all(actions <= 1)
    assert all(key in info for key in ["means", "log_stds", "stds", "goals"])

def test_update():
    """Test network updates."""
    state_dim = 4
    action_dim = 2
    batch_size = 32

    agent = AgenticBehavior(state_dim, action_dim)

    # Create dummy batch
    states = tf.random.normal((batch_size, state_dim))
    actions = tf.random.normal((batch_size, action_dim))
    rewards = tf.random.normal((batch_size,))
    next_states = tf.random.normal((batch_size, state_dim))
    dones = tf.zeros((batch_size,))

    # Test update without target goals
    metrics = agent.update(states, actions, rewards, next_states, dones)
    assert "goal_loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics

    # Test update with target goals
    target_goals = tf.random.normal((batch_size, state_dim))
    metrics = agent.update(
        states,
        actions,
        rewards,
        next_states,
        dones,
        target_goals
    )
    assert "goal_loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics

def test_save_load_weights(tmp_path):
    """Test weight saving and loading."""
    state_dim = 4
    action_dim = 2

    # Create two agents with different weights
    agent1 = AgenticBehavior(state_dim, action_dim)
    agent2 = AgenticBehavior(state_dim, action_dim)

    # Save weights from first agent
    save_path = str(tmp_path / "test_weights")
    agent1.save_weights(save_path)

    # Get predictions before loading
    states = tf.random.normal((1, state_dim))
    before_goals = agent2.predict_goal(states)
    before_actions, _ = agent2.get_action(states)

    # Load weights into second agent
    agent2.load_weights(save_path)

    # Get predictions after loading
    after_goals = agent2.predict_goal(states)
    after_actions, _ = agent2.get_action(states)

    # Verify predictions changed after loading weights
    assert not tf.reduce_all(tf.equal(before_goals, after_goals))
    assert not tf.reduce_all(tf.equal(before_actions, after_actions))
