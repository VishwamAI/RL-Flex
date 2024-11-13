import tensorflow as tf
import numpy as np
from RL_Developments.Tensorflow.advanced_rl_agent import AdvancedRLAgent

def test_advanced_rl_agent_initialization():
    """Test AdvancedRLAgent initialization."""
    state_dim = 4
    action_dim = 2

    # Test with all features enabled
    agent = AdvancedRLAgent(
        state_dim,
        action_dim,
        use_double_q=True,
        use_dueling=True,
        use_noisy=True,
        use_per=True,
        use_n_step=True,
        use_distributional=True
    )

    assert isinstance(agent.online_q, tf.keras.Model)
    assert isinstance(agent.target_q, tf.keras.Model)

    # Test with all features disabled
    agent = AdvancedRLAgent(
        state_dim,
        action_dim,
        use_double_q=False,
        use_dueling=False,
        use_noisy=False,
        use_per=False,
        use_n_step=False,
        use_distributional=False
    )

    assert isinstance(agent.online_q, tf.keras.Model)
    assert isinstance(agent.target_q, tf.keras.Model)

def test_action_selection():
    """Test action selection."""
    state_dim = 4
    action_dim = 2

    # Test with distributional RL
    agent = AdvancedRLAgent(
        state_dim,
        action_dim,
        use_distributional=True
    )
    state = tf.random.normal((state_dim,))

    # Test deterministic selection
    action = agent.get_action(state, deterministic=True)
    assert 0 <= action < action_dim

    # Test stochastic selection
    action = agent.get_action(state, deterministic=False)
    assert 0 <= action < action_dim

    # Test with standard Q-learning
    agent = AdvancedRLAgent(
        state_dim,
        action_dim,
        use_distributional=False
    )

    # Test deterministic selection
    action = agent.get_action(state, deterministic=True)
    assert 0 <= action < action_dim

    # Test stochastic selection
    action = agent.get_action(state, deterministic=False)
    assert 0 <= action < action_dim

def test_update():
    """Test network updates."""
    state_dim = 4
    action_dim = 2
    batch_size = 32

    # Test with distributional RL
    agent = AdvancedRLAgent(
        state_dim,
        action_dim,
        use_distributional=True
    )

    # Create dummy batch
    states = tf.random.normal((batch_size, state_dim))
    actions = tf.random.uniform(
        (batch_size,), 0, action_dim, dtype=tf.int32
    )
    rewards = tf.random.normal((batch_size,))
    next_states = tf.random.normal((batch_size, state_dim))
    dones = tf.zeros((batch_size,))

    # Test update without importance weights
    metrics = agent.update(
        states, actions, rewards, next_states, dones
    )
    assert "loss" in metrics
    assert "q_mean" in metrics

    # Test update with importance weights
    weights = tf.ones((batch_size,))
    metrics = agent.update(
        states, actions, rewards, next_states, dones, weights
    )
    assert "loss" in metrics
    assert "q_mean" in metrics

    # Test with standard Q-learning
    agent = AdvancedRLAgent(
        state_dim,
        action_dim,
        use_distributional=False
    )

    # Test update
    metrics = agent.update(
        states, actions, rewards, next_states, dones
    )
    assert "loss" in metrics
    assert "q_mean" in metrics

def test_target_update():
    """Test target network update."""
    state_dim = 4
    action_dim = 2

    agent = AdvancedRLAgent(state_dim, action_dim)

    # Get initial parameters
    initial_params = [
        tf.identity(param)
        for param in agent.target_q.trainable_variables
    ]

    # Update target network
    agent.update_target(tau=0.5)

    # Verify parameters changed
    for initial, current in zip(
        initial_params,
        agent.target_q.trainable_variables
    ):
        assert not tf.reduce_all(initial == current)
