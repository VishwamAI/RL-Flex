import pytest
import tensorflow as tf
import numpy as np
from RL_Developments.Tensorflow.deep_rl_algorithms import DQNAgent, PPOAgent

def test_tensorflow_setup():
    """Test TensorFlow setup and device configuration."""
    # Check TensorFlow version
    assert tf.__version__ >= "2.0.0"

    # Check device availability
    devices = tf.config.list_physical_devices()
    assert len(devices) > 0

def test_agent_training():
    """Test basic training loop for both DQN and PPO agents."""
    state_dim = 4
    action_dim = 2
    batch_size = 32

    # Test DQN training
    dqn_agent = DQNAgent(state_dim, action_dim)
    states = tf.random.normal((batch_size, state_dim))
    next_states = tf.random.normal((batch_size, state_dim))
    actions = tf.random.uniform((batch_size,), 0, action_dim, dtype=tf.int32)
    rewards = tf.random.normal((batch_size,))
    dones = tf.zeros((batch_size,))

    loss = dqn_agent.update(states[0], actions[0], rewards[0], next_states[0], dones[0])
    assert isinstance(loss, float)

    # Test PPO training
    ppo_agent = PPOAgent(state_dim, action_dim)
    old_log_probs = tf.random.normal((batch_size,))

    metrics = ppo_agent.update(states, actions, rewards, dones, old_log_probs)
    assert isinstance(metrics, dict)
    assert 'total_loss' in metrics
    assert 'policy_loss' in metrics
    assert 'value_loss' in metrics
    assert 'entropy' in metrics
