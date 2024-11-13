import tensorflow as tf
import numpy as np
from RL_Developments.Tensorflow.utils.rl_utils import (
    discount_cumsum,
    get_minibatches,
    PPOBuffer
)

def test_discount_cumsum():
    """Test discounted cumulative sum calculation."""
    x = tf.constant([1.0, 2.0, 3.0])
    gamma = 0.99

    result = discount_cumsum(x, gamma)

    # Manual calculation
    expected = tf.constant([
        1.0 + 0.99 * (2.0 + 0.99 * 3.0),
        2.0 + 0.99 * 3.0,
        3.0
    ])

    assert tf.reduce_all(tf.abs(result - expected) < 1e-6)

def test_get_minibatches():
    """Test minibatch creation."""
    data = {
        'x': tf.random.normal((100, 4)),
        'y': tf.random.normal((100, 2))
    }
    batch_size = 32

    batches = get_minibatches(data, batch_size)

    # Check number of batches
    assert len(batches) == (100 + batch_size - 1) // batch_size

    # Check batch sizes
    for i, batch in enumerate(batches):
        expected_size = min(batch_size, 100 - i * batch_size)
        assert batch['x'].shape[0] == expected_size
        assert batch['y'].shape[0] == expected_size

def test_ppo_buffer():
    """Test PPO buffer functionality."""
    state_dim = 4
    action_dim = 2
    buffer_size = 1000

    buffer = PPOBuffer(state_dim, action_dim, buffer_size)

    # Test storing
    for _ in range(100):
        state = tf.random.normal((state_dim,))
        action = tf.random.normal((action_dim,))
        reward = float(tf.random.normal(()))
        value = float(tf.random.normal(()))
        logp = float(tf.random.normal(()))

        buffer.store(state, action, reward, value, logp)

    # Test path finishing
    last_value = 0.0
    buffer.finish_path(last_value)

    # Test getting data
    buffer.ptr = buffer.max_size  # Simulate full buffer
    data = buffer.get()

    assert 'states' in data
    assert 'actions' in data
    assert 'advantages' in data
    assert 'returns' in data
    assert 'logps' in data

    assert data['states'].shape == (buffer_size, state_dim)
    assert data['actions'].shape == (buffer_size, action_dim)
    assert data['advantages'].shape == (buffer_size,)
    assert data['returns'].shape == (buffer_size,)
    assert data['logps'].shape == (buffer_size,)
