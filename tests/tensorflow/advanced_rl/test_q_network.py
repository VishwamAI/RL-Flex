import tensorflow as tf
import numpy as np
from RL_Developments.Tensorflow.q_network import QNetwork, NoisyDense

def test_q_network_initialization():
    """Test QNetwork initialization."""
    state_dim = 4
    action_dim = 2

    # Test standard Q-network
    qnet = QNetwork(
        state_dim,
        action_dim,
        dueling=False,
        noisy=False,
        distributional=False
    )
    assert isinstance(qnet, tf.keras.Model)

    # Test dueling Q-network
    dueling_qnet = QNetwork(
        state_dim,
        action_dim,
        dueling=True,
        noisy=False,
        distributional=False
    )
    assert isinstance(dueling_qnet, tf.keras.Model)

    # Test noisy Q-network
    noisy_qnet = QNetwork(
        state_dim,
        action_dim,
        dueling=False,
        noisy=True,
        distributional=False
    )
    assert isinstance(noisy_qnet, tf.keras.Model)

    # Test distributional Q-network
    dist_qnet = QNetwork(
        state_dim,
        action_dim,
        dueling=False,
        noisy=False,
        distributional=True
    )
    assert isinstance(dist_qnet, tf.keras.Model)

def test_q_network_forward_pass():
    """Test Q-network forward pass."""
    state_dim = 4
    action_dim = 2
    batch_size = 32

    # Test standard Q-network
    qnet = QNetwork(
        state_dim,
        action_dim,
        dueling=False,
        noisy=False,
        distributional=False
    )
    states = tf.random.normal((batch_size, state_dim))
    q_values = qnet(states)
    assert q_values.shape == (batch_size, action_dim)

    # Test dueling Q-network
    dueling_qnet = QNetwork(
        state_dim,
        action_dim,
        dueling=True,
        noisy=False,
        distributional=False
    )
    q_values = dueling_qnet(states)
    assert q_values.shape == (batch_size, action_dim)

    # Test distributional Q-network
    dist_qnet = QNetwork(
        state_dim,
        action_dim,
        dueling=False,
        noisy=False,
        distributional=True,
        num_atoms=51
    )
    q_dist = dist_qnet(states)
    assert q_dist.shape == (batch_size, action_dim, 51)
    assert tf.reduce_all(q_dist >= 0.0)
    assert tf.reduce_all(q_dist <= 1.0)
    assert tf.reduce_all(
        tf.abs(tf.reduce_sum(q_dist, axis=-1) - 1.0) < 1e-5
    )

def test_noisy_dense():
    """Test NoisyDense layer."""
    input_dim = 4
    output_dim = 2
    batch_size = 32

    layer = NoisyDense(output_dim)
    inputs = tf.random.normal((batch_size, input_dim))

    # Test forward pass
    outputs = layer(inputs)
    assert outputs.shape == (batch_size, output_dim)

    # Test noise reset
    old_weight_epsilon = layer.weight_epsilon
    old_bias_epsilon = layer.bias_epsilon
    layer.reset_noise()
    assert not tf.reduce_all(old_weight_epsilon == layer.weight_epsilon)
    assert not tf.reduce_all(old_bias_epsilon == layer.bias_epsilon)

def test_q_network_noise_reset():
    """Test Q-network noise reset."""
    state_dim = 4
    action_dim = 2

    qnet = QNetwork(
        state_dim,
        action_dim,
        dueling=True,
        noisy=True,
        distributional=True
    )

    # Verify noise reset works
    qnet.reset_noise()  # Should not raise any errors
