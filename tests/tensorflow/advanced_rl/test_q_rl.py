import tensorflow as tf
import numpy as np
from RL_Developments.Tensorflow.q_rl import (
    QRLAgent,
    QuantumLayer,
    QuantumPolicyNetwork,
    QuantumValueNetwork
)

def test_quantum_layer_initialization():
    """Test QuantumLayer initialization."""
    units = 64
    layer = QuantumLayer(units)

    # Verify initialization
    assert layer.units == units
    assert isinstance(layer.activation_fn, tf.keras.activations.Activation)

def test_quantum_policy_network_initialization():
    """Test QuantumPolicyNetwork initialization."""
    state_dim = 4
    action_dim = 2

    network = QuantumPolicyNetwork(state_dim, action_dim)

    # Verify initialization
    assert len(network.quantum_layers) > 0
    assert isinstance(network.output_layer, tf.keras.layers.Dense)

def test_quantum_value_network_initialization():
    """Test QuantumValueNetwork initialization."""
    state_dim = 4

    network = QuantumValueNetwork(state_dim)

    # Verify initialization
    assert len(network.quantum_layers) > 0
    assert isinstance(network.output_layer, tf.keras.layers.Dense)

def test_qrl_agent_initialization():
    """Test QRLAgent initialization."""
    state_dim = 4
    action_dim = 2

    agent = QRLAgent(state_dim, action_dim)

    # Verify initialization
    assert isinstance(agent.policy, QuantumPolicyNetwork)
    assert isinstance(agent.value, QuantumValueNetwork)

def test_quantum_layer_forward_pass():
    """Test QuantumLayer forward pass."""
    batch_size = 32
    input_dim = 4
    units = 64

    layer = QuantumLayer(units)
    inputs = tf.random.normal((batch_size, input_dim))

    # Test forward pass
    outputs = layer(inputs)
    assert outputs.shape == (batch_size, units)

def test_quantum_policy_network_forward_pass():
    """Test QuantumPolicyNetwork forward pass."""
    state_dim = 4
    action_dim = 2
    batch_size = 32

    network = QuantumPolicyNetwork(state_dim, action_dim)
    states = tf.random.normal((batch_size, state_dim))

    # Test forward pass
    outputs = network(states)
    assert outputs.shape == (batch_size, action_dim * 2)

def test_quantum_value_network_forward_pass():
    """Test QuantumValueNetwork forward pass."""
    state_dim = 4
    batch_size = 32

    network = QuantumValueNetwork(state_dim)
    states = tf.random.normal((batch_size, state_dim))

    # Test forward pass
    values = network(states)
    assert values.shape == (batch_size, 1)

def test_action_generation():
    """Test action generation."""
    state_dim = 4
    action_dim = 2
    batch_size = 32

    agent = QRLAgent(state_dim, action_dim)
    states = tf.random.normal((batch_size, state_dim))

    # Test action generation
    actions, info = agent.get_action(states)
    assert actions.shape == (batch_size, action_dim)
    assert tf.reduce_all(actions >= -1)
    assert tf.reduce_all(actions <= 1)
    assert all(key in info for key in ["means", "log_stds", "stds"])

def test_update():
    """Test network updates."""
    state_dim = 4
    action_dim = 2
    batch_size = 32

    agent = QRLAgent(state_dim, action_dim)

    # Create dummy batch
    states = tf.random.normal((batch_size, state_dim))
    actions = tf.random.normal((batch_size, action_dim))
    rewards = tf.random.normal((batch_size,))
    next_states = tf.random.normal((batch_size, state_dim))
    dones = tf.zeros((batch_size,))

    # Test update
    metrics = agent.update(states, actions, rewards, next_states, dones)

    assert "policy_loss" in metrics
    assert "value_loss" in metrics

def test_save_load_weights(tmp_path):
    """Test weight saving and loading."""
    state_dim = 4
    action_dim = 2

    # Create two agents with different weights
    agent1 = QRLAgent(state_dim, action_dim)
    agent2 = QRLAgent(state_dim, action_dim)

    # Save weights from first agent
    save_path = str(tmp_path / "test_weights")
    agent1.save_weights(save_path)

    # Get predictions before loading
    states = tf.random.normal((1, state_dim))
    before_actions, _ = agent2.get_action(states)

    # Load weights into second agent
    agent2.load_weights(save_path)

    # Get predictions after loading
    after_actions, _ = agent2.get_action(states)

    # Verify predictions changed after loading weights
    assert not tf.reduce_all(tf.equal(before_actions, after_actions))
