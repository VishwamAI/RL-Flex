import tensorflow as tf
import numpy as np
from RL_Developments.Tensorflow.cognition import CognitionAgent

def test_cognition_initialization():
    """Test CognitionAgent initialization."""
    state_dim = 4
    action_dim = 2

    agent = CognitionAgent(state_dim, action_dim)

    # Verify initialization
    assert isinstance(agent.perception_net, tf.keras.Sequential)
    assert isinstance(agent.decision_net, tf.keras.Sequential)
    assert isinstance(agent.memory, tf.Variable)
    assert agent.memory.shape == (agent.memory_size, agent.perception_net.layers[-1].units)

def test_attention_mechanism():
    """Test attention mechanism."""
    state_dim = 4
    action_dim = 2
    batch_size = 32
    hidden_dim = 256

    agent = CognitionAgent(state_dim, action_dim)

    # Create dummy inputs
    query = tf.random.normal((batch_size, hidden_dim))
    key = tf.random.normal((agent.memory_size, hidden_dim))
    value = tf.random.normal((agent.memory_size, hidden_dim))

    # Test attention computation
    attention_output = agent.attention(
        query[:, None],
        key[None],
        value[None]
    )
    assert attention_output.shape == (batch_size, 1, hidden_dim)

def test_memory_update():
    """Test memory update mechanism."""
    state_dim = 4
    action_dim = 2

    agent = CognitionAgent(state_dim, action_dim)
    state_encoding = tf.random.normal((1, agent.perception_net.layers[-1].units))

    # Get initial memory state
    initial_memory = agent.memory.numpy().copy()
    initial_counter = agent.memory_counter.numpy()

    # Update memory
    agent.update_memory(state_encoding)

    # Verify memory was updated
    assert agent.memory_counter.numpy() == initial_counter + 1
    assert not np.array_equal(agent.memory.numpy(), initial_memory)
    assert np.array_equal(
        agent.memory[initial_counter % agent.memory_size].numpy(),
        state_encoding[0].numpy()
    )

def test_forward_pass():
    """Test forward pass through networks."""
    state_dim = 4
    action_dim = 2
    batch_size = 32

    agent = CognitionAgent(state_dim, action_dim)
    states = tf.random.normal((batch_size, state_dim))

    # Test forward pass
    actions, info = agent(states)
    assert actions.shape == (batch_size, action_dim)
    assert "state_encoding" in info
    assert "memory_output" in info
    assert info["state_encoding"].shape == (
        batch_size,
        agent.perception_net.layers[-1].units
    )
    assert info["memory_output"].shape == (
        batch_size,
        agent.perception_net.layers[-1].units
    )

def test_update():
    """Test network updates."""
    state_dim = 4
    action_dim = 2
    batch_size = 32

    agent = CognitionAgent(state_dim, action_dim)

    # Create dummy batch
    states = tf.random.normal((batch_size, state_dim))
    actions = tf.random.normal((batch_size, action_dim))
    rewards = tf.random.normal((batch_size,))
    next_states = tf.random.normal((batch_size, state_dim))
    dones = tf.zeros((batch_size,))

    # Test update
    metrics = agent.update(
        states,
        actions,
        rewards,
        next_states,
        dones
    )

    assert "action_loss" in metrics
    assert "memory_sparsity" in metrics
    assert "total_loss" in metrics

def test_save_load_weights(tmp_path):
    """Test weight saving and loading."""
    state_dim = 4
    action_dim = 2

    # Create two agents with different weights
    agent1 = CognitionAgent(state_dim, action_dim)
    agent2 = CognitionAgent(state_dim, action_dim)

    # Save weights from first agent
    save_path = str(tmp_path / "test_weights")
    agent1.save_weights(save_path)

    # Get predictions before loading
    states = tf.random.normal((1, state_dim))
    before_actions, _ = agent2(states)

    # Load weights into second agent
    agent2.load_weights(save_path)

    # Get predictions after loading
    after_actions, _ = agent2(states)

    # Verify predictions changed after loading weights
    assert not tf.reduce_all(tf.equal(before_actions, after_actions))
