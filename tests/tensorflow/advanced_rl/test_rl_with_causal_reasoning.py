import tensorflow as tf
import numpy as np
from RL_Developments.Tensorflow.rl_with_causal_reasoning import (
    RLWithCausalReasoning,
    CausalGraph,
    InterventionModel,
    CounterfactualModel
)

def test_causal_graph_initialization():
    """Test CausalGraph initialization."""
    state_dim = 4

    graph = CausalGraph(state_dim)

    # Verify initialization
    assert isinstance(graph.structural_model, tf.keras.Sequential)
    assert isinstance(graph.adjacency, tf.Variable)
    assert graph.adjacency.shape == (state_dim, state_dim)

def test_intervention_model_initialization():
    """Test InterventionModel initialization."""
    state_dim = 4
    action_dim = 2

    model = InterventionModel(state_dim, action_dim)

    # Verify initialization
    assert isinstance(model.intervention_network, tf.keras.Sequential)

def test_counterfactual_model_initialization():
    """Test CounterfactualModel initialization."""
    state_dim = 4
    action_dim = 2

    model = CounterfactualModel(state_dim, action_dim)

    # Verify initialization
    assert isinstance(model.counterfactual_network, tf.keras.Sequential)

def test_rl_with_causal_reasoning_initialization():
    """Test RLWithCausalReasoning initialization."""
    state_dim = 4
    action_dim = 2

    agent = RLWithCausalReasoning(state_dim, action_dim)

    # Verify initialization
    assert isinstance(agent.causal_graph, CausalGraph)
    assert isinstance(agent.intervention_model, InterventionModel)
    assert isinstance(agent.counterfactual_model, CounterfactualModel)
    assert isinstance(agent.policy, tf.keras.Sequential)
    assert isinstance(agent.value, tf.keras.Sequential)

def test_causal_effects():
    """Test causal effects computation."""
    state_dim = 4
    batch_size = 32

    graph = CausalGraph(state_dim)
    states = tf.random.normal((batch_size, state_dim))

    # Test causal effects computation
    effects, adjacency = graph.get_causal_effects(states)
    assert effects.shape == (batch_size, state_dim)
    assert adjacency.shape == (state_dim, state_dim)
    assert tf.reduce_all(adjacency >= 0)
    assert tf.reduce_all(adjacency <= 1)

def test_intervention():
    """Test intervention computation."""
    state_dim = 4
    action_dim = 2
    batch_size = 32

    model = InterventionModel(state_dim, action_dim)
    states = tf.random.normal((batch_size, state_dim))
    actions = tf.random.normal((batch_size, action_dim))

    # Test intervention
    intervened_states = model.intervene(states, actions)
    assert intervened_states.shape == (batch_size, state_dim)

def test_counterfactual_reasoning():
    """Test counterfactual reasoning."""
    state_dim = 4
    action_dim = 2
    batch_size = 32

    model = CounterfactualModel(state_dim, action_dim)
    states = tf.random.normal((batch_size, state_dim))
    actions = tf.random.normal((batch_size, action_dim))
    next_states = tf.random.normal((batch_size, state_dim))

    # Test counterfactual reasoning
    counterfactual_states = model.reason(states, actions, next_states)
    assert counterfactual_states.shape == (batch_size, state_dim)

def test_action_generation():
    """Test action generation."""
    state_dim = 4
    action_dim = 2
    batch_size = 32

    agent = RLWithCausalReasoning(state_dim, action_dim)
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

    agent = RLWithCausalReasoning(state_dim, action_dim)

    # Create dummy batch
    states = tf.random.normal((batch_size, state_dim))
    actions = tf.random.normal((batch_size, action_dim))
    rewards = tf.random.normal((batch_size,))
    next_states = tf.random.normal((batch_size, state_dim))
    dones = tf.zeros((batch_size,))

    # Test update
    metrics = agent.update(states, actions, rewards, next_states, dones)

    assert "causal_loss" in metrics
    assert "intervention_loss" in metrics
    assert "counterfactual_loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics

def test_save_load_weights(tmp_path):
    """Test weight saving and loading."""
    state_dim = 4
    action_dim = 2

    # Create two agents with different weights
    agent1 = RLWithCausalReasoning(state_dim, action_dim)
    agent2 = RLWithCausalReasoning(state_dim, action_dim)

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
