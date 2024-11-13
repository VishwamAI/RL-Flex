import tensorflow as tf
import numpy as np
from RL_Developments.Tensorflow.mcts import MCTSAgent, MCTSNode

class DummyEnv:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def step(self, action):
        next_state = tf.random.normal((self.state_dim,))
        reward = float(tf.random.normal(()))
        done = False
        info = {}
        return next_state, reward, done, info

def test_mcts_node():
    """Test MCTS Node functionality."""
    state = tf.zeros(4)
    node = MCTSNode(state)

    # Test child creation
    child_state = tf.ones(4)
    child = node.add_child(0, child_state)

    assert 0 in node.children
    assert node.children[0] == child
    assert child.parent == node

    # Test update
    node.update(1.0)
    assert node.visits == 1
    assert node.value == 1.0

    node.update(0.0)
    assert node.visits == 2
    assert node.value == 0.5

def test_mcts_agent_initialization():
    """Test MCTS Agent initialization."""
    state_dim = 4
    action_dim = 2

    agent = MCTSAgent(state_dim, action_dim)

    # Verify network initialization
    assert isinstance(agent.value_network, tf.keras.Sequential)

def test_mcts_agent_action_selection():
    """Test MCTS Agent action selection."""
    state_dim = 4
    action_dim = 2

    agent = MCTSAgent(state_dim, action_dim)
    env = DummyEnv(state_dim, action_dim)

    # Test action selection
    state = tf.zeros(state_dim)
    action = agent.get_action(state, env)

    assert isinstance(action, int)
    assert 0 <= action < action_dim
