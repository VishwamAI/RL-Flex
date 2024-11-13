import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Any
from .utils import get_device_strategy

class MCTSNode:
    """Monte Carlo Tree Search Node."""

    def __init__(self, state, action=None, parent=None):
        """Initialize MCTS node.

        Args:
            state: Current state
            action: Action that led to this state
            parent: Parent node
        """
        self.state = state
        self.action = action
        self.parent = parent
        self.children: Dict[int, MCTSNode] = {}
        self.visits = 0
        self.value = 0.0

    def add_child(self, action: int, state: tf.Tensor) -> 'MCTSNode':
        """Add child node."""
        child = MCTSNode(state, action, self)
        self.children[action] = child
        return child

    def update(self, value: float):
        """Update node statistics."""
        self.visits += 1
        self.value += (value - self.value) / self.visits

class MCTSAgent:
    """Monte Carlo Tree Search Agent implementation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_simulations: int = 50,
        exploration_constant: float = 1.414,
        discount_factor: float = 0.99
    ):
        """Initialize MCTS Agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            num_simulations: Number of MCTS simulations
            exploration_constant: UCT exploration constant
            discount_factor: Reward discount factor
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.discount_factor = discount_factor

        # Get device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create value network for state evaluation
            self.value_network = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)
            ])

            # Build network
            dummy_state = tf.zeros((1, state_dim))
            self.value_network(dummy_state)

    def select_action(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """Select action using UCT formula."""
        best_score = float('-inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            # UCT formula
            exploitation = child.value
            exploration = self.exploration_constant * np.sqrt(
                np.log(node.visits) / (child.visits + 1e-8)
            )
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, node: MCTSNode, env) -> MCTSNode:
        """Expand node with new child."""
        # Get available actions
        available_actions = list(range(self.action_dim))
        taken_actions = list(node.children.keys())

        # Remove already taken actions
        for action in taken_actions:
            available_actions.remove(action)

        if not available_actions:
            return node

        # Select random action
        action = np.random.choice(available_actions)

        # Get next state from environment
        next_state, _, _, _ = env.step(action)

        # Create child node
        return node.add_child(action, next_state)

    def simulate(self, state: tf.Tensor, env, depth: int = 5) -> float:
        """Simulate episode using value network.

        Args:
            state: Current state
            env: Environment instance
            depth: Simulation depth

        Returns:
            Cumulative discounted reward
        """
        if depth == 0:
            return float(self.value_network(tf.expand_dims(state, 0))[0])

        # Random simulation
        value = 0.0
        discount = 1.0
        current_state = state

        for _ in range(depth):
            # Random action
            action = np.random.randint(self.action_dim)

            # Environment step
            next_state, reward, done, _ = env.step(action)

            value += discount * reward
            discount *= self.discount_factor

            if done:
                break

            current_state = next_state

        # Add bootstrap value
        if not done:
            bootstrap_value = float(
                self.value_network(tf.expand_dims(current_state, 0))[0]
            )
            value += discount * bootstrap_value

        return value

    def backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value through tree."""
        while node is not None:
            node.update(value)
            node = node.parent

    def get_action(self, state: tf.Tensor, env) -> int:
        """Get best action using MCTS.

        Args:
            state: Current state
            env: Environment instance

        Returns:
            Selected action
        """
        # Create root node
        root = MCTSNode(state)

        # Run MCTS simulations
        for _ in range(self.num_simulations):
            node = root

            # Selection
            while node.children and len(node.children) == self.action_dim:
                _, node = self.select_action(node)

            # Expansion
            if node.visits > 0:
                node = self.expand(node, env)

            # Simulation
            value = self.simulate(node.state)

            # Backpropagation
            self.backpropagate(node, value)

        # Select best action
        visits = np.zeros(self.action_dim)
        for action, child in root.children.items():
            visits[action] = child.visits

        return int(np.argmax(visits))
