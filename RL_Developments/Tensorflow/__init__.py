# MIT License
#
# Copyright (c) 2024 VishwamAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# TensorFlow version
# __init__.py for the Reinforcement Learning Framework
__version__ = "0.0.1"

# Core implementations
from .td3 import TD3Network, TD3Agent
from .model_based import WorldModel, ModelBasedAgent
from .agentic_behavior import AgenticBehavior
from .cognition import CognitionAgent
from .dqn import DQNAgent
from .ppo import PPOAgent, Actor as PPOActor, Critic as PPOCritic
from .advanced_rl_algorithms import SACAgent
from .curiosity_driven_exploration import CuriosityDrivenAgent
from .imitation_learning import BehavioralCloning, DAgger
from .meta_learning import MAML, Reptile
from .model_based_rl import MBPOAgent, MCTSAgent
from .model_free_rl import QLearningAgent, SARSAgent
from .multi_agent_rl import MultiAgentRL
from .q_network import QNetwork
from .reinforcement_learning_advancements import AdvancedRLAgent
from .self_curing_rl import SelfCuringRL
from .policy_gradient import PolicyGradient
from .rl_algorithms import QLearning, ActorCritic
from .rl_module import (
    PPOBuffer,
    discount_cumsum,
    Actor,
    Critic,
    RLAgent,
    select_action,
    update_ppo,
    get_minibatches,
    train_rl_agent
)
from .hierarchical_rl import HierarchicalRL
from .offline_rl import OfflineRL
from .rl_with_causal_reasoning import RLWithCausalReasoning
from .q_rl import QRLAgent
from .utils import (
    get_device_strategy,
    update_target_network,
    create_optimizer
)

__all__ = [
    # Core implementations
    'TD3Network',
    'TD3Agent',
    'WorldModel',
    'ModelBasedAgent',
    'AgenticBehavior',
    'CognitionAgent',
    'DQNAgent',
    'PPOAgent',
    'PPOActor',
    'PPOCritic',
    'SACAgent',
    'CuriosityDrivenAgent',
    'BehavioralCloning',
    'DAgger',
    'MAML',
    'Reptile',
    'MBPOAgent',
    'MCTSAgent',
    'QLearningAgent',
    'SARSAgent',
    'MultiAgentRL',
    'QNetwork',
    'AdvancedRLAgent',
    'SelfCuringRL',
    'PolicyGradient',
    'QLearning',
    'ActorCritic',

    # RL Module components
    'PPOBuffer',
    'discount_cumsum',
    'Actor',
    'Critic',
    'RLAgent',
    'select_action',
    'update_ppo',
    'get_minibatches',
    'train_rl_agent',

    # Advanced RL components
    'HierarchicalRL',
    'OfflineRL',
    'RLWithCausalReasoning',
    'QRLAgent',

    # Utility functions
    'get_device_strategy',
    'update_target_network',
    'create_optimizer',
]
