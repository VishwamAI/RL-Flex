import pytest
import tensorflow as tf
import numpy as np
from RL_Developments.Tensorflow.model_based import WorldModel, ModelBasedAgent
from RL_Developments.Tensorflow.td3 import TD3Agent
from RL_Developments.Tensorflow.utils import get_device_strategy

def test_world_model_initialization():
    state_dim = 24
    action_dim = 4

    with get_device_strategy().scope():
        model = WorldModel(state_dim, action_dim)
        assert len(model.dynamics_models) == 5  # Default ensemble size

        # Test forward pass
        batch_size = 32
        states = tf.random.normal((batch_size, state_dim))
        actions = tf.random.uniform((batch_size, action_dim), -1, 1)

        next_states, rewards, uncertainties = model(states, actions)
        assert next_states.shape == (batch_size, state_dim)
        assert rewards.shape == (batch_size, 1)
        assert uncertainties.shape == (batch_size, state_dim)

def test_model_based_agent():
    state_dim = 24
    action_dim = 4
    batch_size = 32

    with get_device_strategy().scope():
        base_agent = TD3Agent(state_dim, action_dim)
        agent = ModelBasedAgent(state_dim, action_dim, base_agent)

        # Test action selection
        state = tf.random.normal((1, state_dim))
        action = agent.get_action(state)
        assert action.shape == (1, action_dim)

        # Test world model update
        states = tf.random.normal((batch_size, state_dim))
        actions = tf.random.uniform((batch_size, action_dim), -1, 1)
        next_states = tf.random.normal((batch_size, state_dim))
        rewards = tf.random.normal((batch_size, 1))

        losses = agent.update_world_model(states, actions, next_states, rewards)
        assert isinstance(losses, dict)
        assert 'state_loss' in losses
        assert 'reward_loss' in losses
