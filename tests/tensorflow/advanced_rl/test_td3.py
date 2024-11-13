import pytest
import tensorflow as tf
import numpy as np
from RL_Developments.Tensorflow.td3 import TD3Network, TD3Agent
from RL_Developments.Tensorflow.utils import get_device_strategy

def test_td3_network_initialization():
    state_dim = 24
    action_dim = 4
    
    with get_device_strategy().scope():
        network = TD3Network(state_dim, action_dim)
        assert isinstance(network.actor, tf.keras.Model)
        assert isinstance(network.critic1, tf.keras.Model)
        assert isinstance(network.critic2, tf.keras.Model)

def test_td3_agent_update():
    state_dim = 24
    action_dim = 4
    batch_size = 32
    
    with get_device_strategy().scope():
        agent = TD3Agent(state_dim, action_dim)
        states = tf.random.normal((batch_size, state_dim))
        actions = tf.random.uniform((batch_size, action_dim), -1, 1)
        rewards = tf.random.normal((batch_size, 1))
        next_states = tf.random.normal((batch_size, state_dim))
        dones = tf.zeros((batch_size, 1))
        
        losses = agent.update(states, actions, rewards, next_states, dones)
        assert isinstance(losses, dict)
        assert 'actor_loss' in losses
        assert 'critic_loss' in losses
