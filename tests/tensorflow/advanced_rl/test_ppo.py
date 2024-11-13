import tensorflow as tf
import numpy as np
from RL_Developments.Tensorflow.ppo import PPOAgent, Actor, Critic

def test_networks_initialization():
    """Test Actor and Critic network initialization."""
    state_dim = 4
    action_dim = 2

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)

    # Test forward pass
    batch_size = 32
    states = tf.random.normal((batch_size, state_dim))

    mean, log_std = actor(states)
    values = critic(states)

    assert mean.shape == (batch_size, action_dim)
    assert log_std.shape == (batch_size, action_dim)
    assert values.shape == (batch_size, 1)

def test_ppo_agent_initialization():
    """Test PPO Agent initialization."""
    state_dim = 4
    action_dim = 2

    agent = PPOAgent(state_dim, action_dim)

    # Verify network initialization
    assert isinstance(agent.actor, Actor)
    assert isinstance(agent.critic, Critic)


def test_ppo_agent_update():
    """Test PPO Agent update."""
    state_dim = 4
    action_dim = 2
    batch_size = 32

    agent = PPOAgent(state_dim, action_dim)

    # Create dummy batch
    states = tf.random.normal((batch_size, state_dim))
    actions = tf.random.normal((batch_size, action_dim))
    old_log_probs = tf.random.normal((batch_size,))
    advantages = tf.random.normal((batch_size,))
    returns = tf.random.normal((batch_size,))

    # Update agent
    metrics = agent.update(states, actions, old_log_probs, advantages, returns)

    assert "actor_loss" in metrics
    assert "critic_loss" in metrics
    assert "entropy" in metrics
