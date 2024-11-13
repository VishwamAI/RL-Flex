import pytest
import tensorflow as tf
import numpy as np
from RL_Developments.Tensorflow.utils import get_device_strategy, update_target_network, create_optimizer

def test_device_strategy():
    strategy = get_device_strategy()
    assert isinstance(strategy, tf.distribute.Strategy)

def test_optimizer_creation():
    optimizer = create_optimizer(learning_rate=3e-4)
    assert isinstance(optimizer, tf.keras.optimizers.Optimizer)
    assert optimizer.learning_rate.numpy() == 3e-4

def test_target_network_update():
    with get_device_strategy().scope():
        source = tf.keras.Sequential([tf.keras.layers.Dense(64)])
        target = tf.keras.Sequential([tf.keras.layers.Dense(64)])

        # Initialize networks
        dummy_input = tf.random.normal((1, 32))
        source(dummy_input)
        target(dummy_input)

        initial_weights = target.get_weights()[0].copy()
        update_target_network(target.variables, source.variables, tau=1.0)

        # Check if weights were updated
        assert not np.array_equal(initial_weights, target.get_weights()[0])
