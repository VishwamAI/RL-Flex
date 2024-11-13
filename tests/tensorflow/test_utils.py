import pytest
import tensorflow as tf
from RL_Developments.Tensorflow.utils import get_device_strategy, update_target_network, create_optimizer

def test_device_strategy():
    strategy = get_device_strategy()
    assert isinstance(strategy, (tf.distribute.Strategy))

    # Test strategy scope
    with strategy.scope():
        model = tf.keras.Sequential([tf.keras.layers.Dense(64)])
        assert model is not None

def test_target_network_update():
    with get_device_strategy().scope():
        source = tf.keras.Sequential([tf.keras.layers.Dense(64)])
        target = tf.keras.Sequential([tf.keras.layers.Dense(64)])

        # Initialize with different weights
        source.build((None, 32))
        target.build((None, 32))

        initial_target_weights = target.get_weights()[0].copy()
        update_target_network(target.variables, source.variables, tau=1.0)

        # Check if weights were updated
        assert not np.array_equal(initial_target_weights, target.get_weights()[0])

def test_optimizer_creation():
    optimizer = create_optimizer(learning_rate=3e-4)
    assert isinstance(optimizer, tf.keras.optimizers.Optimizer)
    assert optimizer.learning_rate.numpy() == 3e-4
