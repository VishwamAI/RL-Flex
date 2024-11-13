"""Utility functions for TensorFlow implementations in RL-Flex.

This module provides utility functions for device handling and other common operations
used across different reinforcement learning algorithms in the TensorFlow implementation.
"""

import tensorflow as tf
from typing import Union, Optional


def get_device_strategy() -> Union[tf.distribute.MirroredStrategy, tf.distribute.OneDeviceStrategy]:
    """Get appropriate TensorFlow distribution strategy based on available hardware.

    Returns a MirroredStrategy for multi-GPU setups or when a single GPU is available,
    and OneDeviceStrategy for CPU-only environments. Automatically configures GPU memory
    growth to prevent TensorFlow from allocating all available memory at startup.

    Returns:
        TensorFlow distribution strategy appropriate for the current hardware setup.

    Note:
        Matches device handling patterns used in JAX and PyTorch implementations.
    """
    # Configure GPU memory growth before initializing strategy
    set_memory_growth()

    # Check for available GPUs
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Use MirroredStrategy for GPU training
            return tf.distribute.MirroredStrategy()
        except RuntimeError as e:
            print(f"Error initializing GPU strategy: {e}")
            print("Falling back to CPU...")
            return tf.distribute.OneDeviceStrategy("/cpu:0")

    # Use CPU if no GPUs are available
    return tf.distribute.OneDeviceStrategy("/cpu:0")


def set_memory_growth():
    """Configure GPU memory growth to prevent TensorFlow from allocating all memory.

    This prevents TensorFlow from allocating all available GPU memory at startup,
    allowing for more efficient memory usage in multi-process scenarios.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Memory growth must be set before GPUs are initialized: {e}")


def create_optimizer(learning_rate: float = 3e-4,
                    optimizer_type: str = 'adam',
                    **kwargs) -> tf.keras.optimizers.Optimizer:
    """Create a TensorFlow optimizer with specified parameters.

    Args:
        learning_rate: Learning rate for the optimizer
        optimizer_type: Type of optimizer to create ('adam', 'rmsprop', or 'sgd')
        **kwargs: Additional arguments to pass to the optimizer

    Returns:
        Configured TensorFlow optimizer instance

    Raises:
        ValueError: If optimizer_type is not recognized
    """
    optimizer_type = optimizer_type.lower()
    if optimizer_type == 'adam':
        return tf.keras.optimizers.Adam(learning_rate, **kwargs)
    elif optimizer_type == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate, **kwargs)
    elif optimizer_type == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def update_target_network(target_vars: list,
                         source_vars: list,
                         tau: float = 0.005) -> None:
    """Perform soft update of target network parameters.

    Performs the update: θ_target = (1 - τ) * θ_target + τ * θ_source

    Args:
        target_vars: List of target network variables
        source_vars: List of source network variables
        tau: Soft update coefficient (0 < τ ≤ 1)
    """
    for target, source in zip(target_vars, source_vars):
        target.assign(target * (1 - tau) + source * tau)
