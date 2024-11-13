import tensorflow as tf
import tensorflow_probability as tfp
import gym
import numpy as np

print("\nPackage Versions:")
print("TensorFlow Version:", tf.__version__)
print("TensorFlow Probability Version:", tfp.__version__)
print("Gym Version:", gym.__version__)
print("NumPy Version:", np.__version__)

print("\nDevice Configuration:")
physical_devices = tf.config.list_physical_devices()
print("Available Physical Devices:", physical_devices)

# Basic TensorFlow CPU Operations Test
print("\nBasic TensorFlow Operations Test:")
with tf.device('/CPU:0'):
    # Matrix operations
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
    print("Matrix Addition:", tf.add(a, b).numpy())

    # Neural Network Components
    dense = tf.keras.layers.Dense(2)
    input_data = tf.constant([[1.0, 2.0, 3.0]])
    output = dense(input_data)
    print("Dense Layer Output Shape:", output.shape)

# Test Probability Distributions
print("\nProbability Distribution Test:")
dist = tfp.distributions.Normal(loc=0., scale=1.)
samples = dist.sample(5)
print("Normal Distribution Samples:", samples.numpy())

# Test Environment Creation
print("\nGym Environment Test:")
env = gym.make('CartPole-v1')
print("Environment Action Space:", env.action_space)
print("Environment Observation Space:", env.observation_space)

print("\nEnvironment Setup Complete!")
