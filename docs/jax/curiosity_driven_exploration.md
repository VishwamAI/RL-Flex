# Curiosity-Driven Reinforcement Learning Agent

This project implements a curiosity-driven reinforcement learning agent using JAX, Flax, and Optax libraries. The agent is trained to solve the `MountainCar-v0` environment from OpenAI Gym.

## Overview

The implementation includes:

- **Intrinsic Curiosity Module (ICM):** Encourages exploration by providing intrinsic rewards based on prediction errors.
- **Novelty Detector:** Identifies novel states to enhance the exploration strategy.
- **Curiosity-Driven Agent:** Combines both extrinsic and intrinsic rewards to learn optimal policies.
- **Training Loop:** Trains the agent over multiple episodes and updates the networks accordingly.

## Components

### ICM Class

The `ICM` class implements the intrinsic curiosity module:

- **Feature Encoder:** Encodes state representations using a neural network.
- **Inverse Model:** Predicts the action taken given the state and next state features.
- **Forward Model:** Predicts the next state feature based on the current state feature and action.

**Methods:**

- `__call__(state, next_state, action)`: Computes predicted actions and next state features.
- `compute_intrinsic_reward(state, next_state, action)`: Calculates intrinsic rewards based on prediction errors.

### NoveltyDetector Class

The `NoveltyDetector` class tracks visited states to detect novelty:

- Maintains a memory buffer of past states.
- Computes novelty by measuring the difference between the current state and memory.
- Determines if a state is novel based on a predefined threshold.

**Methods:**

- `compute_novelty(state)`: Calculates novelty score for a given state.
- `update_memory(state)`: Updates the memory with new states.
- `is_novel(state)`: Checks if the state is considered novel.

### CuriosityDrivenAgent Class

The `CuriosityDrivenAgent` integrates the ICM and Novelty Detector:

- **Actor Network:** Determines action probabilities based on the current state.
- **Critic Network:** Estimates the value function for state-value predictions.
- **ICM Module:** Generates intrinsic rewards to encourage exploration.
- **Novelty Detector:** Enhances exploration by identifying novel states.

**Methods:**

- `act(state)`: Selects an action using the actor network.
- `update(state, action, reward, next_state, done)`: Updates actor, critic, and ICM networks based on the experience.

### Training Function

`train_curiosity_agent(env, agent, num_episodes=1000)`

- Initializes the environment and agent.
- Runs the training loop for a specified number of episodes.
- Collects experiences and updates the agent.
- Logs performance metrics every 10 episodes.

### Main Function

`main()`

- Sets up the `MountainCar-v0` environment.
- Initializes the `CuriosityDrivenAgent`.
- Starts the training process by calling `train_curiosity_agent`.

## Requirements

- Python 3.x
- JAX
- Flax
- Optax
- NumPy
- Gymnasium (OpenAI Gym)

## Usage

1. Install the required packages.
2. Run the script to start training:

    ```bash
    python curiosity_agent.py
    ```

## License

This project is licensed under the MIT License.