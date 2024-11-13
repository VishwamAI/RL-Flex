# Cognition Model Documentation

## Overview

This document provides an overview of the Cognition Model implemented using JAX, Flax, and Optax libraries. The model consists of two main components: `CognitionNetwork` and `CognitionAgent`.

## CognitionNetwork Class

The `CognitionNetwork` class is a neural network module designed to process input data, maintain an internal memory state, and produce outputs for decision-making.

### Key Features

- **Input Dimension (`input_dim`)**: Size of the input features.
- **Memory Dimension (`memory_dim`)**: Size of the internal memory state (default is 128).
- **Output Dimension (`output_dim`)**: Size of the output layer for decision-making (default is 10).

### Forward Pass (`__call__` method)

1. **Attention Mechanism**:
   - Applies a dense layer to the input `x` to compute attention weights.
   - Uses a softmax function to normalize these weights.
   - **Code**:
     ```python
     attention_weights = nn.Dense(self.memory_dim)(x)
     attention_weights = nn.softmax(attention_weights)
     ```

2. **Memory State Update**:
   - Updates the `memory_state` by adding the product of attention weights and a transformed input.
   - **Code**:
     ```python
     updated_memory = memory_state + attention_weights * nn.Dense(self.memory_dim)(x)
     ```

3. **Information Processing**:
   - Passes the updated memory through a multi-layer perceptron (MLP) with ReLU activations.
   - **Code**:
     ```python
     x = nn.Dense(128)(updated_memory)
     x = nn.relu(x)
     x = nn.Dense(64)(x)
     x = nn.relu(x)
     ```

4. **Decision-Making Output**:
   - Generates the final output using a dense layer.
   - **Code**:
     ```python
     output = nn.Dense(self.output_dim)(x)
     ```

5. **Return Values**:
   - Outputs the decision-making result and the updated memory state.

## CognitionAgent Class

The `CognitionAgent` class manages the cognition network, including initialization, training, and prediction.

### Initialization (`__init__` method)

- **Parameters**:
  - `input_dim`: Input feature size.
  - `output_dim`: Output size for decisions.
  - `learning_rate`: Learning rate for the optimizer (default is 1e-3).

- **Processes**:
  - Initializes the cognition network and memory state.
  - Sets up the optimizer using Adam from Optax.
  - **Code**:
    ```python
    self.memory_state = jnp.zeros((1, 128))
    self.params = self.cognition_network.init(...)
    self.optimizer = optax.adam(learning_rate)
    self.opt_state = self.optimizer.init(self.params)
    ```

### Methods

1. **Forward Pass (`forward`)**:
   - Runs the network's forward pass with the current parameters.
   - Uses JIT compilation for efficiency.
   - **Code**:
     ```python
     @jax.jit
     def forward(self, params, x, memory_state):
         return self.cognition_network.apply(params, x, memory_state)
     ```

2. **Update (`update`)**:
   - Computes the loss and gradients.
   - Updates the network parameters using the optimizer.
   - **Code**:
     ```python
     def loss_fn(params):
         predicted, _ = self.cognition_network.apply(params, x, self.memory_state)
         loss = jnp.mean((predicted - y) ** 2)
         return loss
     loss, grads = jax.value_and_grad(loss_fn)(params)
     ```

3. **Training (`train`)**:
   - Trains the network over a specified number of epochs.
   - **Code**:
     ```python
     for epoch in range(epochs):
         self.params, self.opt_state, loss = self.update(...)
         print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
     ```

4. **Prediction (`predict`)**:
   - Generates predictions for new input data.
   - Updates the internal memory state.
   - **Code**:
     ```python
     output, self.memory_state = self.forward(self.params, x, self.memory_state)
     return output
     ```

5. **Model Saving and Loading**:
   - **Saving**: `save_model(self, path)`
   - **Loading**: `load_model(self, path)`

## Example Usage

```python
# Initialize the cognition agent
agent = CognitionAgent(input_dim=your_input_size, output_dim=your_output_size)

# Prepare your dataset (list of input-target tuples)
dataset = [ (input1, target1), (input2, target2), ... ]

# Train the agent
agent.train(dataset, epochs=100)

# Make a prediction
new_input = jnp.array([...])  # Your new input data
prediction = agent.predict(new_input)
```
## Conclusion
The CognitionNetwork and CognitionAgent classes together form a powerful model for tasks that require attention mechanisms and memory retention. By understanding each component, you can modify and extend the code to suit specific applications in areas such as sequence modeling, time-series prediction, or reinforcement learning.