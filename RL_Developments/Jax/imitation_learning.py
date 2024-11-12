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
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Tuple, Callable

class ImitationLearningModel(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

class BehavioralCloning:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = ImitationLearningModel(action_dim)
        self.params = self.network.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    def get_action(self, state: jnp.ndarray) -> int:
        logits = self.network.apply(self.params, state)
        return jnp.argmax(logits).item()

    def update(self, states: jnp.ndarray, actions: jnp.ndarray):
        loss, grads = jax.value_and_grad(self._loss_fn)(self.params, states, actions)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss

    def _loss_fn(self, params, states, actions):
        logits = self.network.apply(params, states)
        return optax.softmax_cross_entropy_with_integer_labels(logits, actions).mean()

class DAgger:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = ImitationLearningModel(action_dim)
        self.params = self.network.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    def get_action(self, state: jnp.ndarray) -> int:
        logits = self.network.apply(self.params, state)
        return jnp.argmax(logits).item()

    def update(self, states: jnp.ndarray, actions: jnp.ndarray):
        loss, grads = jax.value_and_grad(self._loss_fn)(self.params, states, actions)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss

    def _loss_fn(self, params, states, actions):
        logits = self.network.apply(params, states)
        return optax.softmax_cross_entropy_with_integer_labels(logits, actions).mean()

    def train(self, env: Callable, expert_policy: Callable, n_iterations: int, n_episodes: int):
        for iteration in range(n_iterations):
            # Collect data using the current policy
            states, actions = [], []
            for _ in range(n_episodes):
                state = env.reset()
                done = False
                while not done:
                    action = self.get_action(state)
                    states.append(state)
                    actions.append(expert_policy(state))  # Query expert for the correct action
                    state, _, done, _ = env.step(action)

            # Update the policy
            states = jnp.array(states)
            actions = jnp.array(actions)
            loss = self.update(states, actions)
            print(f"Iteration {iteration + 1}/{n_iterations}, Loss: {loss}")
