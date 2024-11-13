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

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from .utils import get_device_strategy, create_optimizer

class MAML(tf.keras.Model):
    """Model-Agnostic Meta-Learning (MAML) implementation."""

    def __init__(
        self,
        model_fn: Callable[[], tf.keras.Model],
        inner_learning_rate: float = 0.01,
        meta_learning_rate: float = 1e-3,
        first_order: bool = False,
        **kwargs
    ):
        """Initialize MAML.

        Args:
            model_fn: Function that returns a new model instance
            inner_learning_rate: Learning rate for inner loop optimization
            meta_learning_rate: Learning rate for meta-optimization
            first_order: Whether to use first-order approximation
            **kwargs: Additional arguments
        """
        super().__init__()
        self.model_fn = model_fn
        self.inner_learning_rate = inner_learning_rate
        self.first_order = first_order

        # Get device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create base model
            self.model = model_fn()
            # Create meta-optimizer
            self.meta_optimizer = create_optimizer(meta_learning_rate)

    def clone_model(self) -> tf.keras.Model:
        """Create a clone of the current model."""
        clone = self.model_fn()
        clone.set_weights(self.model.get_weights())
        return clone

    def inner_loop(
        self,
        support_data: Tuple[tf.Tensor, tf.Tensor],
        model: Optional[tf.keras.Model] = None,
        num_steps: int = 1
    ) -> tf.keras.Model:
        """Perform inner loop adaptation.

        Args:
            support_data: Tuple of (inputs, targets) for support set
            model: Optional model to adapt (creates clone if None)
            num_steps: Number of gradient steps

        Returns:
            Adapted model
        """
        if model is None:
            model = self.clone_model()

        support_inputs, support_targets = support_data

        for _ in range(num_steps):
            with tf.GradientTape() as tape:
                predictions = model(support_inputs, training=True)
                loss = tf.reduce_mean(
                    tf.keras.losses.mean_squared_error(support_targets, predictions)
                )

            gradients = tape.gradient(loss, model.trainable_variables)
            for var, grad in zip(model.trainable_variables, gradients):
                var.assign_sub(self.inner_learning_rate * grad)

        return model

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        """Forward pass through the model.

        Args:
            inputs: Input tensor
            training: Whether in training mode

        Returns:
            Model predictions
        """
        return self.model(inputs, training=training)

    def meta_update(
        self,
        tasks: List[Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]]
    ) -> Dict[str, float]:
        """Perform meta-update using MAML.

        Args:
            tasks: List of ((support_x, support_y), (query_x, query_y)) tuples

        Returns:
            Dictionary of metrics
        """
        meta_loss = 0.0

        with tf.GradientTape() as meta_tape:
            for (support_data, query_data) in tasks:
                # Inner loop adaptation
                adapted_model = self.inner_loop(support_data)

                # Compute loss on query set
                query_inputs, query_targets = query_data
                predictions = adapted_model(query_inputs, training=True)
                task_loss = tf.reduce_mean(
                    tf.keras.losses.mean_squared_error(query_targets, predictions)
                )
                meta_loss += task_loss

            meta_loss /= len(tasks)

        # Meta-optimization
        meta_gradients = meta_tape.gradient(meta_loss, self.model.trainable_variables)
        self.meta_optimizer.apply_gradients(
            zip(meta_gradients, self.model.trainable_variables)
        )

        return {"meta_loss": float(meta_loss)}

class Reptile(tf.keras.Model):
    """Reptile meta-learning algorithm implementation."""

    def __init__(
        self,
        model_fn: Callable[[], tf.keras.Model],
        inner_learning_rate: float = 0.01,
        meta_learning_rate: float = 1e-3,
        **kwargs
    ):
        """Initialize Reptile.

        Args:
            model_fn: Function that returns a new model instance
            inner_learning_rate: Learning rate for inner loop optimization
            meta_learning_rate: Learning rate for meta-optimization
            **kwargs: Additional arguments
        """
        super().__init__()
        self.model_fn = model_fn
        self.inner_learning_rate = inner_learning_rate
        self.meta_learning_rate = meta_learning_rate

        # Get device strategy
        self.strategy = get_device_strategy()

        with self.strategy.scope():
            # Create base model
            self.model = model_fn()

    def clone_model(self) -> tf.keras.Model:
        """Create a clone of the current model."""
        clone = self.model_fn()
        clone.set_weights(self.model.get_weights())
        return clone

    def inner_loop(
        self,
        support_data: Tuple[tf.Tensor, tf.Tensor],
        num_steps: int = 5
    ) -> List[tf.Tensor]:
        """Perform inner loop adaptation.

        Args:
            support_data: Tuple of (inputs, targets) for support set
            num_steps: Number of gradient steps

        Returns:
            Final weights after adaptation
        """
        model = self.clone_model()
        support_inputs, support_targets = support_data

        for _ in range(num_steps):
            with tf.GradientTape() as tape:
                predictions = model(support_inputs, training=True)
                loss = tf.reduce_mean(
                    tf.keras.losses.mean_squared_error(support_targets, predictions)
                )

            gradients = tape.gradient(loss, model.trainable_variables)
            for var, grad in zip(model.trainable_variables, gradients):
                var.assign_sub(self.inner_learning_rate * grad)

        return model.get_weights()

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        """Forward pass through the model.

        Args:
            inputs: Input tensor
            training: Whether in training mode

        Returns:
            Model predictions
        """
        return self.model(inputs, training=training)

    def meta_update(
        self,
        tasks: List[Tuple[tf.Tensor, tf.Tensor]]
    ) -> Dict[str, float]:
        """Perform meta-update using Reptile.

        Args:
            tasks: List of (inputs, targets) tuples for different tasks

        Returns:
            Dictionary of metrics
        """
        old_weights = self.model.get_weights()
        new_weights = [tf.zeros_like(w) for w in old_weights]
        total_loss = 0.0

        # Accumulate gradients from all tasks
        for task_data in tasks:
            # Inner loop adaptation
            adapted_weights = self.inner_loop(task_data)

            # Accumulate weight differences
            for new_w, adapted_w in zip(new_weights, adapted_weights):
                new_w.assign_add(adapted_w)

            # Compute task loss for monitoring
            inputs, targets = task_data
            predictions = self.model(inputs, training=True)
            total_loss += float(tf.reduce_mean(
                tf.keras.losses.mean_squared_error(targets, predictions)
            ))

        # Average the accumulated weights
        for new_w in new_weights:
            new_w.assign(new_w / len(tasks))

        # Update model weights using Reptile update rule
        for old_w, new_w in zip(self.model.trainable_variables, new_weights):
            old_w.assign_add(self.meta_learning_rate * (new_w - old_w))

        return {"meta_loss": total_loss / len(tasks)}
