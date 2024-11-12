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
import torch
from typing import List, Tuple
import torch.nn as nn
import torch.optim as optim

class MAMLModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MAMLModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class MAML:
    def __init__(self, input_dim, output_dim, alpha=0.01, beta=0.001):
        self.model = MAMLModel(input_dim, output_dim)
        self.alpha = alpha
        self.beta = beta
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=self.beta)
    
    def inner_update(self, x, y):
        adapted_model = MAMLModel(self.model.net[0].in_features, self.model.net[-1].out_features)
        adapted_model.load_state_dict(self.model.state_dict())
        loss_fn = nn.MSELoss()
        adapted_optimizer = optim.SGD(adapted_model.parameters(), lr=self.alpha)
        adapted_optimizer.zero_grad()
        preds = adapted_model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        adapted_optimizer.step()
        return adapted_model, loss.item()
    
    def outer_update(self, tasks):
        losses = []
        self.meta_optimizer.zero_grad()
        for support_x, support_y, query_x, query_y in tasks:
            adapted_model, _ = self.inner_update(support_x, support_y)
            preds = adapted_model(query_x)
            loss = nn.functional.mse_loss(preds, query_y)
            loss.backward()
            losses.append(loss.item())
        self.meta_optimizer.step()
        return sum(losses) / len(losses)
    
    def meta_train(self, tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], num_epochs):
        for epoch in range(num_epochs):
            loss = self.outer_update(tasks)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss}")

class Reptile:
    def __init__(self, input_dim, output_dim, inner_lr=0.01, meta_lr=0.001):
        self.model = MAMLModel(input_dim, output_dim)
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
    
    def inner_update(self, x, y, num_steps):
        adapted_model = MAMLModel(self.model.net[0].in_features, self.model.net[-1].out_features)
        adapted_model.load_state_dict(self.model.state_dict())
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        for _ in range(num_steps):
            optimizer.zero_grad()
            preds = adapted_model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
        return adapted_model
    
    def outer_update(self, tasks, num_inner_steps):
        updated_state_dicts = []
        for x, y in tasks:
            adapted_model = self.inner_update(x, y, num_inner_steps)
            updated_state_dicts.append(adapted_model.state_dict())
        avg_state_dict = {}
        for key in self.model.state_dict().keys():
            avg_state_dict[key] = sum([state_dict[key] for state_dict in updated_state_dicts]) / len(updated_state_dicts)
        new_state_dict = {}
        for key in self.model.state_dict().keys():
            new_state_dict[key] = self.model.state_dict()[key] + self.meta_lr * (avg_state_dict[key] - self.model.state_dict()[key])
        self.model.load_state_dict(new_state_dict)
    
    def meta_train(self, tasks: List[Tuple[torch.Tensor, torch.Tensor]], num_epochs, num_inner_steps):
        for epoch in range(num_epochs):
            self.outer_update(tasks, num_inner_steps)
            val_losses = []
            for x, y in tasks:
                adapted_model = self.inner_update(x, y, num_inner_steps)
                preds = adapted_model(x)
                loss = nn.functional.mse_loss(preds, y)
                val_losses.append(loss.item())
            print(f"Epoch {epoch + 1}/{num_epochs}, Mean Validation Loss: {sum(val_losses) / len(val_losses)}")