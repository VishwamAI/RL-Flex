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
import torch.nn as nn

# Define the Cognition Network
class CognitionNetwork(nn.Module):
    def __init__(self, input_dim, memory_dim=128, output_dim=10):
        super(CognitionNetwork, self).__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.output_dim = output_dim

        # Attention layers
        self.attention_dense = nn.Linear(input_dim, memory_dim)
        self.attention_softmax = nn.Softmax(dim=-1)

        # Memory update layer
        self.memory_dense = nn.Linear(input_dim, memory_dim)

        # Processing layers
        self.fc1 = nn.Linear(memory_dim, 128)
        self.fc2 = nn.Linear(128, 64)

        # Output layer
        self.output_layer = nn.Linear(64, output_dim)

    def forward(self, x, memory_state):
        # Attention mechanism
        attention_weights = self.attention_dense(x)
        attention_weights = self.attention_softmax(attention_weights)

        # Update memory state
        updated_memory = memory_state + attention_weights * self.memory_dense(x)

        # Process information
        x = self.fc1(updated_memory)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)

        # Decision-making output
        output = self.output_layer(x)
        return output, updated_memory

# Cognition Agent using the Cognition Network
class CognitionAgent:
    def __init__(self, input_dim, output_dim, learning_rate=1e-3):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize the cognition network
        self.cognition_network = CognitionNetwork(input_dim, output_dim=output_dim)
        self.memory_state = torch.zeros(1, 128)  # Initialize memory state

        # Initialize parameters and optimizer
        self.optimizer = torch.optim.Adam(self.cognition_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        output, self.memory_state = self.cognition_network(x, self.memory_state)
        return output

    def update(self, x, y):
        self.optimizer.zero_grad()
        output = self.forward(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, dataset, epochs=100):
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in dataset:
                inputs = torch.tensor(inputs, dtype=torch.float32)
                targets = torch.tensor(targets, dtype=torch.float32)
                loss = self.update(inputs, targets)
                total_loss += loss
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

    def predict(self, x):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            output = self.forward(x)
        return output

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.cognition_network.state_dict(),
            'memory_state': self.memory_state
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.cognition_network.load_state_dict(checkpoint['model_state_dict'])
        self.memory_state = checkpoint['memory_state']