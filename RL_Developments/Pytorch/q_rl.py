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
import numpy as np
import torch.nn as nn
import torch.optim as optim

# Define a Quantum-Inspired Q-Network
class QuantumQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_qubits=4, circuit_layers=2):
        super(QuantumQNetwork, self).__init__()
        self.num_qubits = num_qubits
        self.circuit_layers = circuit_layers
        
        # Quantum Encoding Layer
        self.quantum_encoding = nn.Linear(state_dim, num_qubits)
        
        # Parameterized Quantum Circuit Simulation
        self.quantum_layers = nn.ModuleList([nn.Linear(num_qubits, num_qubits) for _ in range(circuit_layers)])
        
        # Classical Layers
        self.fc1 = nn.Linear(num_qubits, 128)
        self.fc2 = nn.Linear(128, 128)
        self.q_values = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.sigmoid(self.quantum_encoding(x))
        
        for layer in self.quantum_layers:
            x = torch.relu(layer(x))
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_values = self.q_values(x)
        return q_values

# QRL Agent
class QRLAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Initialize the Q-network
        self.q_network = QuantumQNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def select_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()
    
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.q_network(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, dataset, epochs=100):
        for epoch in range(epochs):
            states, actions, rewards, next_states, dones = zip(*dataset)
            loss = self.update(states, actions, rewards, next_states, dones)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
    
    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)
    
    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path))