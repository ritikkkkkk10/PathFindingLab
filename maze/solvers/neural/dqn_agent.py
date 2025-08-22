# solvers/neural/dqn_agent.py

import torch
import torch.nn.functional as F
from .dqn_network import DQNNetwork
from .nn_config import *

class DQNAgent:
    """
    DQN Agent that manages the main network and target network.
    Handles training and action selection.
    """
    
    def __init__(self):
        """Initialize DQN agent with main and target networks."""
        # Main network (updated every step)
        self.q_network = DQNNetwork().to(DEVICE)
        
        # Target network (updated periodically for stability)
        self.target_network = DQNNetwork().to(DEVICE)
        
        # Copy weights to target network
        self.update_target_network()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        
        # Training tracking
        self.step_count = 0
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_step(self, states, actions, rewards, next_states, dones):
        """Perform one training step using a batch of experiences."""
        # Convert to tensors
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        # ✅ This works with NumPy 2.x
        dones = torch.tensor([bool(x) for x in dones], dtype=torch.bool).to(DEVICE)

        
        # Get current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (GAMMA * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
