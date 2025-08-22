# solvers/neural/dqn_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn_config import *

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for maze solving.
    Takes maze state as input and outputs Q-values for each action.
    """
    
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE, 
                 hidden1=HIDDEN_LAYER_1, hidden2=HIDDEN_LAYER_2):
        """
        Initialize the DQN network.
        
        Args:
            state_size: Size of input state vector
            action_size: Number of possible actions
            hidden1: Size of first hidden layer
            hidden2: Size of second hidden layer
        """
        super(DQNNetwork, self).__init__()
        
        # Define network layers
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_size)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for better training stability."""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor [batch_size, state_size]
            
        Returns:
            Q-values for each action [batch_size, action_size]
        """
        # Ensure input is a tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        # Forward pass with ReLU activations
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)  # No activation on output layer
        
        return q_values
    
    def get_action(self, state, epsilon=0.0):
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration probability
            
        Returns:
            action: Selected action (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
        """
        # Random action with probability epsilon
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, ACTION_SIZE, (1,)).item()
        
        # Otherwise, choose best action
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax().item()
    
    def save_model(self, filepath):
        """Save the trained model."""
        torch.save(self.state_dict(), filepath)
    
    def load_model(self, filepath):
        """Load a pre-trained model."""
        self.load_state_dict(torch.load(filepath, map_location=DEVICE))
        self.eval()


