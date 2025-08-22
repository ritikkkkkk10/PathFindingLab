# solvers/neural/replay_buffer.py

import random
import numpy as np
from collections import deque, namedtuple
from .nn_config import REPLAY_BUFFER_SIZE, BATCH_SIZE

# Define experience tuple structure
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.
    Stores transitions and samples random batches for training.
    """
    
    def __init__(self, buffer_size=REPLAY_BUFFER_SIZE, batch_size=BATCH_SIZE):
        """
        Initialize replay buffer.
        
        Args:
            buffer_size: Maximum number of experiences to store
            batch_size: Size of training batches to sample
        """
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state reached
            done: Whether episode terminated
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size=None):
        """
        Sample a random batch of experiences.
        
        Args:
            batch_size: Size of batch to sample (uses default if None)
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Sample random experiences
        experiences = random.sample(self.buffer, batch_size)
        
        # Separate components
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current size of internal buffer."""
        return len(self.buffer)
    
    def is_ready(self, min_size=None):
        """
        Check if buffer has enough experiences for training.
        
        Args:
            min_size: Minimum buffer size required (uses batch_size if None)
            
        Returns:
            bool: True if buffer is ready for sampling
        """
        if min_size is None:
            min_size = self.batch_size
        return len(self.buffer) >= min_size
    
    def clear(self):
        """Clear all experiences from buffer."""
        self.buffer.clear()
    
    def get_statistics(self):
        """
        Get buffer statistics for monitoring.
        
        Returns:
            dict: Statistics about the buffer contents
        """
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'avg_reward': 0,
                'completion_rate': 0
            }
        
        rewards = [exp.reward for exp in self.buffer]
        completions = [exp.done for exp in self.buffer]
        
        return {
            'size': len(self.buffer),
            'avg_reward': np.mean(rewards),
            'completion_rate': np.mean(completions),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards)
        }

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Enhanced replay buffer with prioritized sampling.
    Samples experiences with higher TD-error more frequently.
    """
    
    def __init__(self, buffer_size=REPLAY_BUFFER_SIZE, batch_size=BATCH_SIZE, alpha=0.6):
        """
        Initialize prioritized replay buffer.
        
        Args:
            buffer_size: Maximum number of experiences to store
            batch_size: Size of training batches to sample
            alpha: Prioritization strength (0 = uniform, 1 = full prioritization)
        """
        super().__init__(buffer_size, batch_size)
        self.alpha = alpha
        self.priorities = deque(maxlen=buffer_size)
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done, td_error=None):
        """
        Add an experience with priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state reached
            done: Whether episode terminated
            td_error: TD-error for prioritization (uses max if None)
        """
        super().add(state, action, reward, next_state, done)
        
        # Set priority based on TD-error
        if td_error is None:
            priority = self.max_priority
        else:
            priority = (abs(td_error) + 1e-5) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
        
        self.priorities.append(priority)
    
    def sample(self, batch_size=None, beta=0.4):
        """
        Sample experiences based on priorities.
        
        Args:
            batch_size: Size of batch to sample
            beta: Importance sampling correction strength
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize
        
        # Separate components
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Indices of sampled experiences
            td_errors: New TD-errors for priority calculation
        """
        for i, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-5) ** self.alpha
            self.priorities[i] = priority
            self.max_priority = max(self.max_priority, priority)
