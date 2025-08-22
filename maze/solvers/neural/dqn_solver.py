# solvers/neural/dqn_solver.py

import torch
import numpy as np
import random
from typing import List, Tuple, Optional
import time

from ..solver_base import MazeSolver
from .dqn_agent import DQNAgent 
from .replay_buffer import ReplayBuffer
from .nn_config import *

class DQNMazeSolver(MazeSolver):
    """
    Deep Q-Network maze solver that learns to navigate mazes through reinforcement learning.
    Integrates with the existing MazeSolver interface.
    """
    
    def __init__(self, maze=None, start_pos=None, end_pos=None):
        """
        Initialize DQN maze solver.
        
        Args:
            maze: Maze object to solve
            start_pos: Starting position (row, col)
            end_pos: Target position (row, col)
        """
        super().__init__(maze)
        
        self.maze = maze
        self.start_pos = start_pos or (0, 0)
        self.end_pos = end_pos or (maze.height-1, maze.width-1) if maze else (9, 9)
        
        # Initialize DQN components
        self.agent = DQNAgent()
        self.replay_buffer = ReplayBuffer()
        
        # Training state
        self.current_episode = 0
        self.epsilon = EPSILON_START
        self.training_history = []
        self.best_path = None
        self.best_reward = float('-inf')
        
        # Visited tracking for visualization
        self.visited_cells = set()
        
    def _get_state(self, position: Tuple[int, int]) -> np.ndarray:
        """
        Enhanced state representation including local maze information.
        """
        if not self.maze:
            return np.array([0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        
        row, col = position
        
        # Normalize positions
        norm_x = col / (self.maze.width - 1)
        norm_y = row / (self.maze.height - 1)
        goal_x = self.end_pos[1] / (self.maze.width - 1)
        goal_y = self.end_pos[0] / (self.maze.height - 1)
        
        # Add local maze information - which directions are valid
        can_go_up = 1.0 if self._is_valid_position((row-1, col)) else 0.0
        can_go_right = 1.0 if self._is_valid_position((row, col+1)) else 0.0
        can_go_down = 1.0 if self._is_valid_position((row+1, col)) else 0.0
        can_go_left = 1.0 if self._is_valid_position((row, col-1)) else 0.0
        
        return np.array([norm_x, norm_y, goal_x, goal_y, 
                        can_go_up, can_go_right, can_go_down, can_go_left], dtype=np.float32)

    
    def _get_action_position(self, position: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Get new position after taking an action.
        
        Args:
            position: Current position (row, col)
            action: Action to take (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            
        Returns:
            New position (row, col)
        """
        row, col = position
        
        if action == 0:  # UP
            return (row - 1, col)
        elif action == 1:  # RIGHT
            return (row, col + 1)
        elif action == 2:  # DOWN
            return (row + 1, col)
        elif action == 3:  # LEFT
            return (row, col - 1)
        
        return position
    
    def _is_valid_position(self, position: Tuple[int, int]) -> bool:
        """
        Check if position is valid (within bounds and not a wall).
        
        Args:
            position: Position to check (row, col)
            
        Returns:
            True if position is valid
        """
        row, col = position
        
        if not self.maze:
            return True
            
        # Check bounds
        if row < 0 or row >= self.maze.height or col < 0 or col >= self.maze.width:
            return False
            
        # Check if it's a wall
        return not self.maze.is_wall(row, col)
    
    def _calculate_reward(self, current_pos: Tuple[int, int], action: int, 
                     next_pos: Tuple[int, int], visited: set) -> float:
        """Improved reward structure based on successful implementations."""
        
        # Big positive reward for reaching goal
        if next_pos == self.end_pos:
            return 10.0  # Much higher than step penalties
        
        # Wall hit: negative reward but don't move agent
        if not self._is_valid_position(next_pos):
            return -1.0
        
        # Small negative reward for each step (encourages efficiency)
        step_reward = -0.1
        
        # Additional penalty for revisiting (but smaller)
        if next_pos in visited:
            step_reward -= 0.1
        
        return step_reward

    

    def _get_progressive_start_position(self, episode: int) -> Tuple[int, int]:
        """Start training near goal, gradually expand outward."""
        if episode < 100:
            # First 100 episodes: start very close to goal
            max_distance = 3
        elif episode < 300:
            # Next 200 episodes: moderate distance
            max_distance = 6
        else:
            # Later episodes: anywhere in maze
            return self.start_pos
        
        # Find positions within max_distance of goal
        goal_row, goal_col = self.end_pos
        valid_starts = []
        
        for row in range(max(0, goal_row - max_distance), 
                        min(self.maze.height, goal_row + max_distance + 1)):
            for col in range(max(0, goal_col - max_distance), 
                            min(self.maze.width, goal_col + max_distance + 1)):
                if not self.maze.is_wall(row, col) and (row, col) != self.end_pos:
                    valid_starts.append((row, col))
        
        return random.choice(valid_starts) if valid_starts else self.start_pos


    
    def _train_episode(self) -> Tuple[float, int, bool]:
        """
        Train the agent for one episode.
        
        Returns:
            Tuple of (total_reward, steps_taken, reached_goal)
        """
        # Use progressive starting position for first 300 episodes
        if self.current_episode < 300:
            current_pos = self._get_progressive_start_position(self.current_episode)
        else:
            current_pos = self.start_pos
        visited = set()
        visited.add(current_pos)
        
        total_reward = 0
        steps = 0
        episode_path = [current_pos]
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # Get current state
            state = self._get_state(current_pos)
            
            # Choose action using epsilon-greedy policy
            action = self.agent.q_network.get_action(state, self.epsilon)
            
            # Take action
            next_pos = self._get_action_position(current_pos, action)
            
            # Calculate reward
            reward = self._calculate_reward(current_pos, action, next_pos, visited)
            
            # Check if episode is done
            done = (next_pos == self.end_pos) or (not self._is_valid_position(next_pos))
            
            # Update position if valid
            if self._is_valid_position(next_pos):
                current_pos = next_pos
                visited.add(current_pos)
                episode_path.append(current_pos)
            
            # Get next state
            next_state = self._get_state(current_pos)
            
            # Store experience in replay buffer
            self.replay_buffer.add(state, action, reward, next_state, done)
            
            total_reward += reward
            steps += 1
            
            # Train the network if enough experiences
            if self.replay_buffer.is_ready(MIN_REPLAY_SIZE):
                self._train_network()
            
            # Check if episode ended
            if done:
                reached_goal = (next_pos == self.end_pos)
                
                # Update best path if this episode reached goal
                if reached_goal and (self.best_path is None or len(episode_path) < len(self.best_path)):
                    self.best_path = episode_path.copy()
                    self.best_reward = total_reward
                
                return total_reward, steps, reached_goal
        
        return total_reward, steps, False
    
    def _train_network(self):
        """Train the neural network using a batch of experiences."""
        if not self.replay_buffer.is_ready(BATCH_SIZE):
            return
        
        # Sample batch of experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        # Train the network
        loss = self.agent.train_step(states, actions, rewards, next_states, dones)
        
        return loss
    
    def solve(self, verbose: bool = True) -> Optional[List[Tuple[int, int]]]:
        """
        Solve the maze using DQN.
        
        Args:
            verbose: Whether to print training progress
            
        Returns:
            Path from start to goal, or None if no solution found
        """
        if verbose:
            print(f"Training DQN maze solver...")
            print(f"Maze size: {self.maze.width}x{self.maze.height}")
            print(f"Start: {self.start_pos}, Goal: {self.end_pos}")
        
        start_time = time.time()
        
        # Training loop
        for episode in range(MAX_EPISODES):
            self.current_episode = episode
            
            # Train one episode
            reward, steps, reached_goal = self._train_episode()
            
            # Update epsilon (decay exploration)
            self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
            
            # Update target network periodically
            if episode % TARGET_UPDATE_FREQ == 0:
                self.agent.update_target_network()
            
            # Track training history
            self.training_history.append({
                'episode': episode,
                'reward': reward,
                'steps': steps,
                'reached_goal': reached_goal,
                'epsilon': self.epsilon
            })
            
            # Print progress
            if verbose and episode % 50 == 0:
                success_rate = sum(1 for h in self.training_history[-50:] if h['reached_goal']) / min(50, len(self.training_history))
                print(f"Episode {episode}: Reward={reward:.2f}, Steps={steps}, "
                      f"Success Rate={success_rate:.2f}, Epsilon={self.epsilon:.3f}")
            
            # Early stopping if consistently reaching goal
            if episode >= 100:
                recent_success = sum(1 for h in self.training_history[-20:] if h['reached_goal'])
                if recent_success >= 18:  # 90% success in last 20 episodes
                    if verbose:
                        print(f"Early stopping at episode {episode} (high success rate)")
                    break
        
        training_time = time.time() - start_time
        
        if verbose:
            total_successes = sum(1 for h in self.training_history if h['reached_goal'])
            print(f"Training completed in {training_time:.2f}s")
            print(f"Total successful episodes: {total_successes}/{len(self.training_history)}")
        
        # Return best path found
        if self.best_path:
            self.visited_cells = set(self.best_path)
            return self.best_path
        
        return None
    
    def get_path(self) -> List[Tuple[int, int]]:
        """Get the solution path."""
        return self.best_path if self.best_path else []
    
    def get_visited(self) -> set:
        """Get visited cells for visualization."""
        return self.visited_cells
    
    def get_training_stats(self) -> dict:
        """Get training statistics."""
        if not self.training_history:
            return {}
        
        return {
            'episodes_trained': len(self.training_history),
            'success_rate': sum(1 for h in self.training_history if h['reached_goal']) / len(self.training_history),
            'average_reward': np.mean([h['reward'] for h in self.training_history]),
            'best_reward': self.best_reward,
            'final_epsilon': self.epsilon
        }
