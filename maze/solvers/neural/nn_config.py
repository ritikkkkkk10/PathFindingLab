# solvers/neural/nn_config.py

"""
Configuration parameters for Deep Q-Network (DQN) maze solver.
Based on successful maze-solving DQN implementations from research.
"""

# Network Architecture
STATE_SIZE = 8             # [agent_x, agent_y, goal_x, goal_y]
ACTION_SIZE = 4             # [UP, RIGHT, DOWN, LEFT]
HIDDEN_LAYER_1 = 64         # First hidden layer size
HIDDEN_LAYER_2 = 64         # Second hidden layer size

# Training Hyperparameters
LEARNING_RATE = 0.001       # Learning rate for Adam optimizer
GAMMA = 0.99                # Discount factor for future rewards
EPSILON_START = 0.9        # Initial exploration rate
EPSILON_END = 0.05          # Final exploration rate
EPSILON_DECAY = 0.999     # Epsilon decay rate per episode

# Experience Replay
REPLAY_BUFFER_SIZE = 10000  # Maximum size of replay buffer
BATCH_SIZE = 32             # Training batch size
MIN_REPLAY_SIZE = 1000      # Minimum experiences before training starts

# Training Control
MAX_EPISODES = 2000          # Maximum training episodes
MAX_STEPS_PER_EPISODE = 200 # Maximum steps per episode
TARGET_UPDATE_FREQ = 10     # Update target network every N episodes

# Reward Structure (from successful implementations)
REWARD_GOAL = 1.0           # Reward for reaching goal
REWARD_STEP = -0.05         # Small penalty per step (encourages shorter paths)
REWARD_WALL = -1.0          # Penalty for hitting walls
REWARD_REVISIT = -0.2       # Penalty for revisiting cells

# Device Configuration
DEVICE = "cuda" if hasattr(__import__('torch'), 'cuda') and __import__('torch').cuda.is_available() else "cpu"
