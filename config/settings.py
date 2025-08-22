"""
Global settings for the maze project.
This file contains configuration parameters used throughout the project.
"""

# Maze dimensions
MAZE_WIDTH = 21  # Odd numbers work best for most generation algorithms
MAZE_HEIGHT = 21

# Visualization settings
VISUALIZATION_ENABLED = True
CELL_SIZE = 10  # Pixel size of each cell when rendering
WALL_COLOR = (0.1, 0.1, 0.3)  # Dark blue
PATH_COLOR = (1.0, 1.0, 1.0)  # White
START_COLOR = (0.0, 0.8, 0.0)  # Green
END_COLOR = (0.8, 0.0, 0.0)  # Red

# Algorithm settings
RANDOM_SEED = None  # Set to an integer for reproducible maze generation

# Path settings
OUTPUT_DIRECTORY = "output"  # For saving maze visualizations

# Debug settings
DEBUG_MODE = False  # Enable for additional logging
