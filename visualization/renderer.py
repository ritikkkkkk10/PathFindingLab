import numpy as np 
import matplotlib.pyplot as plt 
from config.settings import (
    WALL_COLOR, 
    PATH_COLOR, 
    START_COLOR, 
    END_COLOR, 
    CELL_SIZE
)

def render_maze(maze, show=True, save_path=None, title="Maze"):
    """
    Render a maze using matplotlib.
    
    Args:
        maze (Maze): The maze object to render
        show (bool): Whether to display the maze (default: True)
        save_path (str): Path to save the maze image (default: None)
        title (str): Title for the maze plot (default: "Maze")
        
    Returns:
        The matplotlib figure
    """

    plt.figure(figsize=(CELL_SIZE * maze.width / 10, CELL_SIZE * maze.height / 10))

    # Create a grid for visualization
    grid = np.ones((maze.height, maze.width, 3))

    # Fill the grid with appropriate colors 
    for y in range(maze.height):
        for x in range(maze.width):
            if (x,y) == maze.start:
                # Start position is green 
                grid[y,x] = START_COLOR
            elif (x,y) == maze.end:
                # End position with red 
                grid[y,x] = END_COLOR
            elif maze.is_wall(x,y):
                # Walls are dark blue 
                grid[y,x] = WALL_COLOR
            else:
                # Paths are white 
                grid[y,x] = PATH_COLOR

    plt.imshow(grid, interpolation='nearest')
    plt.title(title)

    plt.xticks([])
    plt.yticks([])

    plt.grid(False)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()

    return plt.gcf()
