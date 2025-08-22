"""
Example script demonstrating how to solve a maze using A* Search.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Import the required modules
from maze.Maze_builder import Maze
from maze.generators.recursive_backtracker import RecursiveBacktracker
from maze.solvers.traditional.A_star import AStarSolver
from visualization.renderer import render_maze
from config.settings import MAZE_WIDTH, MAZE_HEIGHT, OUTPUT_DIRECTORY


def visualize_solution(maze, solution_path, visited, save_path=None, show=True):
    """Render a maze with the solution path highlighted."""
    plt.figure(figsize=(10, 10 * maze.height / maze.width))
    
    # Create a grid for visualization
    grid = np.ones((maze.height, maze.width, 3))
    
    # Fill the grid with appropriate colors
    for y in range(maze.height):
        for x in range(maze.width):
            if (x, y) == maze.start:
                # Start position is green
                grid[y, x] = [0.0, 0.8, 0.0]
            elif (x, y) == maze.end:
                # End position is red
                grid[y, x] = [0.8, 0.0, 0.0]
            elif (x, y) in solution_path:
                # Solution path is yellow
                grid[y, x] = [1.0, 1.0, 0.0]
            elif (x, y) in visited:
                # Visited cells are light blue
                grid[y, x] = [0.7, 0.7, 1.0]
            elif maze.is_wall(x, y):
                # Walls are dark blue
                grid[y, x] = [0.1, 0.1, 0.3]
            else:
                # Regular paths are white
                grid[y, x] = [1.0, 1.0, 1.0]
    
    # Display the maze
    plt.imshow(grid, interpolation='nearest')
    plt.title(f"A* Solution (Path Length: {len(solution_path)}, Cells Explored: {len(visited)})")
    
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    # Create a maze
    print("Creating maze...")
    maze = Maze(MAZE_WIDTH, MAZE_HEIGHT)
    
    # Generate the maze using recursive backtracking
    generator = RecursiveBacktracker()
    generator.generate(maze)
    
    # Save the unsolved maze
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    unsolved_path = os.path.join(OUTPUT_DIRECTORY, "astar_unsolved.png")
    render_maze(maze, title="Unsolved Maze", save_path=unsolved_path, show=False)
    print(f"Unsolved maze saved to {unsolved_path}")
    
    # Solve the maze using A*
    print("Solving maze with A*...")
    solver = AStarSolver()
    start_time = time.time()
    solution_found = solver.solve(maze)
    solve_time = time.time() - start_time
    
    if solution_found:
        print(f"Solution found!")
        print(f"Path length: {len(solver.get_path())}")
        print(f"Cells explored: {solver.get_exploration_count()}")
        print(f"Solve time: {solve_time:.4f} seconds")
        
        # Visualize the solution
        solution_path = solver.get_path()
        visited = solver.get_visited()
        
        solved_path = os.path.join(OUTPUT_DIRECTORY, "astar_solved.png")
        visualize_solution(maze, solution_path, visited, save_path=solved_path, show=True)
        print(f"Solved maze saved to {solved_path}")
    else:
        print("No solution found!")


if __name__ == "__main__":
    main()
