"""
Example script demonstrating how to solve a maze using Genetic Algorithm.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Import the required modules
from maze.Maze_builder import Maze
from maze.generators.recursive_backtracker import RecursiveBacktracker
from maze.solvers.genetic.ga_solver import GeneticAlgorithmMazeSolver
from visualization.renderer import render_maze
from config.settings import MAZE_WIDTH, MAZE_HEIGHT, OUTPUT_DIRECTORY
from config.ga_config import (
    POPULATION_SIZE, MAX_GENERATIONS, MUTATION_RATE, 
    CROSSOVER_RATE, TOURNAMENT_SIZE, CROSSOVER_TYPE, MUTATION_TYPE
)


def visualize_solution(maze, solution_path, save_path=None, show=True):
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
            elif maze.is_wall(x, y):
                # Walls are dark blue
                grid[y, x] = [0.1, 0.1, 0.3]
            else:
                # Regular paths are white
                grid[y, x] = [1.0, 1.0, 1.0]
    
    # Display the maze
    plt.imshow(grid, interpolation='nearest')
    plt.title(f"Genetic Algorithm Solution (Path Length: {len(solution_path)})")
    
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
    unsolved_path = os.path.join(OUTPUT_DIRECTORY, "ga_unsolved.png")
    render_maze(maze, title="Unsolved Maze", save_path=unsolved_path, show=False)
    print(f"Unsolved maze saved to {unsolved_path}")
    
    # Define start and end positions
    start_pos = maze.start
    end_pos = maze.end
    
    print(f"Start position: {start_pos}, End position: {end_pos}")
    print("Solving maze with Genetic Algorithm...")
    print(f"Population size: {POPULATION_SIZE}, Max generations: {MAX_GENERATIONS}")
    print(f"Mutation rate: {MUTATION_RATE}, Crossover rate: {CROSSOVER_RATE}")
    
    # Solve using genetic algorithm
    solver = GeneticAlgorithmMazeSolver(
        maze=maze,
        start_pos=start_pos,
        end_pos=end_pos,
        population_size=POPULATION_SIZE,
        max_generations=MAX_GENERATIONS,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        tournament_size=TOURNAMENT_SIZE,
        crossover_type=CROSSOVER_TYPE,
        mutation_type=MUTATION_TYPE
    )
    
    # Time the solving process
    start_time = time.time()
    solution_path = solver.solve()
    solve_time = time.time() - start_time
    
    if solution_path:
        print(f"Solution found!")
        print(f"Path length: {len(solution_path)}")
        print(f"Generations executed: {solver.current_generation}")
        print(f"Solve time: {solve_time:.4f} seconds")
        
        # Visualize the solution
        solved_path = os.path.join(OUTPUT_DIRECTORY, "ga_solved.png")
        visualize_solution(maze, solution_path, save_path=solved_path, show=True)
        print(f"Solved maze saved to {solved_path}")
    else:
        print("No solution found!")


if __name__ == "__main__":
    main()
