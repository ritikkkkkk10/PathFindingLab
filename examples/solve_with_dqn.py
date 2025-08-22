"""
Example script demonstrating how to solve a maze using Deep Q-Network (DQN).
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Import the required modules
from maze.Maze_builder import Maze
from maze.generators.recursive_backtracker import RecursiveBacktracker
from maze.solvers.neural.dqn_solver import DQNMazeSolver
from visualization.renderer import render_maze
from config.settings import OUTPUT_DIRECTORY

def visualize_solution(maze, solution_path, visited, training_stats, save_path=None, show=True):
    """Render a maze with the solution path highlighted and training statistics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Maze solution
    grid = np.ones((maze.height, maze.width, 3))
    
    # Fill the grid with appropriate colors
    for y in range(maze.height):
        for x in range(maze.width):
            if (y, x) == maze.start:
                # Start position is green
                grid[y, x] = [0.0, 0.8, 0.0]
            elif (y, x) == maze.end:
                # End position is red
                grid[y, x] = [0.8, 0.0, 0.0]
            elif (y, x) in solution_path:
                # Solution path is yellow
                grid[y, x] = [1.0, 1.0, 0.0]
            elif (y, x) in visited:
                # Visited cells are light blue
                grid[y, x] = [0.7, 0.7, 1.0]
            elif maze.is_wall(y, x):
                # Walls are dark blue
                grid[y, x] = [0.1, 0.1, 0.3]
            else:
                # Regular paths are white
                grid[y, x] = [1.0, 1.0, 1.0]
    
    # Display the maze
    ax1.imshow(grid, interpolation='nearest')
    ax1.set_title(f"DQN Solution\nPath Length: {len(solution_path)}")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.grid(False)
    
    # Right plot: Training progress
    if training_stats and 'episodes_trained' in training_stats and training_stats['episodes_trained'] > 0:
        # This would require training history - simplified version for now
        ax2.text(0.1, 0.8, f"Episodes Trained: {training_stats['episodes_trained']}", 
                transform=ax2.transAxes, fontsize=12)
        ax2.text(0.1, 0.7, f"Success Rate: {training_stats['success_rate']:.2%}", 
                transform=ax2.transAxes, fontsize=12)
        ax2.text(0.1, 0.6, f"Average Reward: {training_stats['average_reward']:.2f}", 
                transform=ax2.transAxes, fontsize=12)
        ax2.text(0.1, 0.5, f"Best Reward: {training_stats['best_reward']:.2f}", 
                transform=ax2.transAxes, fontsize=12)
        ax2.text(0.1, 0.4, f"Final Epsilon: {training_stats['final_epsilon']:.3f}", 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title("Training Statistics")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
    else:
        ax2.text(0.5, 0.5, "No training statistics available", 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title("Training Statistics")
        ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    print("Creating maze...")
    
    # Create a maze - start with smaller size for DQN training
    maze_width, maze_height = 15, 15  # Smaller than your other examples for faster training
    maze = Maze(maze_width, maze_height)
    
    # Generate the maze using recursive backtracking
    generator = RecursiveBacktracker()
    generator.generate(maze)
    
    print(f"Maze dimensions: {maze.width}x{maze.height}")
    print(f"Start position: {maze.start}, End position: {maze.end}")
    
    # Save the unsolved maze
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    unsolved_path = os.path.join(OUTPUT_DIRECTORY, "dqn_unsolved.png")
    render_maze(maze, title="Unsolved Maze", save_path=unsolved_path, show=False)
    print(f"Unsolved maze saved to {unsolved_path}")
    
    # Solve the maze using DQN
    print("Solving maze with Deep Q-Network...")
    print("Note: DQN training may take several minutes depending on maze complexity")
    
    solver = DQNMazeSolver(
        maze=maze,
        start_pos=maze.start,
        end_pos=maze.end
    )
    
    start_time = time.time()
    solution_path = solver.solve(verbose=True)
    solve_time = time.time() - start_time
    
    if solution_path:
        print(f"\nSolution found!")
        print(f"Path length: {len(solution_path)}")
        print(f"Total solve time: {solve_time:.2f} seconds")
        
        # Get training statistics
        training_stats = solver.get_training_stats()
        print(f"Training episodes: {training_stats.get('episodes_trained', 0)}")
        print(f"Success rate: {training_stats.get('success_rate', 0):.2%}")
        
        # Visualize the solution
        visited = solver.get_visited()
        
        solved_path = os.path.join(OUTPUT_DIRECTORY, "dqn_solved.png")
        visualize_solution(maze, solution_path, visited, training_stats, 
                         save_path=solved_path, show=True)
        print(f"Solved maze saved to {solved_path}")
        
        # Print path for verification
        print(f"\nSolution path: {' -> '.join(map(str, solution_path[:5]))}{'...' if len(solution_path) > 5 else ''}")
        
    else:
        print("No solution found!")
        print("Try:")
        print("- Increasing MAX_EPISODES in nn_config.py")
        print("- Using a smaller maze size")
        print("- Adjusting learning rate or other hyperparameters")


def test_small_maze():
    """Test DQN on a very small maze for quick verification."""
    print("\n" + "="*50)
    print("Testing DQN on small maze for quick verification...")
    print("="*50)
    
    # Create a tiny maze
    maze = Maze(7, 7)
    generator = RecursiveBacktracker()
    generator.generate(maze)
    
    print(f"Small maze: {maze.width}x{maze.height}")
    print(f"Start: {maze.start}, End: {maze.end}")
    
    # Quick DQN test
    solver = DQNMazeSolver(maze=maze, start_pos=maze.start, end_pos=maze.end)
    
    start_time = time.time()
    solution = solver.solve(verbose=False)
    test_time = time.time() - start_time
    
    if solution:
        stats = solver.get_training_stats()
        print(f"✓ Quick test successful!")
        print(f"  Path length: {len(solution)}")
        print(f"  Training time: {test_time:.1f}s")
        print(f"  Episodes: {stats.get('episodes_trained', 0)}")
        print(f"  Success rate: {stats.get('success_rate', 0):.1%}")
    else:
        print("✗ Quick test failed - no solution found")
    
    return solution is not None


if __name__ == "__main__":
    # Run quick test first
    if test_small_maze():
        print("\nProceeding to main maze solving...")
        main()
    else:
        print("\nQuick test failed. Check DQN implementation.")
        print("You may need to adjust hyperparameters in nn_config.py")
