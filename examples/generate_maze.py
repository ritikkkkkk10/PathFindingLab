import os 
from maze.Maze_builder import Maze 
from maze.generators.recursive_backtracker import RecursiveBacktracker
from maze.generators.prims import PrimsGenerator
from visualization.renderer import render_maze
from config.settings import MAZE_WIDTH, MAZE_HEIGHT, OUTPUT_DIRECTORY

def main():
    # Use the configured output directory instead of hardcoded "output"
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # Define all generators you want to compare
    generators = {
        "Recursive Backtracker": RecursiveBacktracker(),
        "Prim's Algorithm": PrimsGenerator(),
        # Add more generators here as you create them
        # "Kruskal's Algorithm": KruskalsGenerator(),
        # "Wilson's Algorithm": WilsonsGenerator(),
    }
    
    # Generate and visualize each maze type
    mazes = {}
    for name, generator in generators.items():
        print(f"\nGenerating maze using {name}...")
        
        # Create a fresh maze for each generator
        maze = Maze(MAZE_WIDTH, MAZE_HEIGHT)
        
        # Generate the maze
        generator.generate(maze)
        
        # Store the maze for comparison
        mazes[name] = maze
        
        # Create filename from generator name
        filename = name.lower().replace("'", "").replace(" ", "_") + "_maze.png"
        save_path = os.path.join(OUTPUT_DIRECTORY, filename)
        
        # Display text representation
        print(f"\n{name} Maze (text representation):")
        print(maze)
        
        # Visualize using Matplotlib 
        render_maze(maze, title=f"{name} Maze", save_path=save_path)
        print(f"Saved: {save_path}")
    
    # Generate a comparison visualization
    create_comparison_visualization(mazes)

def create_comparison_visualization(mazes):
    """Create a side-by-side comparison of all generated mazes."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        num_mazes = len(mazes)
        if num_mazes == 0:
            return
            
        # Create subplots for comparison
        fig, axes = plt.subplots(1, num_mazes, figsize=(6 * num_mazes, 6))
        
        # Handle single maze case
        if num_mazes == 1:
            axes = [axes]
        
        # Plot each maze
        for idx, (name, maze) in enumerate(mazes.items()):
            ax = axes[idx]
            
            # Create maze array manually without relying on to_array()
            maze_array = np.ones((maze.height, maze.width), dtype=int)
            
            for y in range(maze.height):
                for x in range(maze.width):
                    if maze.is_path(x, y):
                        maze_array[y][x] = 0  # Path = 0 (white)
                    # Wall stays 1 (black)
            
            # Display the maze
            ax.imshow(maze_array, cmap='binary', interpolation='nearest')
            ax.set_title(f"{name}", fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # Save comparison image
        comparison_path = os.path.join(OUTPUT_DIRECTORY, "maze_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comparison visualization saved: {comparison_path}")
        
    except ImportError:
        print("Matplotlib or numpy not available for comparison visualization")
    except Exception as e:
        print(f"Could not create comparison visualization: {e}")


def compare_specific_generators(generator_names):
    """
    Compare only specific generators by name.
    
    Args:
        generator_names (list): List of generator names to compare
    """
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    all_generators = {
        "Recursive Backtracker": RecursiveBacktracker(),
        "Prim's Algorithm": PrimsGenerator(),
    }
    
    # Filter to only requested generators
    selected_generators = {
        name: generator for name, generator in all_generators.items() 
        if name in generator_names
    }
    
    if not selected_generators:
        print("No valid generators found for comparison")
        return
    
    print(f"Comparing: {', '.join(selected_generators.keys())}")
    
    mazes = {}
    for name, generator in selected_generators.items():
        print(f"\nGenerating maze using {name}...")
        
        maze = Maze(MAZE_WIDTH, MAZE_HEIGHT)
        generator.generate(maze)
        mazes[name] = maze
        
        filename = name.lower().replace("'", "").replace(" ", "_") + "_maze.png"
        save_path = os.path.join(OUTPUT_DIRECTORY, filename)
        
        render_maze(maze, title=f"{name} Maze", save_path=save_path)
        print(f"Saved: {save_path}")
    
    create_comparison_visualization(mazes)

if __name__ == "__main__":
    # Option 1: Compare all available generators
    main()
    
    # Option 2: Compare specific generators only
    # compare_specific_generators(["Recursive Backtracker", "Prim's Algorithm"])
