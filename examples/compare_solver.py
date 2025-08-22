"""
Script to compare different maze-solving algorithms:
- Depth First Search (DFS)
- Breadth First Search (BFS)
- A* Search Algorithm
- Genetic Algorithm

Metrics compared:
- Path length (shorter is better)
- Execution time (faster is better)
- Cells explored (fewer is better)
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate  

# Import maze and solvers
from maze.Maze_builder import Maze
from maze.generators.recursive_backtracker import RecursiveBacktracker
from maze.solvers.traditional.dfs import DFSSolver
from maze.solvers.traditional.bfs import BFSSolver
from maze.solvers.traditional.A_star import AStarSolver
from maze.solvers.genetic.ga_solver import GeneticAlgorithmMazeSolver
from visualization.renderer import render_maze
from config.settings import OUTPUT_DIRECTORY
from config.ga_config import (
    POPULATION_SIZE, MAX_GENERATIONS, MUTATION_RATE, 
    CROSSOVER_RATE, TOURNAMENT_SIZE, CROSSOVER_TYPE, MUTATION_TYPE
)

def visualize_comparison(maze, solutions, filename="comparison.png"):
    """Visualize different solution paths on the same maze for comparison."""
    num_solutions = len(solutions)
    fig, axes = plt.subplots(1, num_solutions, figsize=(5*num_solutions, 5))
    
    if num_solutions == 1:
        axes = [axes]  # Make axes iterable if only one subplot
    
    for i, (name, path, cells_explored) in enumerate(solutions):
        # Create grid for visualization
        grid = np.ones((maze.height, maze.width, 3))
        
        # Mark walls, paths, start and end
        for y in range(maze.height):
            for x in range(maze.width):
                if (x, y) == maze.start:
                    grid[y, x] = [0.0, 0.8, 0.0]  # Start: green
                elif (x, y) == maze.end:
                    grid[y, x] = [0.8, 0.0, 0.0]  # End: red
                elif (x, y) in path:
                    grid[y, x] = [1.0, 1.0, 0.0]  # Path: yellow
                elif cells_explored and (x, y) in cells_explored:
                    grid[y, x] = [0.7, 0.7, 1.0]  # Explored: light blue
                elif maze.is_wall(x, y):
                    grid[y, x] = [0.1, 0.1, 0.3]  # Walls: dark blue
                else:
                    grid[y, x] = [1.0, 1.0, 1.0]  # Open: white
        
        # Display the maze solution
        axes[i].imshow(grid, interpolation='nearest')
        axes[i].set_title(f"{name}\nPath: {len(path)} cells\nExplored: {len(cells_explored) if cells_explored else 'N/A'}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIRECTORY, filename)
    plt.savefig(save_path)
    plt.show()
    return save_path

def run_comparison(maze_sizes=[(20, 20), (50, 50)], num_trials=3):
    """Run a comparison of different maze solving algorithms."""
    results = []
    
    for width, height in maze_sizes:
        print(f"\n{'='*50}")
        print(f"Testing maze of size {width}x{height}")
        print(f"{'='*50}")
        
        # Store results for this maze size
        size_results = []
        
        for trial in range(num_trials):
            print(f"\nTrial {trial+1}/{num_trials}")
            
            # Generate a new maze for each trial
            maze = Maze(width, height)
            generator = RecursiveBacktracker()
            generator.generate(maze)
            
            # Define start and end positions
            start_pos = maze.start
            end_pos = maze.end
            
            # Run solvers
            solver_results = []
            solution_paths = []
            
            # 1. DFS
            print("Running DFS solver...")
            dfs_solver = DFSSolver(maze)
            start_time = time.time()
            dfs_found = dfs_solver.solve(maze)
            dfs_time = time.time() - start_time
            
            if dfs_found:
                dfs_path = dfs_solver.get_path()
                dfs_cells = dfs_solver.get_visited() if hasattr(dfs_solver, 'get_visited') else []
                solver_results.append({
                    'algorithm': 'DFS',
                    'path_length': len(dfs_path),
                    'cells_explored': len(dfs_cells),
                    'time': dfs_time
                })
                solution_paths.append(("DFS", dfs_path, dfs_cells))
                print(f"DFS: Path length={len(dfs_path)}, Explored={len(dfs_cells)}, Time={dfs_time:.6f}s")
            else:
                print("DFS: No solution found")
            
            # 2. BFS
            print("Running BFS solver...")
            bfs_solver = BFSSolver(maze)
            start_time = time.time()
            bfs_found = bfs_solver.solve(maze)
            bfs_time = time.time() - start_time
            
            if bfs_found:
                bfs_path = bfs_solver.get_path()
                bfs_cells = bfs_solver.get_visited() if hasattr(bfs_solver, 'get_visited') else []
                solver_results.append({
                    'algorithm': 'BFS',
                    'path_length': len(bfs_path),
                    'cells_explored': len(bfs_cells),
                    'time': bfs_time
                })
                solution_paths.append(("BFS", bfs_path, bfs_cells))
                print(f"BFS: Path length={len(bfs_path)}, Explored={len(bfs_cells)}, Time={bfs_time:.6f}s")
            else:
                print("BFS: No solution found")
            
            # 3. A*
            print("Running A* solver...")
            astar_solver = AStarSolver(maze)
            start_time = time.time()
            astar_found = astar_solver.solve(maze)
            astar_time = time.time() - start_time
            
            if astar_found:
                astar_path = astar_solver.get_path()
                astar_cells = astar_solver.get_visited() if hasattr(astar_solver, 'get_visited') else []
                solver_results.append({
                    'algorithm': 'A*',
                    'path_length': len(astar_path),
                    'cells_explored': len(astar_cells),
                    'time': astar_time
                })
                solution_paths.append(("A*", astar_path, astar_cells))
                print(f"A*: Path length={len(astar_path)}, Explored={len(astar_cells)}, Time={astar_time:.6f}s")
            else:
                print("A*: No solution found")
            
            # 4. Genetic Algorithm
            print("Running Genetic Algorithm solver...")
            ga_solver = GeneticAlgorithmMazeSolver(
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
            
            start_time = time.time()
            ga_path = ga_solver.solve()
            ga_time = time.time() - start_time
            
            if ga_path:
                solver_results.append({
                    'algorithm': 'GA',
                    'path_length': len(ga_path),
                    'cells_explored': ga_solver.current_generation * POPULATION_SIZE,  # Approximation of cells explored
                    'time': ga_time
                })
                solution_paths.append(("GA", ga_path, None))  # GA doesn't track visited cells the same way
                print(f"GA: Path length={len(ga_path)}, Generations={ga_solver.current_generation}, Time={ga_time:.6f}s")
            else:
                print("GA: No solution found")
            
            # Visualize all solutions
            if solution_paths:
                save_path = visualize_comparison(
                    maze, 
                    solution_paths, 
                    f"comparison_{width}x{height}_trial{trial+1}.png"
                )
                print(f"Comparison visualization saved to {save_path}")
            
            size_results.append(solver_results)
        
        # Average results across trials
        algorithms = ['DFS', 'BFS', 'A*', 'GA']
        avg_results = []
        
        for alg in algorithms:
            # Extract results for this algorithm
            alg_results = [
                result for trial_results in size_results 
                for result in trial_results if result['algorithm'] == alg
            ]
            
            if alg_results:
                avg_path_length = sum(r['path_length'] for r in alg_results) / len(alg_results)
                avg_cells = sum(r['cells_explored'] for r in alg_results) / len(alg_results)
                avg_time = sum(r['time'] for r in alg_results) / len(alg_results)
                
                avg_results.append({
                    'maze_size': f"{width}x{height}",
                    'algorithm': alg,
                    'avg_path_length': avg_path_length,
                    'avg_cells_explored': avg_cells,
                    'avg_time': avg_time
                })
        
        results.extend(avg_results)
    
    # Display final results table
    print("\n\nFinal Comparison Results:")
    headers = ['Maze Size', 'Algorithm', 'Avg Path Length', 'Avg Cells Explored', 'Avg Time (s)']
    table_data = [
        [r['maze_size'], r['algorithm'], 
         f"{r['avg_path_length']:.2f}", 
         f"{r['avg_cells_explored']:.2f}", 
         f"{r['avg_time']:.6f}"]
        for r in results
    ]
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Save results to CSV
    csv_path = os.path.join(OUTPUT_DIRECTORY, "algorithm_comparison.csv")
    with open(csv_path, 'w') as f:
        f.write(','.join(headers) + '\n')
        for row in table_data:
            f.write(','.join(str(item) for item in row) + '\n')
    print(f"\nResults saved to {csv_path}")

def main():
    """Main function to run the comparison."""
    # Make sure output directory exists
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # Define maze sizes to test
    # Start with smaller mazes for initial testing
    maze_sizes = [(20, 20), (30, 30)]
    
    # Run comparison with 3 trials per maze size
    run_comparison(maze_sizes, num_trials=3)

if __name__ == "__main__":
    main()
