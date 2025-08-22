from maze.generators.generator_base import MazeGenerator
import random

class PrimsGenerator(MazeGenerator):
    """
    Implements Prim's algorithm for maze generation.
    
    This algorithm creates mazes with the following characteristics:
    - Exactly one path between any two points (perfect maze)
    - More organic, branching structure compared to recursive backtracking
    - Tends to create shorter dead ends and more uniform distribution
    - Grows the maze outward from a starting point like a tree
    """

    def generate(self, maze):
        # Start with a grid full of walls
        maze.fill_with_walls()

        # Choose a random starting point  
        start_x = random.randrange(0, maze.width)
        start_y = random.randrange(0, maze.height)

        # Make the starting cell a path
        maze.set_path(start_x, start_y)
        visited = {(start_x, start_y)}

        # Initialize frontier with cells adjacent to starting cell
        frontier = []
        self._add_cells_to_frontier(start_x, start_y, maze, frontier, visited)

        # Main algorithm loop
        while frontier:
            # Pick a random cell from frontier
            cell_idx = random.randrange(len(frontier))
            cell_x, cell_y = frontier.pop(cell_idx)

            # Get adjacent cells that are already part of the maze
            maze_neighbors = self._get_maze_neighbors(cell_x, cell_y, maze, visited)

            if maze_neighbors:
                # Connect to a random maze neighbor
                neighbor_x, neighbor_y = random.choice(maze_neighbors)
                
                # Carve the passage: cell + wall between + neighbor
                maze.set_path(cell_x, cell_y)
                maze.set_path(cell_x + (neighbor_x - cell_x) // 2, 
                            cell_y + (neighbor_y - cell_y) // 2)
                
                visited.add((cell_x, cell_y))
                
                # Add new frontier cells
                self._add_cells_to_frontier(cell_x, cell_y, maze, frontier, visited)

        self._add_entrance_and_exit(maze)
        return maze

    def _add_cells_to_frontier(self, x, y, maze, frontier, visited):
        directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            
            if (maze.is_valid_position(new_x, new_y) and 
                (new_x, new_y) not in visited and 
                (new_x, new_y) not in frontier):
                frontier.append((new_x, new_y))

    def _get_maze_neighbors(self, x, y, maze, visited):
        directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
        maze_neighbors = []
        
        for dx, dy in directions:
            neighbor_x, neighbor_y = x + dx, y + dy
            
            if (maze.is_valid_position(neighbor_x, neighbor_y) and 
                (neighbor_x, neighbor_y) in visited):
                maze_neighbors.append((neighbor_x, neighbor_y))
        
        return maze_neighbors


    def _add_walls_to_frontier(self, x, y, maze, frontier):
        """
        Add walls adjacent to the given cell to the frontier.
        
        This method looks at the four directions from the current cell
        and adds any walls that aren't already in the frontier.
        
        Args:
            x (int): Current X coordinate
            y (int): Current Y coordinate  
            maze: The maze object
            frontier (list): List of frontier walls
        """
        directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]  # up, right, down, left
        
        for dx, dy in directions:
            # Calculate wall position (halfway between current cell and target)
            wall_x = x + dx // 2
            wall_y = y + dy // 2
            
            # Calculate target cell position
            target_x = x + dx
            target_y = y + dy
            
            # Check if target position is valid and is a wall
            if (maze.is_valid_position(target_x, target_y) and 
                maze.is_wall(target_x, target_y) and 
                maze.is_valid_position(wall_x, wall_y) and
                maze.is_wall(wall_x, wall_y) and
                (wall_x, wall_y) not in frontier):
                
                frontier.append((wall_x, wall_y))

    def _get_wall_neighbors(self, wall_x, wall_y, maze):
        """
        Get the cells that are separated by this wall.
        
        A wall can separate cells horizontally or vertically.
        This method determines which cells the wall connects.
        
        Args:
            wall_x (int): Wall X coordinate
            wall_y (int): Wall Y coordinate
            maze: The maze object
            
        Returns:
            list: List of (x, y) tuples representing the neighboring cells
        """
        neighbors = []
        
        # Check horizontal neighbors (wall runs vertically)
        if maze.is_valid_position(wall_x - 1, wall_y):
            neighbors.append((wall_x - 1, wall_y))
        if maze.is_valid_position(wall_x + 1, wall_y):
            neighbors.append((wall_x + 1, wall_y))
            
        # Check vertical neighbors (wall runs horizontally)  
        if maze.is_valid_position(wall_x, wall_y - 1):
            neighbors.append((wall_x, wall_y - 1))
        if maze.is_valid_position(wall_x, wall_y + 1):
            neighbors.append((wall_x, wall_y + 1))
            
        return neighbors

    def _add_entrance_and_exit(self, maze):
        """
        Add entrance and exit points to the maze.
        
        This function finds suitable border cells to use as entrance and exit,
        preferably on opposite sides of the maze for maximum path length.
        Uses the same logic as the recursive backtracker.
        
        Args:
            maze: The maze object
        """
        border_cells = []

        # Check top and bottom rows
        for x in range(maze.width):
            if x > 0 and x < maze.width - 1 and maze.is_path(x, 1):
                border_cells.append((x, 0, "top"))
            if x > 0 and x < maze.width - 1 and maze.is_path(x, maze.height - 2):
                border_cells.append((x, maze.height - 1, "bottom"))

        # Check left and right columns
        for y in range(maze.height):
            if y > 0 and y < maze.height - 1 and maze.is_path(1, y):
                border_cells.append((0, y, "left"))
            if y > 0 and y < maze.height - 1 and maze.is_path(maze.width - 2, y):
                border_cells.append((maze.width - 1, y, "right"))

        # If no border cells are found, create openings manually
        if not border_cells:
            # Create start on left side
            maze.set_path(0, 1)
            maze.set_start(0, 1)
            
            # Create end on the right side
            maze.set_path(maze.width - 1, maze.height - 2)
            maze.set_end(maze.width - 1, maze.height - 2)
            
            return

        random.shuffle(border_cells)

        start_x, start_y, start_side = border_cells[0]
        maze.set_path(start_x, start_y)
        maze.set_start(start_x, start_y)

        # Try to find an exit on the opposite side for maximum distance
        opposite_sides = {
            "top": "bottom", 
            "bottom": "top",
            "left": "right",
            "right": "left"
        }

        # Find the side opposite to our start point
        opposite_side = opposite_sides[start_side]
        opposite_cells = [cell for cell in border_cells if cell[2] == opposite_side]

        if opposite_cells:
            end_x, end_y, _ = random.choice(opposite_cells)
        else:
            other_cells = [cell for cell in border_cells[1:] if cell[2] != start_side]
            
            if other_cells:
                # Choose from available cells on other sides
                end_x, end_y, _ = random.choice(other_cells)
            else:
                # Last resort: use another cell on the same side if available
                if len(border_cells) > 1:
                    end_x, end_y, _ = border_cells[1]
                else:
                    # If all else fails, create an exit manually on the opposite side
                    if start_side == "top":
                        end_x, end_y = maze.width // 2, maze.height - 1
                    elif start_side == "bottom":
                        end_x, end_y = maze.width // 2, 0
                    elif start_side == "left":
                        end_x, end_y = maze.width - 1, maze.height // 2
                    else:  # right
                        end_x, end_y = 0, maze.height // 2
                    
                    # Make sure the exit is a path
                    maze.set_path(end_x, end_y)

        # Set the chosen cell as the end point
        maze.set_end(end_x, end_y)
