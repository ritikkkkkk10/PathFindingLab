
from maze.generators.generator_base import MazeGenerator
import random

class RecursiveBacktracker(MazeGenerator):
    """
    Implements the Recursive Backtracking maze generation algorithm.
    
    This algorithm creates mazes with the following characteristics:
    - Exactly one path between any two points (perfect maze)
    - Long, winding corridors with few branches
    - Tends to create mazes with a biased texture
    """

    def generate(self, maze):
        """
        Generate a maze using recursive backtracking.
        
        The algorithm works as follows:
        1. Start with a grid full of walls
        2. Pick a random cell, make it a path, and mark it as visited
        3. While there are unvisited cells:
           a. If the current cell has unvisited neighbors, choose one randomly,
              knock down the wall between them, and move to that cell
           b. Otherwise, backtrack to the last cell that has unvisited neighbors
        
        Args:
            maze: The maze object to generate paths in
            
        Returns:
            The generated maze object with paths carved through it
        """

        # Start with a grid full of walls 
        maze.fill_with_walls()

        ## Choosing a random startin point 

        start_x=random.randrange(0,maze.width)
        start_y=random.randrange(0,maze.height)

        # Make the starting cell a path 
        maze.set_path(start_x,start_y)

        ## Begin the recursive carving process 
        self._carve_passages(start_x,start_y,maze)

        ## Set entrance and exit point on opposite borders 
        self._add_entrance_and_exit(maze)

        return maze
    
    def _carve_passages(self,x,y,maze):


        """
        Recursively carve passages through the maze.
        
        This is the core of the algorithm. For each cell:
        1. Get all possible directions to move
        2. Shuffle them for randomness
        3. For each direction, check if we can carve a passage
        4. If yes, carve the passage and recursively continue from the new cell
        5. If no valid moves, the function returns (backtracking)
        
        Args:
            x (int): Current X coordinate
            y (int): Current Y coordinate
            maze: The maze object
        """

        directions=[(0,-2),(2,0),(0,2),(-2,0)]

        random.shuffle(directions)

        for dx, dy in directions:

            new_x=x+dx
            new_y=y+dy

            if (maze.is_valid_position(x,y)) and maze.is_wall(new_x,new_y):

                # Carve a passage by making both the wall and the new cell 

                # First make a wall between both the wall and the new cell a path 

                maze.set_path(x+dx//2,y+dy//2)

                maze.set_path(new_x,new_y)

                ## Recursion Step 

                self._carve_passages(new_x,new_y,maze)

    def _add_entrance_and_exit(self,maze):
        """
        Add entrance and exit points to the maze.
        
        This function finds suitable border cells to use as entrance and exit,
        preferably on opposite sides of the maze for maximum path length.
        
        Args:
            maze: The maze object
        """
        border_cells=[]

        # Check top and bottom rows 

        for x in range(maze.width):

            if x>0 and x<maze.width-1 and maze.is_path(x,1):
                border_cells.append((x,0,"top"))
            if x>0 and x<maze.width-1 and maze.is_path(x,maze.height-2):
                border_cells.append((x,maze.height-1,"bottom"))

        # Check left and right columns 

        for y in range(maze.height):
             
            if y>0 and y<maze.height-1 and maze.is_path(1,y):
                border_cells.append((0,y,"left"))

            if y>0 and y<maze.height-1 and maze.is_path(maze.width-2,y):
                border_cells.append((maze.width-1,y,"right"))

        # If no border cells are found create openings manually 

        if not border_cells:

            ## Create start on left side 

            maze.set_path(0,1)
            maze.set_start(0,1)
 
            ## Create end on the right side 

            maze.set_path(maze.width-1,maze.height-2)
            maze.set_end(maze.width-1,maze.height-2)

            return
        
        random.shuffle(border_cells)

        start_x,start_y,start_side=border_cells[0]
        maze.set_path(start_x,start_y)
        maze.set_start(start_x,start_y)

        ## Try to find an exit on the opposite side for maximum distance 

        opposite_sides={
            "top":"bottom",
            "bottom":"top",
            "left":"right",
            "right":"left"

        }

        #Find the side opposite to our start point 

        opposite_side=opposite_sides[start_side]
        opposite_cells=[cell for cell in border_cells if cell[2] == opposite_side]

        if opposite_cells:

            end_x, end_y, _= random.choice(opposite_cells)

        else :

            other_cells=[cell for cell in border_cells[1:] if cell[2]!=start_side]

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
                        end_x, end_y = maze.width//2, maze.height-1
                    elif start_side == "bottom":
                        end_x, end_y = maze.width//2, 0
                    elif start_side == "left":
                        end_x, end_y = maze.width-1, maze.height//2
                    else:  # right
                        end_x, end_y = 0, maze.height//2
                    
                    # Make sure the exit is a path
                    maze.set_path(end_x, end_y)
        
        # Set the chosen cell as the end point
        maze.set_end(end_x, end_y)

                   

           

        