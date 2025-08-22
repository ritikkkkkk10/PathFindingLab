

class Maze:
    """
    Maze class for representing and manipulating mazes.
    The maze is represented as a 2D grid where:
    - 0 represents an open path
    - 1 represents a wall
    """

    def __init__(self,width, height):

        """
        Initialize a new maze with the given dimensions.
        By default, the maze is filled with walls.
        
        Args:
            width (int): Width of the maze
            height (int): Height of the maze
        """

        self.width=width
        self.height=height
        self.grid=[[1 for _ in range(self.width)] for _ in range(self.height)]
        self.start=None
        self.end=None

    def is_valid_position(self,x,y):

        """
        Check if the given position is within maze boundaries.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            bool: True if the position is within bounds, False otherwise
        """

        return 0<=x<self.width and 0<=y<self.height
    
    def is_wall(self,x,y):

        """
        Check if the given position is a wall.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            bool: True if the position is a wall, False otherwise
        """

        if not self.is_valid_position(x,y):
            return True
        
        return self.grid[y][x]==1
    
    def is_path(self,x,y):

        """
        Check if the given position is a path (not a wall).
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            bool: True if the position is a path, False otherwise
        """

        return not self.is_wall(x,y)
    
    def set_path(self,x,y):
        """
        Set a path at the given position.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            bool: True if successful, False if position is invalid
        """
        if not self.is_valid_position(x,y):
            return False
        self.grid[y][x]=0
        return True
    
    def set_wall(self,x,y):
        """
        Set a wall at the given position.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            bool: True if successful, False if position is invalid
        """

        if not self.is_valid_position(x,y):
            return False
        
        self.grid[y][x]=1

        return True
    
    def set_start(self,x,y):

        """
        Set the start position of the maze.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            bool: True if successful, False if position is invalid
        """

        if not self.is_valid_position(x,y):
            return False
        
        self.start=(x,y)

        return self.set_path(x,y)
    
    def set_end(self,x,y):

        """
        Set the end position of the maze.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            bool: True if successful, False if position is invalid
        """

        if not self.is_valid_position(x,y):
            return False
        
        self.end=(x,y)

        return self.set_path(x,y)
    
    def get_neighbors(self,x,y):

        """
        Get valid neighboring cells (up, right, down, left).
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            list: List of (x, y) coordinates for valid neighboring cells
        """

        neighbors=[]

        for dx, dy in [(0,-1),(1,0),(0,1),(-1,0)]:

            nx,ny=x+dx,y+dy

            if self.is_valid_position(nx,ny):
                neighbors.append((nx,ny))
            
        return neighbors
    
    def get_path_neighbors(self,x,y):

        """
        Get neighboring cells that are paths (not walls).
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            list: List of (x, y) coordinates for neighboring paths
        """

        return [(nx,ny) for nx, ny in self.get_neighbors(x,y) if self.is_path(nx,ny)]
    
    def fill_with_walls(self):
        """Fill the entire maze with walls."""

        self.grid=[[1 for _ in range(self.width)] for _ in range(self.height)]

    def __str__(self):
        """
        Return a string representation of the maze.
        
        Returns:
            str: String representation where '#' is a wall and ' ' is a path
        """
        result=[]

        for y in range(self.height):
            row=[]

            for x in range(self.width):
                if (x,y)==self.start:
                    row.append('S')
                elif (x,y)==self.end:
                    row.append('E')
                elif self.is_wall(x,y):
                    row.append('#')
                else:
                    row.append(' ')

            result.append(''.join(row))
        
        return '\n'.join(result)
    


