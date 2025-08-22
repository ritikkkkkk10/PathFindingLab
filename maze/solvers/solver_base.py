"""
Abstract base class for all maze solving all maze algorithms.

This module defines the xommon interface that all maze solver implementations and 
must follow, ensuring consistency across different algorithms.


"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

class MazeSolver(ABC):
    """
    Abstract base class for maze solving algorithms.
    
    All maze solving algorithms (BFS, DFS, A*, etc.) should inherit from this class
    and implement the required methods.
    """

    def __init__(self,maze):
        """Initialize the solver."""
        self.path=[]
        self.visited=set()
        self.explored_count=0

    @abstractmethod

    def solve(self,maze)->bool:
        """
        Solve the maze and find a path from start to end.
        
        Args:
            maze: The maze object to solve
            
        Returns:
            bool: True if a solution was found, False otherwise
        """
        pass

    def get_path(self)->List[Tuple[int,int]]:
        """
        Get the solution path from start to end.
        
        Returns:
            List[Tuple[int, int]]: List of (x, y) coordinates representing the solution path
        """
        return self.path
    
    def get_visited(self) -> set:
        """
        Get the set of all visited cells during solving.
        
        Returns:
            set: Set of (x, y) coordinates of all visited cells
        """
        return self.visited
    
    
    def get_exploration_count(self)-> int:
         """
        Get the number of cells explored during solving.
        
        Returns:
            int: Number of cells explored
        """
         return self.explored_count
    
    def clear(self)->None:
        """Reset the solver state for a new maze."""
        self.path=[]
        self.visited=set()
        self.explored_count=0
        


