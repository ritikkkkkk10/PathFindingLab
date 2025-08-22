"""Abstract base class for the maze generators."""

from abc import ABC, abstractmethod

class MazeGenerator(ABC):
    """
    Base class for all maze generation algorithms.

    """
    @abstractmethod

    def generate(self,maze):
        """
        Generate a maze.
        
        Args:
            maze: The maze object to generate paths in
            
        Returns:
            The generated maze object
        """
        pass
