"""Breadth-First Search (BFS) for maze solving.

This module implements the BFS allgorithm for finding the shortest path 
through a maze from start to end.
"""

from collections import deque 
from maze.solvers.solver_base import MazeSolver

class BFSSolver(MazeSolver):
    """
    Breadth-First search maze solver.

    BFS explores the maze in waves of increasing distance from the start point, 
    guaranteeing the shortest possible path in unweighted maze.
    
    """
    def solve(self,maze):
        """
        Solve the maze using Breadth-First Search.
        
        Args:
            maze: The maze object to solve
            
        Returns:
            bool: True if a solution was found, False otherwise
        """
        # Reset the solver state 
        self.clear()

        ## Make sure maze has start and end points

        if maze.start is None or maze.end is None:
            return False
        
        queue=deque([maze.start])
        self.visited.add(maze.start)
        parents={maze.start: None}

        ## BFS algorithm
        while queue:
            current=queue.popleft()
            self.explored_count+=1
            if current==maze.end:
                self._reconstruct_path(maze.start,maze.end,parents)
                return True
            
            for neigbour in maze.get_path_neighbors(*current):
                if neigbour not in self.visited:
                    self.visited.add(neigbour)
                    queue.append(neigbour)
                    parents[neigbour]=current

        return False
    
    def _reconstruct_path(self,start,end,parents):
        """
        Reconstruct the path from start to end using the parent references.
        
        Args:
            start: The start position
            end: The end position
            parents: Dictionary mapping each cell to its parent cell
        """

        current=end
        path=[]

        while current!=start:
            path.append(current)
            current=parents[current]

        path.append(start)

        path.reverse()
        self.path=path

