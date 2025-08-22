""" This module implements the DFS Algorithm for finding the solution for the maze """

from maze.solvers.solver_base import MazeSolver


class DFSSolver(MazeSolver):
    """
    Depth-First Search maze solver.

    DFS explores as deep as possible along each branch before backtracking.
    It uses recursion (or an explicit stack) and will find a valid path if one exists.
    """

    def solve(self, maze)->bool:
        """
        Solve the maze using Depth-First Search.

        Args:
            maze: The Maze object with start and end defined

        Returns:
            True if a path from start to end was found, False otherwise
        """
        self.clear()
        if maze.start is None or maze.end is None :
            return False

        path_exist=self._dfs_solver(maze,maze.start)

        return path_exist
    
    def _dfs_solver(self,maze, cell):
        """
        Recursive helper for DFS.

        Args:
            maze: The Maze object
            cell: Current (x, y) position

        Returns:
            True if end is reached in this branch, False otherwise
        """
        self.visited.add(cell)
        self.explored_count+=1

        if cell==maze.end:
            self.path.append(cell)
            return True
        
        for neighbors in maze.get_path_neighbors(*cell):
            if neighbors not in self.visited:
                if self._dfs_solver(maze,neighbors):
                    self.path.insert(0,cell)
                    return True
        
        return False


        