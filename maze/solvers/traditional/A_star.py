## Implementing the A* algorithm for Maze Solver 

import heapq 
from maze.solvers.solver_base import MazeSolver

class AStarSolver(MazeSolver):
    """
    A* Maze Solver — this baby is smarter than plain BFS or Dijkstra.
    It vibes with both cost-so-far (g_score) and future guess (heuristic).
    """

    def solve(self,maze):

        # Clear previous run data 

        self.clear()

        # Sanity check: we need a start and end to run this thing 

        if maze.start is None or maze.end is None :
            return False
        
        # This is our priority queue, where we toss in nodes with their vibes (f_score + g_score)

        start_node=maze.start
        f_start=self._heuristic(start_node,maze.end)
        open_list=[(f_start,0,start_node)]  # (total estimated cost, path cost so far, node)
        heapq.heapify(open_list)

        # Cost from start to this node 

        g_score={start_node: 0}

        # Estimated cost (g+h) to end 

        f_score={start_node: f_start}

        ## Track how we reached each node 

        parents={start_node: None}

        ## Fast lookup to see if a node is in open list 

        open_set={start_node}

        while open_list:
            ## Pulling out the most promising node based on the total cost(f_score)
            current_f,current_g,current=heapq.heappop(open_list)
            open_set.remove(current)

            self.visited.add(current)
            self.explored_count+=1

            # We hit the target 
            if current==maze.end:
                self._reconstruct_path(maze.start,maze.end,parents)
                return True
            for neighbor in maze.get_path_neighbors(*current):
                tentative_g=g_score[current]+1

                ## If this path is shorter than anything we have seen 

                if neighbor not in g_score or tentative_g<g_score[neighbor]:
                    parents[neighbor]=current
                    g_score[neighbor]=tentative_g
                    f_score[neighbor]=tentative_g+self._heuristic(neighbor,maze.end)

                    if neighbor not in open_set:
                        heapq.heappush(open_list,(f_score[neighbor], tentative_g, neighbor))
                        open_set.add(neighbor)

        return False # No path found 
    
    def _heuristic(self,a,b):
       # Manhattan Distance – like how you'd walk in a city grid
 

        return abs(a[0]-b[0])+abs(a[1]-b[1])
    
    def _reconstruct_path(self,start,end,parents):
        current =end
        path=[]

        ## Backtrack from end to start using the parent map 
        while current!=start:
            path.append(current)
            current=parents[current]

        path.append(start)
        path.reverse()
        self.path=path

