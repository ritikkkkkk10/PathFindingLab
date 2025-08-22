# solvers/genetic/fitness.py

class FitnessEvaluator:
    """
    Evaluates the fitness of maze-solving individuals in a genetic algorithm.
    Fitness is based on how close the individual gets to the goal, whether it
    reaches the goal, and how efficiently it navigates the maze.
    """
    
    def __init__(self, maze, start_pos, end_pos):
        """
        Initialize the fitness evaluator with maze information.
        
        Args:
            maze: The maze object containing wall information and dimensions
            start_pos: Starting position (row, col) tuple
            end_pos: Goal position (row, col) tuple
        """
        self.maze = maze
        self.start_pos = start_pos
        self.end_pos = end_pos
        # Maximum possible Manhattan distance in the maze
        self.max_distance = maze.height + maze.width
        
    def calculate_fitness(self, individual):
        """
        Calculate fitness score for an individual.
        
        A higher score means a better solution.
        
        Args:
            individual: An Individual object with a genome representing moves
            
        Returns:
            float: Fitness score
        """
        # Start from the initial position
        current_pos = self.start_pos.copy() if hasattr(self.start_pos, 'copy') else self.start_pos
        
        # Keep track of all positions visited (for path length calculation)
        path = [current_pos]
        
        # Track if we've reached the goal
        goal_reached = False
        
        # Simulate the individual's moves through the maze
        for move in individual.get_move_sequence():
            # Calculate new position based on direction
            new_pos = self._apply_move(current_pos, move)
            
            # Check if move is valid (not a wall, within maze boundaries)
            if self._is_valid_move(new_pos):
                current_pos = new_pos
                path.append(current_pos)
                
                # Check if we've reached the goal
                if current_pos == self.end_pos:
                    goal_reached = True
                    break
        
        # === FITNESS CALCULATION COMPONENTS ===
        
        # 1. Distance Component: How close did we get to the goal?
        # (Convert to a 0-1 scale where 1 is best)
        final_distance = self._calculate_distance(current_pos, self.end_pos)
        distance_component = 1.0 - (final_distance / self.max_distance)
        
        # 2. Goal Bonus: Did we reach the destination?
        goal_bonus = 10.0 if goal_reached else 0.0
        
        # 3. Efficiency Component: Penalize long paths
        # (Convert to a 0-1 scale where 1 is best)
        max_possible_steps = self.maze.height * self.maze.width
        path_length = len(path)
        
        # Only apply efficiency penalty if goal was reached
        if goal_reached:
            efficiency_component = 1.0 - (path_length / max_possible_steps)
        else:
            efficiency_component = 0.0
            
        # === FINAL FITNESS CALCULATION ===
        
        # Weight the components and combine
        # Distance is important even if goal not reached
        fitness = (
            (5.0 * distance_component) +  # Distance to goal (weighted)
            goal_bonus +                  # Large bonus for reaching goal
            (2.0 * efficiency_component)  # Reward for short paths
        )
        
        return fitness
    
    def _apply_move(self, position, direction):
        """
        Apply a move in a given direction from the current position.
        
        Args:
            position: Current (row, col) position
            direction: Direction enum value (UP, RIGHT, DOWN, LEFT)
            
        Returns:
            tuple: New (row, col) position
        """
        row, col = position
        
        # Direction values correspond to UP=0, RIGHT=1, DOWN=2, LEFT=3
        if direction.value == 0:  # UP
            return (row - 1, col)
        elif direction.value == 1:  # RIGHT
            return (row, col + 1)
        elif direction.value == 2:  # DOWN
            return (row + 1, col)
        elif direction.value == 3:  # LEFT
            return (row, col - 1)
        
        # If an invalid direction is provided, return the original position
        return position
    
    def _is_valid_move(self, position):
        """
        Check if a position is valid (within bounds and not a wall).
        
        Args:
            position: (row, col) tuple to check
            
        Returns:
            bool: True if the position is valid, False otherwise
        """
        row, col = position
        
        # Check if position is within maze boundaries
        if row < 0 or row >= self.maze.height or col < 0 or col >= self.maze.width:
            return False
            
        # Check if position is not a wall
        if self.maze.is_wall(row, col):
            return False
            
        return True
    
    def _calculate_distance(self, pos1, pos2):
        """
        Calculate Manhattan distance between two positions.
        
        Args:
            pos1: First position (row, col)
            pos2: Second position (row, col)
            
        Returns:
            int: Manhattan distance
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
