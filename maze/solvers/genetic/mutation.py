# solvers/genetic/mutation.py
import random
from .individual import Direction

class Mutation:
    """
    Implements various mutation strategies for genetic algorithms in maze solving.
    Mutation introduces random changes to maintain genetic diversity and explore new solutions.
    """
    
    def __init__(self, mutation_rate=0.01, mutation_type="random"):
        """
        Initialize the mutation operator.
        
        Args:
            mutation_rate: Probability of mutating each gene (0.0-1.0)
            mutation_type: Type of mutation strategy to use
                          ("random", "change_last", "out_of_dead_end", "combined")
        """
        self.mutation_rate = mutation_rate
        self.mutation_type = mutation_type
    
    def mutate(self, individual, maze=None, start_pos=None):
        """
        Apply mutation to an individual based on the selected strategy.
        
        Args:
            individual: The Individual object to mutate
            maze: The maze object (needed for intelligent mutations)
            start_pos: Starting position in the maze
            
        Returns:
            Individual: The mutated individual
        """
        if self.mutation_type == "random":
            return self.random_mutation(individual)
        elif self.mutation_type == "change_last" and maze and start_pos:
            return self.change_last_mutation(individual, maze, start_pos)
        elif self.mutation_type == "out_of_dead_end" and maze and start_pos:
            return self.out_of_dead_end_mutation(individual, maze, start_pos)
        elif self.mutation_type == "combined" and maze and start_pos:
            # Apply multiple mutation strategies
            ind = self.random_mutation(individual)
            ind = self.change_last_mutation(ind, maze, start_pos)
            ind = self.out_of_dead_end_mutation(ind, maze, start_pos)
            return ind
        else:
            # Default to random mutation if type not recognized or maze not provided
            return self.random_mutation(individual)
    
    def random_mutation(self, individual):
        """
        Standard random mutation that changes random directions in the path.
        
        Args:
            individual: The Individual object to mutate
            
        Returns:
            Individual: The mutated individual
        """
        # Create a copy of the genome to avoid modifying the original
        mutated_genome = individual.genome.copy()
        
        # Go through each gene and possibly mutate it
        for i in range(len(mutated_genome)):
            if random.random() < self.mutation_rate:
                # Replace with a random direction different from current one
                current_direction = mutated_genome[i]
                possible_directions = [d for d in Direction if d != current_direction]
                mutated_genome[i] = random.choice(possible_directions)
        
        # Update the individual's genome
        individual.genome = mutated_genome
        return individual
    
    def change_last_mutation(self, individual, maze, start_pos):
        """
        Change Last Operator: Find the first gene that leads to a wall and
        replace it with a valid move.
        
        Args:
            individual: The Individual object to mutate
            maze: The maze object
            start_pos: Starting position in the maze
            
        Returns:
            Individual: The mutated individual
        """
        if random.random() > self.mutation_rate * 5:  # Higher probability for intelligent mutations
            return individual
            
        # Copy genome to avoid modifying original
        mutated_genome = individual.genome.copy()
        
        # Start from initial position
        current_pos = list(start_pos)
        
        # Find the first invalid move (leading to a wall or out of bounds)
        for i, move in enumerate(mutated_genome):
            # Calculate new position based on direction
            new_pos = self._apply_move(current_pos, move)
            
            # Check if move is valid
            if not self._is_valid_move(new_pos, maze):
                # Find a valid direction
                valid_directions = []
                for direction in Direction:
                    test_pos = self._apply_move(current_pos, direction)
                    if self._is_valid_move(test_pos, maze):
                        valid_directions.append(direction)
                
                # If valid directions exist, replace the invalid move
                if valid_directions:
                    mutated_genome[i] = random.choice(valid_directions)
                    # We've fixed one issue, so we'll stop here
                    break
                
            # Update position for next iteration
            current_pos = new_pos if self._is_valid_move(new_pos, maze) else current_pos
        
        # Update the individual's genome
        individual.genome = mutated_genome
        return individual
    
    def out_of_dead_end_mutation(self, individual, maze, start_pos):
        """
        Out of Dead End Operator: Detect if the path leads to a dead end
        and modify it to find an alternative path.
        
        Args:
            individual: The Individual object to mutate
            maze: The maze object
            start_pos: Starting position in the maze
            
        Returns:
            Individual: The mutated individual
        """
        if random.random() > self.mutation_rate * 5:  # Higher probability for intelligent mutations
            return individual
            
        # Copy genome to avoid modifying original
        mutated_genome = individual.genome.copy()
        
        # Start from initial position
        current_pos = list(start_pos)
        path_positions = [tuple(current_pos)]
        
        # Find a dead end by following the path
        dead_end_index = -1
        
        for i, move in enumerate(mutated_genome):
            # Calculate new position
            new_pos = self._apply_move(current_pos, move)
            
            # Check if move is valid
            if self._is_valid_move(new_pos, maze):
                current_pos = new_pos
                path_positions.append(tuple(current_pos))
                
                # Check if we're in a dead end (only one valid move - the way back)
                valid_exits = 0
                for direction in Direction:
                    test_pos = self._apply_move(current_pos, direction)
                    if (self._is_valid_move(test_pos, maze) and 
                        tuple(test_pos) not in path_positions):
                        valid_exits += 1
                
                if valid_exits == 0:
                    dead_end_index = i
                    break
            else:
                # Invalid move, consider it a dead end
                dead_end_index = i
                break
        
        # If we found a dead end, try to fix it
        if dead_end_index >= 0 and dead_end_index < len(mutated_genome) - 1:
            # Go back a few steps and try a different direction
            backtrack_steps = min(3, dead_end_index)
            
            # Replace the moves leading to the dead end with new random directions
            for j in range(dead_end_index - backtrack_steps + 1, dead_end_index + 1):
                mutated_genome[j] = random.choice(list(Direction))
        
        # Update the individual's genome
        individual.genome = mutated_genome
        return individual
    
    def _apply_move(self, position, direction):
        """Helper method to calculate new position after a move"""
        row, col = position
        
        if direction.value == 0:  # UP
            return [row - 1, col]
        elif direction.value == 1:  # RIGHT
            return [row, col + 1]
        elif direction.value == 2:  # DOWN
            return [row + 1, col]
        elif direction.value == 3:  # LEFT
            return [row, col - 1]
        
        return position
    
    def _is_valid_move(self, position, maze):
        """Helper method to check if a move is valid (within bounds and not a wall)"""
        row, col = position
        
        # Check if position is within maze boundaries
        if row < 0 or row >= maze.height or col < 0 or col >= maze.width:
            return False
            
        # Check if position is not a wall
        return not maze.is_wall(row, col)
