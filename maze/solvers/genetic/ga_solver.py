# solvers/genetic/ga_solver.py

from ..solver_base import MazeSolver
from .ga_base import GeneticAlgorithm
from .individual import Individual, Direction
from .population import Population
from .fitness import FitnessEvaluator
from .selection import Selection
from .crossover import Crossover
from .mutation import Mutation

class GeneticAlgorithmMazeSolver(GeneticAlgorithm, MazeSolver):
    """
    Concrete implementation of a Genetic Algorithm for solving mazes.
    Integrates all genetic algorithm components into a complete solver.
    """
    
    def __init__(self, maze, start_pos, end_pos, population_size=100, 
                 max_generations=500, mutation_rate=0.02, crossover_rate=0.8,
                 tournament_size=3, crossover_type="improved_segment", 
                 mutation_type="combined"):
        """
        Initialize the genetic algorithm maze solver.
        
        Args:
            maze: The maze to solve
            start_pos: Starting position in the maze
            end_pos: Goal position in the maze
            population_size: Number of individuals in the population
            max_generations: Maximum number of generations to evolve
            mutation_rate: Probability of gene mutation
            crossover_rate: Probability of crossover
            tournament_size: Number of individuals in tournament selection
            crossover_type: Type of crossover to use
            mutation_type: Type of mutation to use
        """
        # Initialize GeneticAlgorithm parent class
        GeneticAlgorithm.__init__(self, population_size, max_generations, 
                                  mutation_rate, crossover_rate)
        
        # Initialize MazeSolver parent class
        MazeSolver.__init__(self, maze)
        
        # Store maze parameters
        self.maze = maze
        self.start_pos = start_pos
        self.end_pos = end_pos
        
        # Create the supporting components
        self.fitness_evaluator = FitnessEvaluator(maze, start_pos, end_pos)
        self.selection_operator = Selection(tournament_size)
        self.crossover_operator = Crossover(crossover_type)
        self.mutation_operator = Mutation(mutation_rate, mutation_type)
        
        # Tracking best solution and optimal path
        self.best_individual = None
        self.best_path = []
        self.best_fitness = 0
        
        # Population manager
        self.population_manager = Population(population_size, maze)
    
    def initialize_population(self):
        """Initialize the population with random individuals"""
        self.population = self.population_manager.initialize()
    
    def calculate_fitness(self, individual):
        """Calculate fitness for an individual"""
        return self.fitness_evaluator.calculate_fitness(individual)
    
    def select_parents(self):
        """Select parents for reproduction using tournament selection"""
        parent1 = self.selection_operator.tournament_selection(self.population)
        parent2 = self.selection_operator.tournament_selection(self.population)
        
        # Try to ensure different parents if possible
        if len(self.population) > 1:
            attempts = 0
            while parent1 is parent2 and attempts < 3:
                parent2 = self.selection_operator.tournament_selection(self.population)
                attempts += 1
                
        return parent1, parent2
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        return self.crossover_operator.perform_crossover(parent1, parent2)
    
    def mutate(self, individual):
        """Apply mutation to an individual"""
        return self.mutation_operator.mutate(individual, self.maze, self.start_pos)
    
    def termination_condition(self):
        """
        Check if algorithm should terminate early.
        Terminates if an optimal solution is found or if no improvement
        for a significant number of generations.
        """
        # If best individual represents a perfect solution (reached goal with efficient path)
        if self.best_individual and self.best_individual.fitness > 10.0:  
            # 10.0 indicates goal was reached (based on FitnessEvaluator)
            return True
            
        # No improvement for many generations
        if (self.current_generation > 100 and 
            hasattr(self, 'last_improvement_generation') and
            (self.current_generation - self.last_improvement_generation) > 50):
            return True
            
        return False
    
    def solve(self):
        """
        Solve the maze using genetic algorithm.
        
        Returns:
            list: The sequence of positions representing the best path found
        """
        # Run the genetic algorithm
        self.best_individual = self.evolve()
        
        if self.best_individual:
            # Convert the best individual's genome into a path
            self.best_path = self._genome_to_path(self.best_individual.genome)
            
        return self.best_path
    
    def _genome_to_path(self, genome):
        print("DEBUG: _genome_to_path called")
        """
        Convert a genome (sequence of directions) to a path (sequence of positions).
        
        Args:
            genome: List of Direction enum values
            
        Returns:
            list: Sequence of (row, col) positions representing the path
        """
        path = [self.start_pos]
        current_pos = list(self.start_pos)
        visited_order = []  # Track the order of unique visits
        
        for direction in genome:
            # Calculate new position
            new_pos = None
            
            if direction.value == 0:  # UP
                new_pos = [current_pos[0] - 1, current_pos[1]]
            elif direction.value == 1:  # RIGHT
                new_pos = [current_pos[0], current_pos[1] + 1]
            elif direction.value == 2:  # DOWN
                new_pos = [current_pos[0] + 1, current_pos[1]]
            elif direction.value == 3:  # LEFT
                new_pos = [current_pos[0], current_pos[1] - 1]
            
            # Check if move is valid
            if (new_pos[0] >= 0 and new_pos[0] < self.maze.height and 
                new_pos[1] >= 0 and new_pos[1] < self.maze.width and
                not self.maze.is_wall(new_pos[0], new_pos[1])):
                current_pos = new_pos
                path.append(tuple(current_pos))
                
                # Stop if we reached the goal
                if tuple(current_pos) == self.end_pos:
                    break
        
        print(f"DEBUG: Original path length: {len(path)}")
        
        # Return only unique positions in order of first visit
        unique_path = []
        seen = set()
        for pos in path:
            if pos not in seen:
                seen.add(pos)
                unique_path.append(pos)
        
        print(f"DEBUG: Unique path length: {len(unique_path)}")
        return unique_path
