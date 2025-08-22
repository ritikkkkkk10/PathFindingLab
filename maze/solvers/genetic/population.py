# solvers/genetic/population.py

from .individual import Individual, Direction
import random

class Population:
    """
    Manages a population of individuals (potential maze solutions) for a genetic algorithm.
    Handles population initialization, diversity management, and generational transitions.
    """
    
    def __init__(self, size, maze=None, individual_genome_length=None):
        """
        Initialize a population manager.
        
        Args:
            size: Number of individuals in the population
            maze: Maze object (used to determine appropriate genome length)
            individual_genome_length: Optional explicit genome length
        """
        self.size = size
        self.maze = maze
        self.individuals = []
        self.generation = 0
        self.genome_length = individual_genome_length
        
        # If no explicit genome length is provided but maze is available,
        # calculate a reasonable genome length based on maze dimensions
        if self.genome_length is None and self.maze is not None:
            self.genome_length = maze.width * maze.height * 2
        elif self.genome_length is None:
            # Default fallback if no maze dimensions are available
            self.genome_length = 100
    
    def initialize(self):
        """
        Create an initial random population of individuals.
        Each individual represents a potential solution (path through the maze).
        """
        # Clear any existing population
        self.individuals = []
        
        # Create the specified number of individuals with random moves
        for _ in range(self.size):
            individual = Individual(genome_length=self.genome_length, maze=self.maze)
            self.individuals.append(individual)
            
        self.generation = 0
        return self.individuals
    
    def get_population(self):
        """Return the current population of individuals."""
        return self.individuals
    
    def set_population(self, new_population):
        """
        Replace the current population with a new one.
        Typically called after creating a new generation through selection,
        crossover, and mutation.
        
        Args:
            new_population: List of Individual objects to become the new population
        """
        # Ensure the population size stays constant
        if len(new_population) > self.size:
            self.individuals = new_population[:self.size]  # Truncate if too large
        elif len(new_population) < self.size:
            # If too small, fill the rest with random individuals
            shortfall = self.size - len(new_population)
            for _ in range(shortfall):
                individual = Individual(genome_length=self.genome_length, maze=self.maze)
                new_population.append(individual)
                
        self.individuals = new_population
        self.generation += 1  # Increment generation counter
        
    def get_best_individual(self):
        """
        Return the individual with the highest fitness in the population.
        
        Returns:
            Individual: The best individual or None if population is empty
        """
        if not self.individuals:
            return None
            
        return max(self.individuals, key=lambda ind: ind.fitness)
    
    def calculate_diversity(self):
        """
        Calculate the genetic diversity of the population.
        Higher diversity means the population has more varied solutions.
        
        Returns:
            float: A measure of population diversity (0-1 scale)
        """
        if not self.individuals or len(self.individuals) < 2:
            return 0.0
            
        # Sample a subset of individuals for efficiency in large populations
        sample_size = min(20, len(self.individuals))
        sample = random.sample(self.individuals, sample_size)
        
        # Calculate average genome difference between individuals
        total_diff = 0
        comparisons = 0
        
        for i in range(sample_size):
            for j in range(i+1, sample_size):
                ind1 = sample[i]
                ind2 = sample[j]
                
                # Count differences in their genomes
                differences = sum(1 for g1, g2 in zip(ind1.genome, ind2.genome) if g1 != g2)
                max_possible_diff = min(len(ind1.genome), len(ind2.genome))
                
                # Normalize to 0-1 range
                if max_possible_diff > 0:
                    total_diff += differences / max_possible_diff
                    comparisons += 1
        
        # Average diversity across all comparisons
        return total_diff / comparisons if comparisons > 0 else 0.0
    
    def inject_diversity(self, percentage=0.1):
        """
        Inject diversity by replacing a percentage of the population
        with new random individuals.
        
        Args:
            percentage: Portion of population to replace (0-1)
        """
        if not self.individuals:
            return
            
        # Calculate how many individuals to replace
        num_to_replace = max(1, int(self.size * percentage))
        
        # Sort by fitness and keep the best ones
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        
        # Replace the worst-performing individuals with new random ones
        for i in range(self.size - num_to_replace, self.size):
            self.individuals[i] = Individual(genome_length=self.genome_length, maze=self.maze)
