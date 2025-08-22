# solvers/genetic/selection.py
import random

class Selection:
    """
    Implements selection methods for genetic algorithms,
    focusing on tournament selection for choosing parents for reproduction.
    """
    
    def __init__(self, tournament_size=3):
        """
        Initialize selection with tournament size parameter.
        
        Args:
            tournament_size: Number of individuals competing in each tournament
                             Higher values increase selection pressure toward fitter individuals
        """
        self.tournament_size = tournament_size
    
    def tournament_selection(self, population):
        """
        Performs tournament selection on the population.
        
        Args:
            population: List of Individual objects to select from
            
        Returns:
            Individual: The tournament winner (fittest individual in the tournament)
        """
        # Handle edge case of small population
        if len(population) <= self.tournament_size:
            return max(population, key=lambda ind: ind.fitness)
        
        # Select random individuals for the tournament
        tournament = random.sample(population, self.tournament_size)
        
        # Return the individual with the highest fitness
        return max(tournament, key=lambda ind: ind.fitness)
    
    def select_parents(self, population, num_parents):
        """
        Select multiple parents using tournament selection.
        
        Args:
            population: List of Individual objects to select from
            num_parents: Number of parents to select
            
        Returns:
            list: Selected parent individuals
        """
        parents = []
        for _ in range(num_parents):
            parent = self.tournament_selection(population)
            parents.append(parent)
        return parents
    
    def select_parent_pairs(self, population, num_pairs):
        """
        Select pairs of parents for crossover.
        
        Args:
            population: List of Individual objects to select from
            num_pairs: Number of parent pairs to select
            
        Returns:
            list: List of parent pairs (tuples of two individuals)
        """
        pairs = []
        for _ in range(num_pairs):
            # Select two parents using tournament selection
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            # Ensure parents are different if possible
            if len(population) > 1:
                while parent2 is parent1:  # Check object identity
                    parent2 = self.tournament_selection(population)
            
            pairs.append((parent1, parent2))
        return pairs
