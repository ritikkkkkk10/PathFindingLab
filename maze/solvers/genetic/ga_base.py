from abc import ABC, abstractmethod
import random

class GeneticAlgorithm(ABC):  # Fixed spelling
    def __init__(self, population_size=50, max_generations=500, mutation_rate=0.01, crossover_rate=0.7):
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.current_generation = 0
        self.population = []
        self.best_individual = None

    @abstractmethod
    def initialize_population(self):
        """Initialize the population with random individuals"""  # Fixed typo
        pass 

    @abstractmethod
    def calculate_fitness(self, individual):
        """Calculate fitness for an individual"""
        pass 

    @abstractmethod
    def select_parents(self):
        """Select parents for reproduction"""
        pass 

    @abstractmethod
    def crossover(self, parent1, parent2):
        """Perform the crossover between two parents"""
        pass 
    
    @abstractmethod  # Added missing abstract method
    def mutate(self, individual):
        """Apply mutation to an individual"""
        pass
    
    def evolve(self):
        """Run genetic algorithm"""
        self.initialize_population()
        while self.current_generation < self.max_generations:
            # Evaluate fitness 
            for individual in self.population:
                individual.fitness = self.calculate_fitness(individual)
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            if not self.best_individual or self.population[0].fitness > self.best_individual.fitness:
                self.best_individual = self.population[0]

            if self.termination_condition():
                break 

            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()
                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = self.crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2
                    
                self.mutate(offspring1)
                self.mutate(offspring2)
                
                new_population.append(offspring1)
                new_population.append(offspring2)
                
            self.population = new_population[:self.population_size]
            self.current_generation += 1
            
        return self.best_individual
    
    @abstractmethod
    def termination_condition(self):
        """Check if algorithm should terminate."""
        pass
