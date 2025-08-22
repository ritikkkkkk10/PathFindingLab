# ga_config.py

# Configuration parameters for the genetic algorithm maze solver

POPULATION_SIZE = 100
MAX_GENERATIONS = 500
MUTATION_RATE = 0.02
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 3
CROSSOVER_TYPE = "improved_segment"  # Options: single_point, two_point, uniform, improved_segment
MUTATION_TYPE = "combined"  # Options: random, change_last, out_of_dead_end, combined

# Additional parameters can be added here as needed

# Example usage:
# from ga_config import POPULATION_SIZE, MUTATION_RATE
# print(POPULATION_SIZE, MUTATION_RATE)
