import random 
from enum import Enum 

class Direction(Enum):
    UP=0
    RIGHT=1
    DOWN=2
    LEFT=3

class Individual:
    def __init__(self, genome_length=None, genome=None, maze=None):
        ## Adjust genome lenghth based  o n maze size if provided 
        if genome_length is None and maze is not None:
            genome_length=maze.width*maze.height*2
        elif genome_length is None:
            genome_length =100 ## fallback default 

        self.genome=genome if genome else [random.choice(list(Direction))  for _ in range(genome_length)]
        self.fitness=0 ## Will be calculated later 

    def get_move_sequence(self):
        return self.genome
    
    def mutate(self, mutation_rate=0.01):
        for i in range(len(self.genome)):
            if random.random()<mutation_rate:
                self.genome[i]=random.choice(list(Direction))

    def __str__(self):
        return f"Individual: Fitness={self.fitness}, Genome length={len(self.genome)}"
