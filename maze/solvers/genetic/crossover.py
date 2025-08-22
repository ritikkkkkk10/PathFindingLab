# solvers/genetic/crossover.py
import random

class Crossover:
    """
    Implements different crossover operations for genetic algorithms.
    Crossover combines genetic material from two parent solutions
    to create offspring solutions for maze path finding.
    """
    
    def __init__(self, crossover_type="single_point"):
        """
        Initialize the crossover operator.
        
        Args:
            crossover_type: Type of crossover to use 
                            ("single_point", "two_point", "uniform", or "improved_segment")
        """
        self.crossover_type = crossover_type
    
    def perform_crossover(self, parent1, parent2):
        """
        Perform the selected type of crossover between two parents.
        
        Args:
            parent1: First parent Individual
            parent2: Second parent Individual
            
        Returns:
            tuple: Two offspring Individuals
        """
        if self.crossover_type == "single_point":
            return self.single_point_crossover(parent1, parent2)
        elif self.crossover_type == "two_point":
            return self.two_point_crossover(parent1, parent2)
        elif self.crossover_type == "uniform":
            return self.uniform_crossover(parent1, parent2)
        elif self.crossover_type == "improved_segment":
            return self.improved_segment_crossover(parent1, parent2)
        else:
            # Default to single point if type not recognized
            return self.single_point_crossover(parent1, parent2)
    
    def single_point_crossover(self, parent1, parent2):
        """
        Performs single-point crossover by choosing a random point and
        swapping genetic material from two parents at that point.
        
        Args:
            parent1: First parent Individual
            parent2: Second parent Individual
            
        Returns:
            tuple: Two offspring Individuals
        """
        from .individual import Individual
        
        # Create copies of parent genomes
        genome1 = parent1.genome.copy()
        genome2 = parent2.genome.copy()
        
        # Check if genomes are long enough for crossover
        if len(genome1) <= 1 or len(genome2) <= 1:
            return parent1, parent2
            
        # Select crossover point (not at the very beginning or end)
        crossover_point = random.randint(1, min(len(genome1), len(genome2)) - 1)
        
        # Create offspring by swapping segments
        offspring1_genome = genome1[:crossover_point] + genome2[crossover_point:]
        offspring2_genome = genome2[:crossover_point] + genome1[crossover_point:]
        
        # Create new individuals with the crossed genomes
        offspring1 = Individual(genome=offspring1_genome)
        offspring2 = Individual(genome=offspring2_genome)
        
        return offspring1, offspring2
    
    def two_point_crossover(self, parent1, parent2):
        """
        Performs two-point crossover by choosing two random points and
        swapping the segment between these points.
        
        Args:
            parent1: First parent Individual
            parent2: Second parent Individual
            
        Returns:
            tuple: Two offspring Individuals
        """
        from .individual import Individual
        
        # Create copies of parent genomes
        genome1 = parent1.genome.copy()
        genome2 = parent2.genome.copy()
        
        # Check if genomes are long enough for two-point crossover
        min_length = min(len(genome1), len(genome2))
        if min_length <= 2:
            return self.single_point_crossover(parent1, parent2)
            
        # Select two crossover points
        point1 = random.randint(1, min_length - 2)
        point2 = random.randint(point1 + 1, min_length - 1)
        
        # Create offspring by swapping middle segments
        offspring1_genome = genome1[:point1] + genome2[point1:point2] + genome1[point2:]
        offspring2_genome = genome2[:point1] + genome1[point1:point2] + genome2[point2:]
        
        # Create new individuals with the crossed genomes
        offspring1 = Individual(genome=offspring1_genome)
        offspring2 = Individual(genome=offspring2_genome)
        
        return offspring1, offspring2
    
    def uniform_crossover(self, parent1, parent2):
        """
        Performs uniform crossover by randomly selecting genes from
        either parent with equal probability.
        
        Args:
            parent1: First parent Individual
            parent2: Second parent Individual
            
        Returns:
            tuple: Two offspring Individuals
        """
        from .individual import Individual
        
        # Create copies of parent genomes
        genome1 = parent1.genome.copy()
        genome2 = parent2.genome.copy()
        
        # Determine length to use (minimum of the two genomes)
        min_length = min(len(genome1), len(genome2))
        
        # Create offspring genomes
        offspring1_genome = []
        offspring2_genome = []
        
        # For each gene position, randomly select which parent contributes to which child
        for i in range(min_length):
            if random.random() < 0.5:
                offspring1_genome.append(genome1[i])
                offspring2_genome.append(genome2[i])
            else:
                offspring1_genome.append(genome2[i])
                offspring2_genome.append(genome1[i])
        
        # Add any remaining genes from longer parent if applicable
        if len(genome1) > min_length:
            offspring1_genome.extend(genome1[min_length:])
        if len(genome2) > min_length:
            offspring2_genome.extend(genome2[min_length:])
        
        # Create new individuals with the crossed genomes
        offspring1 = Individual(genome=offspring1_genome)
        offspring2 = Individual(genome=offspring2_genome)
        
        return offspring1, offspring2
    
    def improved_segment_crossover(self, parent1, parent2):
        """
        Performs improved segment crossover as recommended for maze solving.
        Divides genomes into three segments with the first segment being the
        best working sequence between parents.
        
        Args:
            parent1: First parent Individual
            parent2: Second parent Individual
            
        Returns:
            tuple: Two offspring Individuals
        """
        from .individual import Individual
        
        # Create copies of parent genomes
        genome1 = parent1.genome.copy()
        genome2 = parent2.genome.copy()
        
        # Determine segment size (approximate 1/3 of genome)
        min_length = min(len(genome1), len(genome2))
        
        if min_length <= 3:
            return self.single_point_crossover(parent1, parent2)
        
        segment_size = min_length // 3
        
        # For maze solving, we assume the beginning of the path (first segment)
        # is more critical as it navigates out of the starting area
        # In a real implementation, this could be more sophisticated based on fitness evaluation
        
        # Create offspring by randomly selecting segments from parents
        if parent1.fitness > parent2.fitness:
            # If parent1 is fitter, use its first segment
            first_segment = genome1[:segment_size]
        else:
            # If parent2 is fitter, use its first segment
            first_segment = genome2[:segment_size]
        
        # Randomly decide how to combine remaining segments
        if random.random() < 0.5:
            offspring1_genome = first_segment + genome1[segment_size:2*segment_size] + genome2[2*segment_size:min_length]
            offspring2_genome = first_segment + genome2[segment_size:2*segment_size] + genome1[2*segment_size:min_length]
        else:
            offspring1_genome = first_segment + genome2[segment_size:2*segment_size] + genome1[2*segment_size:min_length]
            offspring2_genome = first_segment + genome1[segment_size:2*segment_size] + genome2[2*segment_size:min_length]
        
        # Create new individuals with the crossed genomes
        offspring1 = Individual(genome=offspring1_genome)
        offspring2 = Individual(genome=offspring2_genome)
        
        return offspring1, offspring2
