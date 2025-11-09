import numpy as np
import random

class GeneticOptimizer:
    def __init__(self):
        self.config = Config()
        
        self.population_size = self.config.get('optimization.population_size')
        self.generations = self.config.get('optimization.generations')
        self.mutation_rate = self.config.get('optimization.mutation_rate')
        self.crossover_rate = self.config.get('optimization.crossover_rate')
        
        self.objectives = self.config.get('optimization.objectives')
    
    def optimize(self, surrogate_model, parameter_bounds, alpha=5.0):
        population = self._initialize_population(parameter_bounds)
        
        best_individual = None
        best_fitness = float('inf')
        
        for generation in range(self.generations):
            fitness_scores = self._evaluate_population(population, surrogate_model, alpha)
            
            new_population = []
            
            elite_indices = np.argsort(fitness_scores)[:self.population_size//10]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                child1 = self._mutate(child1, parameter_bounds)
                child2 = self._mutate(child2, parameter_bounds)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            
            current_best_idx = np.argmin(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx]
            
            if generation % 10 == 0:
                print(f"mwasifanwar Generation {generation}, Best Fitness: {best_fitness:.6f}")
        
        return best_individual, best_fitness
    
    def _initialize_population(self, bounds):
        population = []
        for _ in range(self.population_size):
            individual = []
            for lower, upper in bounds:
                individual.append(random.uniform(lower, upper))
            population.append(np.array(individual))
        return population
    
    def _evaluate_population(self, population, surrogate_model, alpha):
        fitness_scores = []
        for individual in population:
            coefficients = surrogate_model.predict(individual, alpha)
            fitness = self._fitness_function(coefficients)
            fitness_scores.append(fitness)
        return np.array(fitness_scores)
    
    def _fitness_function(self, coefficients):
        cd = coefficients['cd']
        cl = coefficients['cl']
        
        if 'drag' in self.objectives and 'lift' in self.objectives:
            return cd / max(cl, 0.1)
        elif 'drag' in self.objectives:
            return cd
        else:
            return -cl
    
    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        selected = random.sample(range(len(population)), tournament_size)
        best_idx = selected[np.argmin(fitness_scores[selected])]
        return population[best_idx]
    
    def _crossover(self, parent1, parent2):
        alpha = random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2
    
    def _mutate(self, individual, bounds, mutation_strength=0.1):
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                lower, upper = bounds[i]
                mutation = random.gauss(0, mutation_strength * (upper - lower))
                mutated[i] = np.clip(mutated[i] + mutation, lower, upper)
        return mutated