from dataclasses import dataclass
import numpy as np
import copy

# Represents a single solution (individual) in the GA
@dataclass
class Individual:
    bits: np.ndarray           # Bitstring (dtype=bool)
    fitness: float             # Fitness value
    age: int = 0               # Used for tie-breaking (younger preferred)

class GA:
    def __init__(
        self,
        problem,                 # Problem to optimize (IOH PBO problem)
        budget = 100_000,        # Max fitness evaluations
        rng: np.random.Generator = np.random.default_rng(0),
        mu = 1,
        ):
        self.problem = problem
        self.n = problem.meta_data.n_variables
        self.mu = mu
        self.rng = rng
        self.budget = budget

        self.eval_count = 0
        self.best = None

    # Evaluate fitness of a bitstring
    def _evaluate(self, bits):
        val = float(self.problem(bits.astype(np.int32)))
        self.eval_count += 1
        return val
    
    # Generate a random individual
    def _random_individual(self) -> Individual:
        bits = self.rng.integers(low=0, high=2, size=self.n, dtype=np.int32)
        fit = self._evaluate(bits)
        return Individual(bits=bits, fitness=fit, age=0)
    
    # Initialize population
    def _initialize(self):
        P = [self._random_individual() for _ in range(self.mu)]
        self.best = max(P, key=lambda ind: ind.fitness)
        return P
    
    def _mutate(self, x_bits):
        return x_bits.copy()
    
    def run(self):
        x = self._initialize()[0]
        eval_opt = 0
        while self.eval_count < self.budget:
            x_bits = self._mutate(x.bits)
            fit = self._evaluate(x_bits)
            if fit > self.best.fitness:
                x.bits = x_bits
                x.fitness = fit
                self.best = copy.copy(x)
                eval_opt = self.eval_count
            
        return {
            "best_individual": self.best,
            "eval_count": self.eval_count,
            "best_eval": eval_opt
        }