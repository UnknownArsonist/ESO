from GA_base import GA
from uniform_ga_mut import GeneticAlgorithm
import numpy as np

def get_algorithm(name):
    if (name == "EA"):
        return EA
    if name == "RLS":
        return RLS
    if name == "GA":
        return GeneticAlgorithm
    return None

class RLS(GA):
    def __init__(self, problem, budget = 100_000, rng: np.random.Generator = np.random.default_rng(0)):
        super().__init__(problem, budget, rng, 1)
    
    def _mutate(self, x_bits):
        bits = x_bits.copy()
        randi = self.rng.integers(self.n)
        bits[randi] = 0 if bits[randi] == 1 else 1
        return bits

class EA(GA):
    def __init__(self, problem, budget = 100_000, rng: np.random.Generator = np.random.default_rng(0)):
        super().__init__(problem, budget, rng, 1)
    
    def _mutate(self, x_bits):
        bits = np.copy(x_bits)
        mutation = self.rng.integers(low=0, high=self.n, size=self.n) < 1
        bits[mutation] = 1 - bits[mutation]
        return bits