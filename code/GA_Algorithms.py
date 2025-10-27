from GA_base import GA
from uniform_ga_mut import GeneticAlgorithm
import numpy as np
from gsemo import GSEMO, GSEMOConfig

def get_algorithm(name):
    if name == "EA":
        return EA
    if name == "RLS":
        return RLS
    if name == "GA":
        return GeneticAlgorithm
    if name == "GSEMO":
        def make(problem, budget, rng, k=10):
            cfg = GSEMOConfig(budget=budget, seed=int(rng.integers(1_000_000)))
            return _GSEMOAdapter(problem, k=k, cfg=cfg, rng=rng)
        return make
    return None


class RLS(GA):
    def __init__(self, problem, budget=100_000, rng: np.random.Generator=np.random.default_rng(0)):
        super().__init__(problem, budget, rng, 1)

    def _mutate(self, x_bits):
        bits = x_bits.copy()
        randi = self.rng.integers(self.n)
        bits[randi] = 0 if bits[randi] == 1 else 1
        return bits


class EA(GA):
    def __init__(self, problem, budget=100_000, rng: np.random.Generator=np.random.default_rng(0)):
        super().__init__(problem, budget, rng, 1)

    def _mutate(self, x_bits):
        bits = np.copy(x_bits)
        # flip each bit with prob 1/n (compact form)
        flips = self.rng.random(self.n) < (1.0 / self.n)
        bits[flips] = 1 - bits[flips]
        return bits


class _GSEMOAdapter:
    def __init__(self, problem, k, cfg: GSEMOConfig, rng: np.random.Generator):
        self.inner = GSEMO(problem=problem, k=k, cfg=cfg)
        self.rng = rng

    def run(self):
        self.inner.run()
        return None
