from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import ioh

from multi_obj_fitness import MultiObjectiveFitness, ParetoSet, strictly_dominates

@dataclass
class GSEMOConfig:
    budget: int = 10_000
    seed: int = 1
    lookahead: int = 1

class GSEMO:
    """
    Pareto-based GSEMO with bit-flip mutation (p = 1/n) over {0,1}^n.
    Uses the multi-objective (f1,f2) with unit-cost nodes and a (k+lookahead) guard.
    """
    def __init__(self, problem: ioh.Problem, k: int, cfg: Optional[GSEMOConfig] = None):
        self.problem = problem
        self.n = problem.meta_data.n_variables
        self.k = int(k)
        self.cfg = cfg or GSEMOConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.mo_eval = MultiObjectiveFitness(problem=self.problem, k=self.k, lookahead=self.cfg.lookahead)
        self.archive_vals = None  # filled after run()

    def _mutate(self, x: np.ndarray) -> np.ndarray:
        flip = self.rng.random(self.n) < (1.0 / self.n)
        if not flip.any():  # ensure at least one bit flips
            i = self.rng.integers(self.n)
            flip[i] = True
        y = x.copy()
        y[flip] ^= 1
        return y

    def run(self) -> None:
        # Init population with a single random point and the empty set
        x0 = (self.rng.random(self.n) < 0.1).astype(np.int8)  # light init
        pop: List[np.ndarray] = [x0, np.zeros(self.n, dtype=np.int8)]

        archive = ParetoSet()
        evals = 0

        # Evaluate initial population
        for x in pop:
            cost = int(np.count_nonzero(x))
            if cost <= self.k + self.cfg.lookahead:
                f1, f2 = self.mo_eval.evaluate(x)
                evals += 1
                archive.add(x, (f1, f2))
            else:
                # infeasible
                f1, f2 = (-1.0, -float(cost))
                archive.add(x, (f1, f2))

        # Main loop
        while evals < self.cfg.budget:
            sols = archive.solutions()
            parent = sols[self.rng.integers(len(sols))] if sols else np.zeros(self.n, np.int8)

            child = self._mutate(parent)
            cost = int(np.count_nonzero(child))

            # Evaluate child
            if cost <= self.k + self.cfg.lookahead:
                f1, f2 = self.mo_eval.evaluate(child)
                evals += 1
                archive.add(child, (f1, f2))
            else:
                f1, f2 = (-1.0, -float(cost))
                archive.add(child, (f1, f2))

        # keep final Pareto front values available
        self.archive_vals = np.array(archive.values(), dtype=float)
        return
