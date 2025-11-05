from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Iterable, List
import numpy as np

MOValue = Tuple[float, float]

@dataclass
class MultiObjectiveFitness:
    """
    Multi-objective fitness wrapper for monotone submodular problems.
    Using the bi-objective/Pareto formulation found in lecture.
    """
    problem: Callable[[np.ndarray], float]
    k: int
    lookahead: int = 1

    def evaluate(self, x: np.ndarray) -> MOValue:
        # Ensure binary vector
        if x.dtype != np.int8 and x.dtype != np.int32 and x.dtype != np.int64:
            x = x.astype(np.int8, copy=False)
        # Cost is number of selected nodes (unit cost per node)
        cost = int(np.count_nonzero(x))
        # Quality is the problem’s submodular value, but only inside the relaxed bound
        if cost <= self.k + self.lookahead:
            f1 = float(self.problem(x))  # f(x) from IOH problem
        else:
            f1 = -1.0 
        f2 = -float(cost)  # minimise cost
        return (f1, f2)


def weakly_dominates(a: MOValue, b: MOValue) -> bool:
    """Return True iff a weakly dominates b under (maximize f1, maximize f2)."""
    return (a[0] >= b[0]) and (a[1] >= b[1])


def strictly_dominates(a: MOValue, b: MOValue) -> bool:
    """Return True iff a strictly dominates b."""
    return weakly_dominates(a, b) and (a[0] > b[0] or a[1] > b[1])


class ParetoSet:
    """
    Simple Pareto archive for (f1, f2) pairs. Keeps only non-dominated entries.
    """
    def __init__(self):
        self._vals: List[MOValue] = []
        self._xs: List[np.ndarray] = []

    def add(self, x: np.ndarray, val: MOValue) -> bool:
        # If dominated by an existing point, discard
        for v in self._vals:
            if strictly_dominates(v, val) or v == val:
                return False
        # Remove points dominated by the new one
        keep_idx = []
        for i, v in enumerate(self._vals):
            if not strictly_dominates(val, v):
                keep_idx.append(i)
        if len(keep_idx) != len(self._vals):
            self._vals = [self._vals[i] for i in keep_idx]
            self._xs   = [self._xs[i]   for i in keep_idx]
        # Insert
        self._vals.append(val)
        self._xs.append(x.copy())
        return True

    def values(self) -> List[MOValue]:
        return list(self._vals)

    def solutions(self) -> List[np.ndarray]:
        return [s.copy() for s in self._xs]

    def best_feasible_by_f1(self, k: int) -> Tuple[np.ndarray, MOValue] | None:
        """
        Return the Pareto member with |x|<=k that has max f1.
        (Use for a single ‘recommendation’ at the end.)
        """
        best = None
        for x, v in zip(self._xs, self._vals):
            if int(np.count_nonzero(x)) <= k and v[0] >= 0:
                if best is None or v[0] > best[1][0]:
                    best = (x, v)
        return best
