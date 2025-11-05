from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from GA_base import Individual, GA
from typing import List, Tuple, Optional, Any, Dict

# Try importing IOHexperimenter, fallback if not available
try:
    import ioh  # IOHexperimenter
except Exception:
    ioh = None  # Allows importing without IOH available (for testing code structure)

# Compute Hamming distance between two bitstrings
def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


# Genetic Algorithm implementation
class GeneticAlgorithm(GA):
    def __init__(
        self,
        problem: Any,                 # Problem to optimize (IOH PBO problem)
        budget: int = 100_000,        # Max fitness evaluations
        rng: np.random.Generator = np.random.default_rng(0),
        mu: int = 20,                 # Population size
        lam: int = 20,                # #offspring per generation
        pc: float = 0.9,              # Crossover probability
        pm: float = None,             # Mutation probability (default 1/n)
        adaptive_on_stall: bool = True,   # Enable adaptive mutation when stalled
        stall_window: int = 200,          # Window to trigger adaptation
        evaluations_include_init: bool = True,
        record_best_each_eval: bool = True,
    ):
        super().__init__(problem, budget, rng, mu)
        if pm is None:
            pm = 1.0 / self.n  # Default mutation rate = 1/n
        self.lam = lam
        self.pc = pc
        self.pm_base = pm
        self.pm = pm
        self.adaptive_on_stall = adaptive_on_stall
        self.stall_window = stall_window
        self.evaluations_include_init = evaluations_include_init
        self.record_best_each_eval = record_best_each_eval
        # Runtime variables
        self.stall = 0

    # Generate a random individual
    def _random_individual(self) -> Individual:
        bits = self.rng.integers(0, 2, size=self.n, dtype=np.int8).astype(bool)
        fit = self._evaluate(bits)
        return Individual(bits=bits, fitness=fit, age=0)

    # Initialize population
    def _initialize(self) -> List[Individual]:
        P: List[Individual] = [self._random_individual() for _ in range(self.mu)]
        self.best = max(P, key=lambda ind: ind.fitness)
        return P

    # Binary tournament selection (tie-break by distance to global best, then random)
    def _binary_tournament(self, P: List[Individual]) -> Individual:
        a, b = self.rng.integers(0, len(P), size=2)
        ia, ib = P[a], P[b]
        if ia.fitness > ib.fitness:
            return ia
        if ib.fitness > ia.fitness:
            return ib
        # Tie-break using Hamming distance to global best
        assert self.best is not None
        da = hamming_distance(ia.bits, self.best.bits)
        db = hamming_distance(ib.bits, self.best.bits)
        if da > db:
            return ia
        if db > da:
            return ib
        # Still tie → random pick
        return ia if self.rng.random() < 0.5 else ib

    # Select parent pairs
    def _select_parents(self, P: List[Individual], pairs: int) -> List[Tuple[Individual, Individual]]:
        return [(self._binary_tournament(P), self._binary_tournament(P)) for _ in range(pairs)]

    # Uniform crossover
    def _uniform_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.pc:
            return p1.copy() if self.rng.random() < 0.5 else p2.copy()
        mask = self.rng.integers(0, 2, size=self.n, dtype=np.int8).astype(bool)
        child = np.where(mask, p1, p2)
        return child.copy()

    # Bit-flip mutation
    def _bitflip_mutation(self, bits: np.ndarray, pm: float) -> np.ndarray:
        if pm <= 0:
            return bits
        flips = self.rng.random(self.n) < pm
        if flips.any():
            bits = bits.copy()
            bits[flips] = ~bits[flips]
        return bits

    # Check if population already contains the bitstring
    def _diversity_guard_contains(self, P: List[Individual], bits: np.ndarray) -> bool:
        for ind in P:
            if np.array_equal(ind.bits, bits):
                return True
        return False

    # Replacement strategy with elitism & diversity guard
    def _replace(self, P: List[Individual], O: List[Individual]) -> List[Individual]:
        pool = P + O
        # Sort by fitness (desc), tie-break by age (younger better)
        pool_sorted = sorted(pool, key=lambda s: (s.fitness, -s.age), reverse=True)

        elites: List[Individual] = []
        for cand in pool_sorted:
            if len(elites) == 0:
                elites.append(cand)
                continue
            # Skip duplicates (same bits, worse or equal fitness)
            if any(np.array_equal(cand.bits, e.bits) and cand.fitness <= e.fitness for e in elites):
                continue
            if len(elites) < self.mu:
                elites.append(cand)
            if len(elites) >= self.mu:
                break
        # Age individuals
        for e in elites:
            e.age += 1
        return elites

    # Main evolutionary loop
    def run(self) -> Dict[str, Any]:
        P = self._initialize()

        best_so_far = self.best.fitness if self.best else -np.inf
        last_improvement_eval = self.eval_count

        while self.eval_count < self.budget:
            # Mutation rate adaptation if stalled
            pm_gen = self.pm_base
            if self.adaptive_on_stall and (self.eval_count - last_improvement_eval) >= self.stall_window:
                pm_gen = min(2.0 / self.n, 0.5)

            # Generate offspring
            pairs = self.lam
            parent_pairs = self._select_parents(P, pairs=pairs)

            O: List[Individual] = []
            for p1, p2 in parent_pairs:
                if self.eval_count >= self.budget:
                    break
                # Create child
                child_bits = self._uniform_crossover(p1.bits, p2.bits)
                child_bits = self._bitflip_mutation(child_bits, pm_gen)

                # Ensure not duplicate → flip one random bit
                if self._diversity_guard_contains(P, child_bits):
                    j = self.rng.integers(0, self.n)
                    child_bits = child_bits.copy()
                    child_bits[j] = ~child_bits[j]

                # Evaluate child
                child_fit = self._evaluate(child_bits)
                O.append(Individual(bits=child_bits, fitness=child_fit, age=0))

                # Update best if improved
                if child_fit > best_so_far:
                    best_so_far = child_fit
                    self.best = max([self.best] + O + P, key=lambda ind: ind.fitness) if self.best else O[-1]
                    last_improvement_eval = self.eval_count

            # Replace population
            P = self._replace(P, O)
            gen_best = max(P, key=lambda ind: ind.fitness)
            if gen_best.fitness > best_so_far:
                best_so_far = gen_best.fitness
                self.best = gen_best
                last_improvement_eval = self.eval_count

        # Return results
        return {
            "best_individual": self.best,
            "eval_count": self.eval_count,
            "best_eval": last_improvement_eval
        }


# Run GA on benchmark functions with IOHexperimenter
def run_experiment(
    function_ids: List[int],
    dim: int = 100,
    reps: int = 10,
    budget: int = 100_000,
    seed: int = 99,
    mu: int = 20,
    lam: int = 20,
    pc: float = 0.9,
    pm: Optional[float] = None,
    adaptive_on_stall: bool = True,
    stall_window: int = 200,
    experiment_name: str = "A2_GA",
    algorithm_name: str = "UniformGA",
    output_directory: Optional[str] = None,
) -> None:
    """
    Fixed IOH usage:
      - No 'experiment_name' kwarg to Analyzer (use algorithm_info instead).
      - Create ONE Analyzer for the whole run (prevents ex3_data, ex3_data-1, ... proliferation).
      - Detach logger with no arguments; close analyzer once at the end.
    """
    if ioh is None:
        raise RuntimeError("The 'ioh' package is not available in this environment. Install with 'pip install ioh'.")

    base_rng = np.random.default_rng(seed)

    # Create a single analyzer for the whole experiment (one folder)
    analyzer = ioh.logger.Analyzer(
        algorithm_name=algorithm_name,
        algorithm_info=experiment_name,            # store the experiment name as metadata
        folder_name=output_directory or "ex3_data" # single folder instead of auto-incrementing
    )

    try:
        # Iterate problems and repetitions, attaching the same analyzer each time
        for fid in function_ids:
            for rep in range(reps):
                rep_seed = int(base_rng.integers(0, 2**31 - 1))
                rng = np.random.default_rng(rep_seed)

                # Get PBO problem
                problem = ioh.get_problem(fid, instance=1, dimension=dim, problem_class=ioh.ProblemClass.PBO)

                # Attach analyzer (logging results) to this problem
                problem.attach_logger(analyzer)
                try:
                    # Run GA
                    ga = GeneticAlgorithm(
                        problem=problem,
                        n=dim,
                        mu=mu,
                        lam=lam,
                        pc=pc,
                        pm=pm,
                        adaptive_on_stall=adaptive_on_stall,
                        stall_window=stall_window,
                        rng=rng,
                        budget=budget,
                        evaluations_include_init=True,
                    )
                    ga.run()
                finally:
                    # Detach logger (no arguments in current IOH API)
                    problem.detach_logger()
    finally:
        # Ensure analyzer is closed to flush/finish logging
        analyzer.close()


# Command-line interface
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GA (Uniform crossover, Bit-flip mutation) for PBO via IOH.")
    parser.add_argument("--functions", type=int, nargs="+", default=[1,2,3,18,23,24,25], help="PBO function IDs")
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--budget", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--mu", type=int, default=20)
    parser.add_argument("--lam", type=int, default=20)
    parser.add_argument("--pc", type=float, default=0.9)
    parser.add_argument("--pm", type=float, default=None)
    parser.add_argument("--stall_window", type=int, default=200)
    parser.add_argument("--no_adaptive", action="store_true", help="Disable adaptive mutation bump on stall")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory for IOH Analyzer logs (defaults to 'ex3_data')")
    args = parser.parse_args()

    run_experiment(
        function_ids=args.functions,
        dim=args.dim,
        reps=args.reps,
        budget=args.budget,
        seed=args.seed,
        mu=args.mu,
        lam=args.lam,
        pc=args.pc,
        pm=args.pm,
        adaptive_on_stall=(not args.no_adaptive),
        stall_window=args.stall_window,
        output_directory=args.outdir,
    )