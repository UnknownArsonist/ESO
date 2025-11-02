#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 4 runner (uses your existing implementations via GA_Algorithms.get_algorithm):
- Algorithms: RLS, EA, GA, GSEMO, SOEA, MOEA
- Problems: MaxCoverage2100..2103, MaxInfluence2200..2203
- Repetitions: 30, Budget: 100,000
- Logs: IOHprofiler Analyzer format (use IOHanalyzer for fixed-budget plots)
- Trade-off plots: saves GSEMO trade-off (cost vs objective) for the FIRST run on each instance

Requires:
  pip install IOHexperimenter matplotlib
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import ioh

from GA_Algorithms import get_algorithm  # uses your factory mapping

DEFAULT_FUNCS = [2100, 2101, 2102, 2103, 2200, 2201, 2202, 2203]
DEFAULT_ALGS = ["RLS", "EA", "GA", "GSEMO", "SOEA", "MOEA"]

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_gsemo_tradeoff(alg_adapter, out_png: Path, title: str = ""):
    """
    Your GSEMO stores archive values as (f1, f2). By construction f2 = -cost,
    so plotting cost vs f1 is simply (-f2, f1). We only plot for the FIRST run.
    """
    inner = getattr(alg_adapter, "inner", None)
    if inner is None:
        return
    vals = getattr(inner, "archive_vals", None)
    if vals is None or len(vals) == 0:
        return
    vals = np.asarray(vals, dtype=float)
    costs = -vals[:, 1]
    f1 = vals[:, 0]

    ensure_dir(out_png.parent)
    plt.figure()
    plt.scatter(costs, f1, s=10)
    plt.xlabel("Cost (#selected nodes)")
    plt.ylabel("Objective value f(x)")
    plt.title(title or "GSEMO Trade-offs")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algs", nargs="+", default=DEFAULT_ALGS,
                        help="Algorithms to run (RLS EA GA GSEMO SOEA MOEA)")
    parser.add_argument("--functions", nargs="+", type=int, default=DEFAULT_FUNCS,
                        help="Function IDs (Graph problems)")
    parser.add_argument("--dim", type=int, default=100, help="Problem dimension (n)")
    parser.add_argument("--reps", type=int, default=30, help="Repetitions per problem")
    parser.add_argument("--budget", type=int, default=100_000, help="Fitness evaluation budget")
    parser.add_argument("--seed", type=int, default=63, help="Base RNG seed")
    parser.add_argument("--pop", type=int, default=20, help="Population size for GA/SOEA/MOEA")
    parser.add_argument("--k", type=int, default=10, help="Uniform constraint K (used by GSEMO & as target_k)")
    parser.add_argument("--outroot", type=str, default="DATA_EX4",
                        help="Output root folder (IOH logs + tradeoffs)")
    args = parser.parse_args()

    outroot = ensure_dir(args.outroot)
    trade_dir = ensure_dir(outroot / "tradeoffs")

    # Create the set of problems (GRAPH type)
    problems = [
        ioh.get_problem(fid=pid, dimension=args.dim, problem_class=ioh.ProblemClass.GRAPH)
        for pid in args.functions
    ]

    base_rng = np.random.default_rng(args.seed)

    for alg_name in args.algs:
        folder = f"run-{alg_name}"
        print(f"\n=== Algorithm: {alg_name} ===")
        analyzer = ioh.logger.Analyzer(
            root=str(outroot),
            folder_name=folder,
            algorithm_name=alg_name,
            algorithm_info=f"Exercise4_{alg_name}",
        )

        try:
            for p in problems:
                p.attach_logger(analyzer)
                try:
                    print(f"- Problem {p.meta_data.problem_id} ({p.meta_data.name}), "
                          f"n={p.meta_data.n_variables}, reps={args.reps}, budget={args.budget}")

                    for rep in range(args.reps):
                        rep_seed = int(base_rng.integers(0, 2**31 - 1))
                        rng = np.random.default_rng(rep_seed)

                        factory = get_algorithm(alg_name)

                        # === adaptive constructor handling ===
                        if alg_name.upper() == "GSEMO":
                            alg = factory(p, args.budget, rng, k=args.k)
                        elif alg_name.upper() in {"SOEA", "MOEA", "GA"}:
                            # Try with positional pop & k
                            try:
                                alg = factory(p, args.budget, rng, args.pop, args.k)
                            except TypeError:
                                # Fallback: some GA classes may only take (problem, budget, rng)
                                alg = factory(p, args.budget, rng)
                        else:
                            alg = factory(p, args.budget, rng)

                        # Run the algorithm
                        alg.run()

                        # Save GSEMO trade-off on the first repetition only
                        if alg_name.upper() == "GSEMO" and rep == 0:
                            out_png = trade_dir / f"{p.meta_data.name}_{p.meta_data.problem_id}_GSEMO_run1.png"
                            save_gsemo_tradeoff(
                                alg_adapter=alg,
                                out_png=out_png,
                                title=f"GSEMO Trade-offs: {p.meta_data.name} ({p.meta_data.problem_id})"
                            )

                        # Reset problem state for next repetition
                        p.reset()

                finally:
                    p.detach_logger()
        finally:
            analyzer.close()

    print("\nDone.")
    print(f"- IOHprofiler logs under: {outroot}/run-*/IOH_data")
    print(f"- GSEMO trade-off plots under: {trade_dir}")

if __name__ == "__main__":
    main()
