from GA_Algorithms import get_algorithm
import numpy as np
import ioh  # IOHexperimenter
import sys
import argparse

ALGORITHM = sys.argv[1]
ROOT = "DATA"

def main():
    parser = argparse.ArgumentParser(description="via IOH.")
    parser.add_argument("--alg", type=str, nargs="+", default=["RLS","EA","GA"])
    parser.add_argument("--functions", type=int, nargs="+", default=[2100,2101,2102,2103,2200,2201,2202,2203,2300,2301,2302], help="PBO function IDs")
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--budget", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--pop", type=int, default=20)
    args = parser.parse_args()

    for algorithm in args.alg:
        folder = f"run-{algorithm}"
        print(f"Algorithm: {algorithm}")
        print(f"Problems: {args.functions}")
        print(f"n: {args.dim}")
        print(f"Output: {ROOT}/{folder}")
        alg_logger = ioh.logger.Analyzer(
            root=ROOT,
            folder_name=folder,
            algorithm_name=algorithm,
            algorithm_info=f"{algorithm} algorithm",
        )

        # build problems
        problems = [ioh.get_problem(fid=pid, dimension=args.dim, problem_class=ioh.ProblemClass.GRAPH)
                    for pid in args.functions]

        # run the chosen algorithm (each algorithm handles its own runs internally)
        rng = np.random.default_rng(args.seed)
        for p in problems:
            p.attach_logger(alg_logger)
            budget = args.budget
            print(f"Running {algorithm} on problem {p.meta_data.problem_id} (n={args.dim}) | budget={budget}")
            for r in range(args.reps):
                alg = get_algorithm(algorithm)(p, budget, rng)
                alg.run()
                p.reset()

            p.detach_logger()

        alg_logger.close()

    print("\nDone")

if __name__ == "__main__":
    main()