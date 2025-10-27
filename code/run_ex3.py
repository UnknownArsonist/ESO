import argparse, numpy as np, ioh, os
from GA_Algorithms import get_algorithm

ROOT = "DATA"

def run_once(alg_name, f_id, n, budget, seed, pop=None, k=10):
    p = ioh.get_problem(f_id, problem_class=ioh.ProblemClass.GRAPH, instance=1, dimension=n)
    rng = np.random.default_rng(seed)
    factory = get_algorithm(alg_name)
    if alg_name.upper() in {"SOEA","MOEA","GA"} and pop is not None:
        alg = factory(p, budget, rng, pop_size=pop, target_k=k)
    elif alg_name.upper() == "GSEMO":
        alg = factory(p, budget, rng, k=k)
    else:
        alg = factory(p, budget, rng)
    exp_name = f"EX3_{alg_name}_f{f_id}_n{n}_k{k}_pop{pop if pop else '-'}"
    lg = ioh.logger.Analyzer(root=ROOT, folder_name=exp_name, algorithm_name=alg_name)
    p.attach_logger(lg)
    alg.run()
    p.detach_logger()
    lg.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algs", nargs="+", default=["SOEA","MOEA","GSEMO"])
    parser.add_argument("--funcs", nargs="+", type=int,
                        default=[2100,2101,2102,2103,2200,2201,2202,2203])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--budget", type=int, default=10_000)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--pops", nargs="+", type=int, default=[10,20,50])
    parser.add_argument("--seed", type=int, default=63)
    args = parser.parse_args()

    for f_id in args.funcs:
        for alg in args.algs:
            for rep in range(args.reps):
                seed = int(np.random.SeedSequence([args.seed, f_id, rep]).generate_state(1)[0])
                if alg.upper() in {"SOEA","MOEA","GA"}:
                    for pop in args.pops:
                        run_once(alg, f_id, args.n, args.budget, seed, pop=pop, k=args.k)
                else:
                    run_once(alg, f_id, args.n, args.budget, seed, pop=None, k=args.k)

    print("Finished Exercise 3 batch runs. Use IOHAnalyzer to create fixed-budget plots.")

if __name__ == "__main__":
    main()
