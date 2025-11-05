Choose an algorithm to run using the --alg parameter, then specify runs/budget/dimension with the respective flags.

For example:
    - RLS (exercise 1): "python main.py --alg RLS --functions 2100 2101 2102 2103 2200 2201 2202 2203 --dim 100 --budget 10000 --reps 30"
    - GSEMO (exercise 2): "python main.py --alg GSEMO --functions 2100 2101 2102 2103 2200 2201 2202 2203 --dim 100 --budget 10000 --reps 30 --k 10"

The --functions parameter should be followed by the function IDs you want to run (e.g., MaxCoverage instances are 2100, 2101, 2102, 2103)

Note that GSEMO also requires the k parameter (subset size limit). 

Output files are automatically saved under DATA/run-(algorithm)/