## Figure 1: Comparison of different Populations in GSEMO

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_gsemo_performance(file_path, label, color, marker):
    data = pd.read_csv(file_path, sep=r"\s+", comment='#', skiprows=1,
                       names=["evaluations", "raw_y"],
                       dtype={"evaluations": float, "raw_y": float})
    plt.plot(data["evaluations"], data["raw_y"],
             marker=marker, markersize=3, linewidth=1.5,
             color=color, label=label)
    return data

# path
paths = [
    ("../DATA/Multi-objective_ex3/EX3_GSEMO_f2100_n100_k10_pop--1/data_f2101_MaxCoverage2101/IOHprofiler_f2101_DIM450.dat",
     "GSEMO (Population = 10)", "#1f77b4", "o"),
    ("../DATA/Multi-objective_ex3/EX3_GSEMO_f2100_n100_k10_pop--2/data_f2101_MaxCoverage2101/IOHprofiler_f2101_DIM450.dat",
     "GSEMO (Population = 20)", "#ff7f0e", "s"),
    ("../DATA/Multi-objective_ex3/EX3_GSEMO_f2100_n100_k10_pop--4/data_f2101_MaxCoverage2101/IOHprofiler_f2101_DIM450.dat",
     "GSEMO (Population = 50)", "#2ca02c", "^"),
]

# drawing
plt.figure(figsize=(8, 5), dpi=120, facecolor="white")

for path, label, color, marker in paths:
    plot_gsemo_performance(path, label, color, marker)

plt.locator_params(axis="x", nbins=6)
plt.locator_params(axis="y", nbins=6)
plt.tick_params(axis="x", labelrotation=30)

plt.xlabel("Number of Evaluations", fontsize=11)
plt.ylabel("Fitness", fontsize=11)
plt.title("Performance of GSEMO with Different Population Sizes on MaxCoverage (f2100)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()

# save as png
plt.savefig("Fig1_GSEMO_population.png", dpi=300)
plt.show()


## Figure 2: Comparison of Algorithms (GSEMO vs MOEA vs SOEA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_curve(file_path):
    return pd.read_csv(file_path, sep=r"\s+", comment='#', skiprows=1,
                       names=["evaluations", "raw_y"],
                       dtype={"evaluations": float, "raw_y": float})

# path
paths_algorithms = [
    ("../DATA/Multi-objective_ex3/EX3_GSEMO_f2100_n100_k10_pop--1/data_f2101_MaxCoverage2101/IOHprofiler_f2101_DIM450.dat",
     "GSEMO (Pop = 10)", "#1f77b4", "o"),
    ("../DATA/Multi-objective_ex3/EX3_MOEA_f2100_n100_k10_pop10/data_f2101_MaxCoverage2101/IOHprofiler_f2101_DIM450.dat",
     "MOEA (Pop = 10)", "#9467bd", "s"),
    ("../DATA/Multi-objective_ex3/EX3_SOEA_f2100_n100_k10_pop10/data_f2101_MaxCoverage2101/IOHprofiler_f2101_DIM450.dat",
     "SOEA (Pop = 10)", "#d62728", "^"),
]

# drawing
plt.figure(figsize=(8, 5), dpi=120, facecolor="white")

for path, label, color, marker in paths_algorithms:
    df = load_curve(path)
    plt.plot(df["evaluations"], df["raw_y"],
             label=label, color=color, linewidth=1.5,
             marker=marker, markersize=3)

plt.locator_params(axis="x", nbins=6)
plt.locator_params(axis="y", nbins=6)
plt.tick_params(axis="x", labelrotation=30)

plt.xlabel("Number of Evaluations", fontsize=11)
plt.ylabel("Fitness", fontsize=11)
plt.title("Comparison of Algorithms on MaxCoverage (f2100, Population = 10)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()

plt.savefig("Fig2_Algorithm_comparison.png", dpi=300)
plt.show()



## Figure 3: Comparison of Problems (f2100 vs f2200) using GSEMO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_curve(file_path):
    return pd.read_csv(file_path, sep=r"\s+", comment='#', skiprows=1,
                       names=["evaluations", "raw_y"],
                       dtype={"evaluations": float, "raw_y": float})

# path
paths_problems = [
    ("../DATA/Multi-objective_ex3/EX3_GSEMO_f2100_n100_k10_pop--1/data_f2101_MaxCoverage2101/IOHprofiler_f2101_DIM450.dat",
     "GSEMO on f2100", "#1f77b4", "o"),
    ("../DATA/Multi-objective_ex3/EX3_GSEMO_f2200_n100_k10_pop--1/data_f2201_MaxInfluence2201/IOHprofiler_f2201_DIM4039.dat",
     "GSEMO on f2200", "#ff7f0e", "s"),
]

# drawing
plt.figure(figsize=(8, 5), dpi=120, facecolor="white")

for path, label, color, marker in paths_problems:
    df = load_curve(path)
    plt.plot(df["evaluations"], df["raw_y"],
             label=label, color=color, linewidth=1.5,
             marker=marker, markersize=3)

plt.locator_params(axis="x", nbins=6)
plt.locator_params(axis="y", nbins=6)
plt.tick_params(axis="x", labelrotation=30)

plt.xlabel("Number of Evaluations", fontsize=11)
plt.ylabel("Fitness", fontsize=11)
plt.title("Performance of GSEMO on Different Problems (f2100 vs f2200)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()

plt.savefig("Fig3_GSEMO_problems.png", dpi=300)
plt.show()
