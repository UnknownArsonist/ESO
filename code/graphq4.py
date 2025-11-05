# graphq4_all_cov_infl.py
# Generates progress plots for ALL MaxCoverage and MaxInfluence problems in Ex4.
# Usage:
#   python3 graphq4_all_cov_infl.py \
#       --ex4-dir "/path/to/DATA/Ex4" \
#       --out "./outputs" \
#       --grid

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ex4-dir",
        default=str(Path(__file__).resolve().parent.parent / "DATA" / "Ex4"),
        help="Path to DATA/Ex4 directory"
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "outputs"),
        help="Output directory for figures and CSV"
    )
    parser.add_argument(
        "--eval-cap",
        type=int,
        default=100_000,
        help="Maximum evaluations to display on progress plots"
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Also produce a single multi-panel overview figure"
    )
    return parser.parse_args()

def find_dat_files(ex4_dir: Path) -> pd.DataFrame:
    """List all IOHProfiler .dat files under DATA/Ex4 with algorithm/run/problem info."""
    recs = []
    for algo_dir in sorted(p for p in ex4_dir.iterdir() if p.is_dir()):
        algo = algo_dir.name
        for run_dir in algo_dir.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith("run-"):
                continue
            for prob_dir in run_dir.iterdir():
                if not prob_dir.is_dir() or not prob_dir.name.startswith("data_"):
                    continue
                for f in prob_dir.iterdir():
                    if f.suffix == ".dat" and not f.name.startswith("._"):
                        recs.append({
                            "algo": algo,
                            "run": run_dir.name,
                            "problem": prob_dir.name,
                            "file": str(f)
                        })
    return pd.DataFrame(recs)

def parse_file(path: str) -> pd.DataFrame:
    """Read IOHProfiler .dat with whitespace delimiter and compute running best-so-far."""
    df = pd.read_csv(path, sep=r"\s+", engine="python", comment="#")
    df.columns = [c.strip() for c in df.columns]
    if "evaluations" not in df.columns or "raw_y" not in df.columns:
        if df.shape[1] == 2:
            df.columns = ["evaluations", "raw_y"]
        else:
            raise ValueError(f"Unexpected columns in {path}: {df.columns.tolist()}")
    df["evaluations"] = pd.to_numeric(df["evaluations"], errors="coerce")
    df["raw_y"] = pd.to_numeric(df["raw_y"], errors="coerce")
    df = df.dropna(subset=["evaluations", "raw_y"]).sort_values("evaluations")
    df["best_so_far"] = df["raw_y"].cummax()
    return df

def plot_progress_for_problem(files_df, problem, out_path, eval_cap):
    """Plot best-so-far vs evaluations for all algorithms on a given problem."""
    series = []
    seen = set()
    for _, row in files_df[files_df["problem"] == problem].iterrows():
        dat = parse_file(row["file"])
        dat = dat[dat["evaluations"] <= eval_cap].copy()
        if len(dat) > 400:
            idx = np.linspace(0, len(dat) - 1, 400, dtype=int)
            dat = dat.iloc[idx]
        label = row["algo"]
        if label in seen:
            label = f"{label} ({row['run']})"
        seen.add(row["algo"])
        series.append((label, dat[["evaluations", "best_so_far"]]))

    if not series:
        return False

    plt.figure()
    for label, df in series:
        plt.plot(df["evaluations"], df["best_so_far"], label=label)
    plt.axvline(10_000, linestyle="--", color="gray")
    plt.title(f"Best-So-Far Progress on {problem}")
    plt.xlabel("Evaluations")
    plt.ylabel("Best-So-Far (raw_y)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True

def plot_grid_overview(files_df, problems, out_path, eval_cap):
    """Optional: single grid figure combining all problems."""
    n = len(problems)
    if n == 0:
        return False
    rows = math.ceil(math.sqrt(n))
    cols = math.ceil(n / rows)

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols+1, 3*rows+1), squeeze=False)
    axes = axes.flatten()

    for ax, problem in zip(axes, problems):
        series = []
        seen = set()
        for _, row in files_df[files_df["problem"] == problem].iterrows():
            dat = parse_file(row["file"])
            dat = dat[dat["evaluations"] <= eval_cap].copy()
            if len(dat) > 300:
                idx = np.linspace(0, len(dat) - 1, 300, dtype=int)
                dat = dat.iloc[idx]
            label = row["algo"]
            if label in seen:
                label = f"{label} ({row['run']})"
            seen.add(row["algo"])
            series.append((label, dat[["evaluations", "best_so_far"]]))

        if series:
            for label, df in series:
                ax.plot(df["evaluations"], df["best_so_far"], label=label)
            ax.axvline(10_000, linestyle="--", color="gray")
            ax.set_title(problem, fontsize=10)
            ax.set_xlabel("evals", fontsize=9)
            ax.set_ylabel("best", fontsize=9)
        else:
            ax.set_title(problem + " (no data)")
            ax.axis("off")

    for i in range(len(problems), len(axes)):
        axes[i].axis("off")

    # Combined legend
    handles_labels = {}
    for ax in axes[:len(problems)]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles_labels[l] = h
    if handles_labels:
        fig.legend(handles_labels.values(), handles_labels.keys(), loc="upper center", ncol=4)

    fig.suptitle("Best-So-Far Progress on All MaxCoverage & MaxInfluence Problems", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close()
    return True

def main():
    args = parse_args()
    ex4_dir = Path(args.ex4_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files_df = find_dat_files(ex4_dir)
    if files_df.empty:
        raise SystemExit(f"No .dat files found under {ex4_dir}")

    # Include BOTH MaxCoverage and MaxInfluence problems
    problems = sorted([
        p for p in files_df["problem"].unique()
        if "MaxCoverage" in p or "MaxInfluence" in p
    ])
    if not problems:
        raise SystemExit("No MaxCoverage or MaxInfluence problems found.")

    # Individual plots
    for problem in problems:
        out_png = out_dir / f"ex4_progress_{problem}.png"
        ok = plot_progress_for_problem(files_df, problem, out_png, args.eval_cap)
        if ok:
            print(f"Saved {out_png}")
        else:
            print(f"Skipped {problem} (no series)")

    # Optional combined overview
    if args.grid:
        grid_png = out_dir / "ex4_progress_CoverageInfluence_ALL.png"
        if plot_grid_overview(files_df, problems, grid_png, args.eval_cap):
            print(f"Saved {grid_png}")

if __name__ == "__main__":
    main()
