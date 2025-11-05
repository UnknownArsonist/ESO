import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

def get_function_name(function_id):
    if 2100 <= function_id <= 2103:
        return f"MaxCoverage{function_id}"
    elif 2200 <= function_id <= 2203:
        return f"MaxInfluence{function_id}"
    else:
        return f"PackWhileTravel{function_id}"

def get_data_directory(function_id):
    if function_id >= 2300:
        return "DATA/run-GSEMO-1"
    return "DATA/run-GSEMO"

def load_pareto_front(function_id):
    """Load the Pareto front from the first run"""
    func_name = get_function_name(function_id)
    base_dir = get_data_directory(function_id)
    data_dir = f"{base_dir}/data_f{function_id}_{func_name}"
    data_file = os.listdir(data_dir)[0]
    data_path = os.path.join(data_dir, data_file)
    
    # Read raw data file
    with open(data_path, 'r') as dat_file:
        # Skip header line and get data lines
        header = dat_file.readline()  # Read and skip header
        data_lines = [line.strip() for line in dat_file if line.strip()]
    
    # Process data lines
    evaluations = []
    objectives = []
    for line in data_lines:
        try:
            eval_str, obj_str = line.split()
            evaluations.append(int(eval_str))
            objectives.append(float(obj_str))
        except ValueError:
            continue  # Skip lines that can't be parsed
    
    # Extract non-dominated solutions
    solutions = []
    best_obj = float('-inf')
    
    for eval_num, obj_val in zip(evaluations, objectives):
        if obj_val > best_obj:
            best_obj = obj_val
            cost = bin(eval_num).count('1')
            solutions.append((cost, best_obj))
    
    # Sort by cost and remove duplicates
    solutions = list(dict.fromkeys(solutions))
    solutions.sort()
    
    # Return costs and objectives as separate lists
    if solutions:
        costs, objs = zip(*solutions)
        return list(costs), list(objs)
    return [], []
    
def create_tradeoff_plots():
    """Create PDF with trade-off plots for all instances"""
    with PdfPages('DATA/tradeoffs_ex2.pdf') as pdf:
        function_ids = [
            *range(2100, 2104),  # MaxCoverage
            *range(2200, 2204),  # MaxInfluence
            *range(2300, 2303)   # PackWhileTravel
        ]
        
        for fid in function_ids:
            try:
                costs, objectives = load_pareto_front(fid)
                
                plt.figure(figsize=(10, 6))
                plt.plot(costs, objectives, 'bo-', markersize=8, label='Pareto Front')
                plt.title(f'Trade-off Plot for {get_function_name(fid)} (First Run)')
                plt.xlabel('Number of Selected Items (Cost)')
                plt.ylabel('Objective Value')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                
                pdf.savefig()
                plt.close()
                
                print(f"Created trade-off plot for {get_function_name(fid)}")
                
            except Exception as e:
                print(f"Error processing {get_function_name(fid)}: {str(e)}")

if __name__ == "__main__":
    create_tradeoff_plots()