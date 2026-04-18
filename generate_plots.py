import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import ast

def plot_tsp_route_from_solution(coord_csv_path, solution_csv_path, output_dir="plots"):
    # Read coordinates (assume ; separator, as in TSPA.csv)
    coords_df = pd.read_csv(coord_csv_path, sep=';', header=None, usecols=[0, 1])
    coords = coords_df.values
    n_points = coords.shape[0]

    # Find the best_solution line and extract the route
    best_route = None
    with open(solution_csv_path, 'r') as f:
        for line in f:
            if line.startswith('best_solution'):
                # Format: best_solution,([idx1, idx2, ...], score)
                try:
                    # Extract the part inside the parentheses
                    tuple_str = line.split(',', 1)[1].strip()
                    route_tuple = ast.literal_eval(tuple_str)
                    route = route_tuple[0]  # The list of indices
                    if isinstance(route, list) and all(isinstance(idx, int) and 0 <= idx < n_points for idx in route):
                        best_route = route
                except Exception as e:
                    print(f"Failed to parse best_solution in {solution_csv_path}: {e}")
                break
    if best_route is None:
        print(f"No valid best_solution found in {solution_csv_path}!")
        return
    solution = np.array(best_route)
    closed_tour = np.append(solution, solution[0])
    ordered_coords = coords[closed_tour]
    x_path = ordered_coords[:, 0]
    y_path = ordered_coords[:, 1]

    plt.figure(figsize=(12, 8))
    plt.plot(x_path, y_path, color='#2c3e50', linewidth=1.5, alpha=0.8, zorder=1, label='Path')
    plt.scatter(coords[:, 0], coords[:, 1], color='#e74c3c', s=30, edgecolors='white', zorder=2)
    plt.scatter(x_path[0], y_path[0], color='#27ae60', s=120, marker='P', label='Start/End', zorder=3)
    plt.title(f"TSP Solution Visualization\n{os.path.basename(solution_csv_path)}", fontsize=14)
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()

    # Save plot to file
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.splitext(os.path.basename(solution_csv_path))[0] + ".png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")

def plot_all_solutions(coord_csv_path, solutions_dir, output_dir="plots"):
    for fname in os.listdir(solutions_dir):
        if fname.endswith('.csv'):
            solution_path = os.path.join(solutions_dir, fname)
            print(f"Plotting: {fname}")
            plot_tsp_route_from_solution(coord_csv_path, solution_path, output_dir=output_dir)

if __name__ == "__main__":
    coord_csv = './input_data/TSPA.csv'
    solutions_dir = './solutions'
    output_dir = './plots'
    plot_all_solutions(coord_csv, solutions_dir, output_dir)